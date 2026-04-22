import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from pathlib import Path


# ============= Base Diffusion class =============

class Diffusion:
    """Base Diffusion class providing the training framework interface."""

    def __init__(self):
        self.accelerator = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None

    def set_accelerator(self, accelerator, optimizer):
        raise NotImplementedError

    def init_dataloader(self, train_loader, val_loader):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError


# ============= Transformer backbone =============

class IndustrialTransformer(nn.Module):
    """
    Multimodal Transformer backbone for industrial fault diagnosis.

    Architecture:
      Token Embedding + Positional Encoding + Modality Embedding
      Pre-norm Transformer Encoder Layers
      Output Projection (weight-tied with Token Embedding)
    """

    def __init__(self, vocab_size, seq_len, hidden_dim=256, num_heads=8,
                 num_layers=6, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        # 0=text, 1=accel, 2=audio, 3=temp
        self.modality_embedding = nn.Embedding(4, hidden_dim)

        self.emb_dropout = nn.Dropout(dropout)
        self.emb_layer_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        # Weight tying
        self.output_proj.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

    def forward(self, input_ids, modality_mask=None, attention_mask=None):
        """
        Args:
            input_ids:      [B, L]
            modality_mask:  [B, L]  0=text 1=accel 2=audio 3=temp
            attention_mask: [B, L]  1=valid  0=padding
        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = input_ids.shape
        device = input_ids.device

        input_ids = input_ids.clamp(0, self.vocab_size - 1)

        tok_emb = self.token_embedding(input_ids)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)

        if modality_mask is not None:
            mod_emb = self.modality_embedding(modality_mask.clamp(0, 3))
        else:
            mod_emb = 0

        x = self.emb_layer_norm(self.emb_dropout(tok_emb + pos_emb + mod_emb))

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_proj(x)
        return logits


# ============= Industrial Diffusion model =============

class IndustrialDiffusion(Diffusion):
    """
    Multimodal generative model for industrial fault diagnosis.

    Based on Masked Diffusion Language Model (MDLM): diffusion is
    performed in a unified discrete token space covering vibration /
    audio / temperature / text modalities.
    """

    def __init__(self, config, text_tokenizer, signal_tokenizer, device="cuda"):
        super().__init__()
        self.config = config
        self.text_tokenizer = text_tokenizer
        self.signal_tokenizer = signal_tokenizer
        self.device = device

        signal_vocab_size = signal_tokenizer.total_vocab_size
        text_vocab_size = len(text_tokenizer)
        total_vocab_size = signal_vocab_size + text_vocab_size

        # Reserve one extra slot for the MASK token (last index)
        self.mask_token_id = total_vocab_size
        total_vocab_size_with_mask = total_vocab_size + 1

        seq_len = config.model.length

        self.backbone = IndustrialTransformer(
            vocab_size=total_vocab_size_with_mask,
            seq_len=seq_len,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_attention_heads,
            num_layers=config.model.get("num_transformer_layers", 6),
            dropout=config.model.get("dropout", 0.1),
        ).to(device)

        print("IndustrialDiffusion initialized:")
        print(f"  Total vocab size : {total_vocab_size_with_mask}")
        print(f"  Sequence length  : {seq_len}")
        print(f"  Backbone params  : {sum(p.numel() for p in self.backbone.parameters()):,}")

    # ============= Training framework interface =============

    def set_accelerator(self, accelerator, optimizer=None):
        """Configure Hugging Face Accelerator and optimizer."""
        self.accelerator = accelerator

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.backbone.parameters(),
                lr=self.config.trainer.learning_rate,
                weight_decay=self.config.trainer.get("weight_decay", 1e-4),
            )
        self.optimizer = optimizer

        prepared = self.accelerator.prepare(self.backbone, self.optimizer)
        self.backbone, self.optimizer = prepared[0], prepared[1]

        if self.train_loader is not None:
            self.train_loader = self.accelerator.prepare(self.train_loader)
        if self.val_loader is not None:
            self.val_loader = self.accelerator.prepare(self.val_loader)

    def init_dataloader(self, train_loader, val_loader):
        """Assign train / val data loaders."""
        if self.accelerator is not None:
            self.train_loader = self.accelerator.prepare(train_loader)
            self.val_loader = self.accelerator.prepare(val_loader)
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader

    def eval(self):
        """Switch backbone to evaluation mode."""
        self.backbone.eval()

    # ============= Main training loop =============

    def train(self):
        """Full training loop with warmup + cosine LR schedule and checkpointing."""
        cfg = self.config
        num_epochs   = cfg.trainer.get("num_epochs", 100)
        save_every   = cfg.trainer.get("save_every", 10)
        eval_every   = cfg.trainer.get("eval_every", 5)
        log_every    = cfg.logging.get("log_every", 50) if hasattr(cfg, "logging") else 50
        warmup_steps = cfg.trainer.get("warmup_steps", 1000)
        grad_clip    = cfg.trainer.get("gradient_clip", 1.0)

        save_dir = Path(cfg.checkpoint.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        total_steps = num_epochs * len(self.train_loader)
        scheduler = self._build_scheduler(total_steps, warmup_steps)
        if self.accelerator is not None:
            scheduler = self.accelerator.prepare(scheduler)

        best_val_loss = float("inf")
        global_step = 0

        print(f"\n{chr(61)*60}")
        print(f"Start training: {num_epochs} epochs, {len(self.train_loader)} steps/epoch")
        print(f"{chr(61)*60}\n")

        for epoch in range(1, num_epochs + 1):
            self.backbone.train()
            epoch_loss = 0.0
            t0 = time.time()

            for step, batch in enumerate(self.train_loader, 1):
                loss = self._train_step(batch)

                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                if self.accelerator is not None:
                    self.accelerator.clip_grad_norm_(self.backbone.parameters(), grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), grad_clip)

                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                if step % log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch:3d} | Step {step:5d}/{len(self.train_loader)}"
                          f" | Loss: {epoch_loss/step:.4f} | LR: {lr:.2e}")

            avg_loss = epoch_loss / len(self.train_loader)
            print(f"\nEpoch {epoch:3d} done | Avg Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")

            if epoch % eval_every == 0:
                val_loss = self._validate()
                print(f"  Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = save_dir / cfg.checkpoint.best_model_name
                    self._save_checkpoint(best_path, epoch, global_step, val_loss)
                    print(f"  New best model saved: {best_path}")

            if epoch % save_every == 0:
                ckpt = save_dir / f"checkpoint_epoch{epoch:04d}.pt"
                self._save_checkpoint(ckpt, epoch, global_step, avg_loss)
                print(f"  Checkpoint saved: {ckpt}")

        print(f"\n{chr(61)*60}")
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")
        print(f"{chr(61)*60}\n")

    def _train_step(self, batch):
        """Single training step: encode -> mask -> forward -> loss."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        unified_tokens, modality_mask = self.signal_tokenizer.encode_multimodal(batch)
        unified_tokens, modality_mask = self._pad_or_trim(unified_tokens, modality_mask)

        noisy, targets, loss_mask = self._apply_masking(unified_tokens)
        logits = self.backbone(noisy, modality_mask=modality_mask)
        return self._compute_loss(logits, targets, loss_mask)

    def _validate(self):
        """Compute average loss on the validation set."""
        self.backbone.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                unified_tokens, modality_mask = self.signal_tokenizer.encode_multimodal(batch)
                unified_tokens, modality_mask = self._pad_or_trim(unified_tokens, modality_mask)
                noisy, targets, loss_mask = self._apply_masking(unified_tokens)
                logits = self.backbone(noisy, modality_mask=modality_mask)
                total_loss += self._compute_loss(logits, targets, loss_mask).item()
                n += 1
        self.backbone.train()
        return total_loss / max(n, 1)

    def _pad_or_trim(self, tokens, modality_mask):
        """Trim or zero-pad token sequences to config.model.length."""
        max_len = self.config.model.length
        L = tokens.shape[1]
        if L > max_len:
            return tokens[:, :max_len], modality_mask[:, :max_len]
        if L < max_len:
            pad = max_len - L
            tokens = F.pad(tokens, (0, pad), value=0)
            modality_mask = F.pad(modality_mask, (0, pad), value=0)
        return tokens, modality_mask

    def _apply_masking(self, tokens, mask_prob=0.15):
        """
        BERT-style random masking:
          80%  -> MASK token
          10%  -> random token
          10%  -> unchanged (still contributes to loss)
        """
        B, L = tokens.shape
        targets = tokens.clone()
        noisy = tokens.clone()

        rand = torch.rand(B, L, device=tokens.device)
        loss_mask = rand < mask_prob

        mask_80 = loss_mask & (torch.rand(B, L, device=tokens.device) < 0.8)
        noisy[mask_80] = self.mask_token_id

        mask_10 = loss_mask & ~mask_80 & (torch.rand(B, L, device=tokens.device) < 0.5)
        rand_ids = torch.randint(0, self.mask_token_id, (B, L), device=tokens.device)
        noisy[mask_10] = rand_ids[mask_10]

        return noisy, targets, loss_mask

    def _compute_loss(self, logits, targets, loss_mask):
        """Cross-entropy loss over masked positions."""
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        masked_logits = logits[loss_mask]
        masked_targets = targets[loss_mask].clamp(0, logits.shape[-1] - 1)
        return F.cross_entropy(masked_logits, masked_targets, reduction="mean")

    def _save_checkpoint(self, path, epoch, step, loss):
        """Save model checkpoint (unwrapped for Accelerate compatibility)."""
        backbone_sd = (
            self.accelerator.unwrap_model(self.backbone).state_dict()
            if self.accelerator is not None
            else self.backbone.state_dict()
        )
        torch.save(
            {
                "model_state_dict": backbone_sd,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "config": dict(self.config),
            },
            path,
        )

    def _build_scheduler(self, total_steps, warmup_steps):
        """Warmup + cosine decay learning rate schedule."""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ============= Generation =============

    def sample(self, num_samples, condition=None, guidance_scale=1.0, num_steps=50):
        """
        Generate a multimodal token sequence via iterative masked decoding.

        Args:
            num_samples:    number of samples to generate
            condition:      dict with "input_ids" (condition tokens), optional
            guidance_scale: > 1 strengthens conditional guidance (classifier-free)
            num_steps:      denoising iterations
        Returns:
            tokens: [num_samples, seq_len]
        """
        self.backbone.eval()
        seq_len = self.config.model.length
        device = self.device

        with torch.no_grad():
            # Start with all-MASK sequence
            tokens = torch.full(
                (num_samples, seq_len), self.mask_token_id,
                dtype=torch.long, device=device,
            )

            # Replace text positions with condition tokens if provided
            text_len = self.config.model.get("text_length", 256)
            if condition is not None and "input_ids" in condition:
                cond_ids = condition["input_ids"][:num_samples]
                cond_len = min(cond_ids.shape[1], text_len)
                tokens[:num_samples, :cond_len] = cond_ids[:, :cond_len]

            modality_mask = self._create_default_modality_mask(num_samples)

            for step in range(num_steps):
                logits = self.backbone(tokens, modality_mask=modality_mask)

                mask_positions = tokens == self.mask_token_id
                if not mask_positions.any():
                    break

                temperature = max(0.1, 1.0 - step / num_steps)
                probs = torch.softmax(logits / temperature, dim=-1)

                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), num_samples=1
                ).view(num_samples, seq_len)

                # Confidence-based decoding: unmask highest-confidence positions
                confidence = probs.max(dim=-1).values
                confidence[~mask_positions] = -1.0

                num_to_unmask = max(1, mask_positions.sum().item() // (num_steps - step))

                flat_conf = confidence.view(-1)
                flat_sampled = sampled.view(-1)
                flat_mask = mask_positions.view(-1)
                masked_indices = flat_mask.nonzero(as_tuple=True)[0]

                if len(masked_indices) > 0:
                    k = min(int(num_to_unmask), len(masked_indices))
                    _, top_idx = flat_conf[masked_indices].topk(k)
                    tokens_flat = tokens.view(-1)
                    tokens_flat[masked_indices[top_idx]] = flat_sampled[masked_indices[top_idx]]
                    tokens = tokens_flat.view(num_samples, seq_len)

        return tokens

    # ============= Helper methods =============

    def _construct_diagnostic_prompt(self, fault_description):
        """Build a structured diagnostic prompt for LLM-based knowledge augmentation."""
        few_shot_examples = """
[Example 1]
Fault: Bearing outer race fault
Analysis:
1. Mechanism: spalling or pitting on the outer raceway; rolling elements cause periodic impact.
2. Signal features:
   - Vibration: prominent BPFO frequency, high-frequency resonance band 2-5 kHz
   - Audio:     significant modulation, sideband spacing equals shaft frequency
   - Temperature: local rise of 2-5 degC
3. Modal correlation: vibration impulse and audio pulse are synchronous; temperature lags ~30-60 s.
4. Severity: moderate, maintenance within 2 weeks recommended.

[Example 2]
Fault: Gear tooth breakage
Analysis:
1. Mechanism: fatigue crack propagation at gear root leading to fracture.
2. Signal features:
   - Vibration: large amplitude at meshing frequency with sidebands
   - Audio:     distinct periodic impact at shaft frequency
   - Temperature: rapid rise of 10-15 degC
3. Modal correlation: each mesh event produces a strong impulse; all three modalities respond synchronously.
4. Severity: critical, immediate shutdown required.
"""
        prompt = f"""{few_shot_examples}

Now, acting as an industrial equipment fault-diagnosis expert with 20 years of experience,
provide a deep analysis of the following fault:

[Fault description]
{fault_description}

[Diagnostic analysis]
Please follow the structured format below:

1. Fault mechanism:
   Physical cause: <describe the physical mechanism>
   Progression: <early stage to severe stage>

2. Multimodal signal features:
   Vibration: time domain / frequency domain / energy distribution
   Audio:     time domain / frequency domain / characteristic frequencies
   Temperature: rise amplitude / rate of change / spatial distribution

3. Modal correlation:
   Temporal relationship: <timing and delays among the three signals>
   Causality: <which signal is primary, which are responses>
   Correlation: <quantified correlation>

4. Diagnostic conclusion:
   Fault type / Severity / Confidence / Maintenance recommendation
"""
        return prompt

    def _create_attention_mask(self, batch, batch_size):
        """Create a 2-D attention mask respecting modality boundaries."""
        input_ids = batch["input_ids"]
        seq_length = input_ids.shape[1]

        attention_mask = torch.ones(batch_size, seq_length, device=self.device)
        pad_token_id = self.text_tokenizer.pad_token_id
        attention_mask = attention_mask * (input_ids != pad_token_id).long()

        if "modality_mask" in batch:
            modality_mask = batch["modality_mask"]
            text_positions  = (modality_mask == 0).nonzero(as_tuple=True)[1]
            accel_positions = (modality_mask == 1).nonzero(as_tuple=True)[1]
            audio_positions = (modality_mask == 2).nonzero(as_tuple=True)[1]
            temp_positions  = (modality_mask == 3).nonzero(as_tuple=True)[1]

            attention_mask_2d = torch.zeros(
                batch_size, seq_length, seq_length, device=self.device
            )
            for b in range(batch_size):
                if len(text_positions) > 0:
                    attention_mask_2d[b, :, text_positions] = 1
                for pos in [accel_positions, audio_positions, temp_positions]:
                    if len(pos) > 0:
                        attention_mask_2d[b, pos[:, None], pos] = 1
                if self.config.model.get("enable_cross_modal_attention", True):
                    sig = torch.cat([accel_positions, audio_positions, temp_positions])
                    if len(sig) > 0:
                        attention_mask_2d[b, sig[:, None], sig] = 1
            return attention_mask_2d

        return attention_mask

    def _compute_modality_specific_loss(self, batch, modality_mask):
        """Compute cross-entropy loss restricted to a given modality."""
        logits  = batch.get("logits")
        targets = batch.get("target_tokens", batch.get("input_ids"))
        if logits is None:
            raise ValueError("batch must contain 'logits'")

        masked_logits  = logits[modality_mask]
        masked_targets = targets[modality_mask]

        if len(masked_targets) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return F.cross_entropy(masked_logits, masked_targets, reduction="mean")

    def _create_default_modality_mask(self, batch_size):
        """
        Create a default modality mask based on configured segment lengths.

        Layout: [text | accel | audio | temp]
        """
        total_length = self.config.model.length
        text_length  = self.config.model.get("text_length",  256)
        accel_length = self.config.model.get("accel_length", 512)
        audio_length = self.config.model.get("audio_length", 512)
        temp_length  = self.config.model.get("temp_length",  256)

        assert text_length + accel_length + audio_length + temp_length == total_length, (
            f"Modality lengths sum ({text_length+accel_length+audio_length+temp_length}) "
            f"!= total length ({total_length})"
        )

        mask = torch.zeros(batch_size, total_length, device=self.device, dtype=torch.long)
        start = text_length
        mask[:, start:start + accel_length] = 1; start += accel_length
        mask[:, start:start + audio_length] = 2; start += audio_length
        mask[:, start:start + temp_length]  = 3
        return mask

    def _split_tokens_by_modality(self, tokens, modality_mask):
        """Split a unified token sequence back into per-modality tensors."""
        mapping = {0: "text", 1: "acceleration", 2: "audio", 3: "temperature"}
        separated = {name: [] for name in mapping.values()}

        for b in range(tokens.shape[0]):
            for mid, name in mapping.items():
                separated[name].append(tokens[b][modality_mask[b] == mid])

        for key in separated:
            if separated[key]:
                max_len = max(t.shape[0] for t in separated[key])
                padded = []
                for t in separated[key]:
                    if t.shape[0] < max_len:
                        pad = torch.full(
                            (max_len - t.shape[0],),
                            self.text_tokenizer.pad_token_id,
                            device=self.device, dtype=t.dtype,
                        )
                        t = torch.cat([t, pad])
                    padded.append(t)
                separated[key] = torch.stack(padded)

        return separated

    def _get_modality_statistics(self, batch):
        """Return per-modality token ratio statistics for a batch."""
        if "modality_mask" not in batch:
            return {}
        m = batch["modality_mask"]
        return {
            "text_ratio":  (m == 0).float().mean().item(),
            "accel_ratio": (m == 1).float().mean().item(),
            "audio_ratio": (m == 2).float().mean().item(),
            "temp_ratio":  (m == 3).float().mean().item(),
        }

    def _validate_batch_structure(self, batch):
        """Validate that a batch has the required fields and consistent shapes."""
        for key in ("input_ids", "modality_mask"):
            if key not in batch:
                raise ValueError(f"Batch missing required field: {key}")

        ids = batch["input_ids"]
        mm  = batch["modality_mask"]
        if ids.shape != mm.shape:
            raise ValueError(f"input_ids {ids.shape} != modality_mask {mm.shape}")

        unique_mods = torch.unique(mm)
        valid_mods  = torch.tensor([0, 1, 2, 3], device=self.device)
        if not all(m in valid_mods for m in unique_mods):
            raise ValueError(f"Invalid modality IDs: {unique_mods.tolist()}")

        max_token_id = self.signal_tokenizer.total_vocab_size + len(self.text_tokenizer)
        if (ids >= max_token_id).any():
            raise ValueError(f"Token ID out of range: max={ids.max()}, vocab={max_token_id}")

        return True

    def get_modality_boundaries(self, batch):
        """Return start / end positions for each modality in the sequence."""
        modality_mask = batch.get("modality_mask", self._create_default_modality_mask(1))
        boundaries = {}
        for mid, name in [(0,"text"),(1,"acceleration"),(2,"audio"),(3,"temperature")]:
            positions = (modality_mask[0] == mid).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                boundaries[name] = {
                    "start":  positions[0].item(),
                    "end":    positions[-1].item() + 1,
                    "length": len(positions),
                }
            else:
                boundaries[name] = {"start": -1, "end": -1, "length": 0}
        return boundaries


# ============= Usage example =============

def example_usage():
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from signal_tokenizer import MultiModalSignalTokenizer

    config = OmegaConf.create({
        "model": {
            "llama_ckpt": "gpt2",
            "hidden_dim": 256,
            "num_attention_heads": 8,
            "length": 576,
            "text_length": 256,
            "accel_length": 128,
            "audio_length": 128,
            "temp_length": 64,
            "enable_cross_modal_attention": True,
            "accel_vqvae": {"input_dim": 3,   "hidden_dim": 128, "num_embeddings": 1024, "embedding_dim": 32},
            "audio_vqvae": {"input_dim": 120, "hidden_dim": 128, "num_embeddings": 1024, "embedding_dim": 32},
            "temp_vqvae":  {"input_dim": 1,   "hidden_dim": 64,  "num_embeddings": 512,  "embedding_dim": 16},
        },
        "trainer": {
            "loss_weights": {
                "modality_balance": 0.1,
                "temporal_alignment": 0.2,
                "class_balance": 0.15,
                "knowledge_consistency": 0.1,
            }
        },
        "checkpoint": {"save_dir": "./outputs", "best_model_name": "best_model.pt"},
    })

    text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    signal_tokenizer = MultiModalSignalTokenizer(config, device="cpu")

    model = IndustrialDiffusion(
        config=config,
        text_tokenizer=text_tokenizer,
        signal_tokenizer=signal_tokenizer,
        device="cpu",
    )

    batch_size = 2
    test_batch = {
        "input_ids":     torch.randint(0, 10000, (batch_size, 576)),
        "modality_mask": model._create_default_modality_mask(batch_size),
    }
    model._validate_batch_structure(test_batch)
    boundaries = model.get_modality_boundaries(test_batch)
    print("Modality boundaries:", boundaries)
    stats = model._get_modality_statistics(test_batch)
    print("Modality statistics:", stats)
    print("\nAll components OK!")


if __name__ == "__main__":
    example_usage()
