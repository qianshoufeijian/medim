import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VectorQuantizer(nn.Module):
    """矢量量化层"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z):
        # z: [B, D, T]
        z = z.permute(0, 2, 1).contiguous()  # [B, T, D]
        flat_z = z.view(-1, self.embedding_dim)  # [B*T, D]
        
        # 计算距离并找到最近的codebook entry
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) + 
                    torch.sum(self.embedding.weight**2, dim=1) - 
                    2 * torch.matmul(flat_z, self.embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1)  # [B*T]
        
        # 量化
        z_q = self.embedding(encoding_indices).view(z.shape)
        
        # VQ loss
        e_latent_loss = F.mse_loss(z_q.detach(), z)
        q_latent_loss = F.mse_loss(z_q, z.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 2, 1).contiguous()  # [B, D, T]
        
        return z_q, encoding_indices, vq_loss
    
    def get_codebook_entry(self, indices):
        """从indices获取codebook entry"""
        z_q = self.embedding(indices)
        if z_q.dim() == 2:
            z_q = z_q.unsqueeze(0)
        return z_q.permute(0, 2, 1).contiguous()


class SignalVQVAE(nn.Module):
    """信号VQ-VAE"""
    def __init__(self, input_dim, hidden_dim=256, num_embeddings=4096, embedding_dim=64):
        super().__init__()
        self.num_embeddings = num_embeddings
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size=3, padding=1)
        )
        
        # 矢量量化层
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        )
    
    def encode(self, x):
        """编码为离散token"""
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [B, T] -> [B, 1, T]
        if x.dim() == 3 and x.shape[1] != self.encoder[0].in_channels:
            x = x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
        
        z = self.encoder(x)
        z_q, indices, vq_loss = self.vq_layer(z)
        return indices, vq_loss
    
    def decode(self, indices):
        """从token重构信号"""
        z_q = self.vq_layer.get_codebook_entry(indices)
        x_recon = self.decoder(z_q)
        return x_recon
    
    def forward(self, x):
        """完整的前向传播"""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3 and x.shape[1] != self.encoder[0].in_channels:
            x = x.permute(0, 2, 1)
            
        z = self.encoder(x)
        z_q, indices, vq_loss = self.vq_layer(z)
        x_recon = self.decoder(z_q)
        return x_recon, indices, vq_loss


class MultiModalSignalTokenizer:
    """统一的多模态信号tokenizer"""
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        
        # 为每种模态创建VQ-VAE
        self.accel_vqvae = SignalVQVAE(
            input_dim=config.model.accel_vqvae.input_dim,
            hidden_dim=config.model.accel_vqvae.hidden_dim,
            num_embeddings=config.model.accel_vqvae.num_embeddings,
            embedding_dim=config.model.accel_vqvae.embedding_dim
        ).to(device)
        
        self.audio_vqvae = SignalVQVAE(
            input_dim=config.model.audio_vqvae.input_dim,
            hidden_dim=config.model.audio_vqvae.hidden_dim,
            num_embeddings=config.model.audio_vqvae.num_embeddings,
            embedding_dim=config.model.audio_vqvae.embedding_dim
        ).to(device)
        
        self.temp_vqvae = SignalVQVAE(
            input_dim=config.model.temp_vqvae.input_dim,
            hidden_dim=config.model.temp_vqvae.hidden_dim,
            num_embeddings=config.model.temp_vqvae.num_embeddings,
            embedding_dim=config.model.temp_vqvae.embedding_dim
        ).to(device)
        
        # Token偏移量
        self.accel_offset = 0
        self.audio_offset = config.model.accel_vqvae.num_embeddings
        self.temp_offset = self.audio_offset + config.model.audio_vqvae.num_embeddings
        self.text_offset = self.temp_offset + config.model.temp_vqvae.num_embeddings
        
        self.total_vocab_size = self.text_offset
        
        print(f"MultiModalSignalTokenizer initialized:")
        print(f"  Accel vocab: [0, {self.audio_offset})")
        print(f"  Audio vocab: [{self.audio_offset}, {self.temp_offset})")
        print(f"  Temp vocab: [{self.temp_offset}, {self.text_offset})")
        print(f"  Total signal vocab size: {self.total_vocab_size}")
    
    def encode_multimodal(self, batch):
        """编码多模态信号"""
        device = self.device
        
        # 提取各模态数据
        accel = batch['acceleration'].to(device)
        audio = batch['audio'].to(device)
        temp = batch['temperature'].to(device)
        
        # 编码
        accel_tokens, _ = self.accel_vqvae.encode(accel)
        audio_tokens, _ = self.audio_vqvae.encode(audio)
        temp_tokens, _ = self.temp_vqvae.encode(temp)
        
        # 添加偏移
        accel_tokens = accel_tokens + self.accel_offset
        audio_tokens = audio_tokens + self.audio_offset
        temp_tokens = temp_tokens + self.temp_offset
        
        # 获取文本tokens
        text_tokens = batch['text'].to(device) + self.text_offset
        
        # 拼接
        batch_size = accel.shape[0]
        unified_tokens = torch.cat([
            text_tokens,
            accel_tokens.view(batch_size, -1),
            audio_tokens.view(batch_size, -1),
            temp_tokens.view(batch_size, -1)
        ], dim=1)
        
        # 创建模态mask
        modality_mask = self._create_modality_mask(
            batch_size,
            text_tokens.shape[1],
            accel_tokens.shape[0] if accel_tokens.dim() == 1 else accel_tokens.shape[1],
            audio_tokens.shape[0] if audio_tokens.dim() == 1 else audio_tokens.shape[1],
            temp_tokens.shape[0] if temp_tokens.dim() == 1 else temp_tokens.shape[1]
        )
        
        return unified_tokens, modality_mask
    
    def _create_modality_mask(self, batch_size, text_len, accel_len, audio_len, temp_len):
        """创建模态mask"""
        total_len = text_len + accel_len + audio_len + temp_len
        mask = torch.zeros(batch_size, total_len, dtype=torch.long, device=self.device)
        
        start = 0
        # 文本: 0
        mask[:, start:start+text_len] = 0
        start += text_len
        
        # 加速度: 1
        mask[:, start:start+accel_len] = 1
        start += accel_len
        
        # 声音: 2
        mask[:, start:start+audio_len] = 2
        start += audio_len
        
        # 温度: 3
        mask[:, start:start+temp_len] = 3
        
        return mask
    
    def decode_multimodal(self, tokens, modality_mask):
        """解码多模态tokens"""
        # 分离各模态
        text_tokens = tokens[modality_mask == 0] - self.text_offset
        accel_tokens = tokens[modality_mask == 1] - self.accel_offset
        audio_tokens = tokens[modality_mask == 2] - self.audio_offset
        temp_tokens = tokens[modality_mask == 3] - self.temp_offset
        
        # 解码
        accel_signal = self.accel_vqvae.decode(accel_tokens)
        audio_signal = self.audio_vqvae.decode(audio_tokens)
        temp_signal = self.temp_vqvae.decode(temp_tokens)
        
        return {
            'acceleration': accel_signal,
            'audio': audio_signal,
            'temperature': temp_signal,
            'text': text_tokens
        }
    
    def save_pretrained(self, save_dir):
        """保存tokenizer"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.accel_vqvae.state_dict(), f"{save_dir}/accel_vqvae.pt")
        torch.save(self.audio_vqvae.state_dict(), f"{save_dir}/audio_vqvae.pt")
        torch.save(self.temp_vqvae.state_dict(), f"{save_dir}/temp_vqvae.pt")
        
        print(f"Signal tokenizer saved to {save_dir}")
    
    def load_pretrained(self, save_dir):
        """加载tokenizer"""
        self.accel_vqvae.load_state_dict(torch.load(f"{save_dir}/accel_vqvae.pt"))
        self.audio_vqvae.load_state_dict(torch.load(f"{save_dir}/audio_vqvae.pt"))
        self.temp_vqvae.load_state_dict(torch.load(f"{save_dir}/temp_vqvae.pt"))
        
        print(f"Signal tokenizer loaded from {save_dir}")