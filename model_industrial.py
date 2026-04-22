# ============= 完整实现的辅助方法 =============

class IndustrialDiffusion(Diffusion):
    # ... (前面的代码保持不变) ...
    
    def _construct_diagnostic_prompt(self, fault_description):
        """
        构建诊断prompt，引导LLM生成高质量的领域知识
        
        Prompt工程关键点：
        1. 明确角色定位（专家）
        2. 结构化输出格式
        3. 引导生成多模态特征关联
        4. 包含few-shot示例
        """
        # 添加few-shot示例，提高LLM输出质量
        few_shot_examples = """
【示例1】
故障：轴承外圈故障
分析：
1. 故障机理：外圈滚道表面剥落或点蚀，滚动体通过时产生周期性冲击
2. 信号特征：
   - 振动：BPFO频率明显，高频共振带宽约2-5kHz
   - 声音：调制现象显著，边频带间隔等于转频
   - 温度：局部温升2-5°C
3. 模态关联：振动冲击与声音脉冲同步，温度滞后约30-60秒
4. 严重程度：中度，建议2周内维护

【示例2】
故障：齿轮断齿
分析：
1. 故障机理：齿轮齿根疲劳裂纹扩展导致断裂
2. 信号特征：
   - 振动：啮合频率处出现显著幅值增大和边频带
   - 声音：明显的周期性撞击声，频率等于轴频
   - 温度：急剧升高，可达10-15°C
3. 模态关联：每次啮合产生强冲击，三种模态同步响应
4. 严重程度：严重，需立即停机检修
"""
        
        prompt = f"""{few_shot_examples}

现在，请作为一位拥有20年经验的工业设备故障诊断专家，对以下故障进行深度分析：

【故障描述】
{fault_description}

【诊断分析】
请严格按照以下结构化格式提供分析：

1. 故障机理：
   物理成因：<描述故障的物理机制>
   发展过程：<描述从初期到严重的演变>

2. 多模态信号特征：
   振动特征：
   - 时域：RMS值、峰值因子、波形特点
   - 频域：主要频率成分、谐波、边频带
   - 能量分布：<频段能量占比>
   
   声音特征：
   - 时域：响度变化、冲击特性
   - 频域：频谱包络、调制深度
   - 特征频率：<Hz>
   
   温度特征：
   - 温升幅度：<°C>
   - 变化速率：<°C/分钟>
   - 空间分布：<热点位置>

3. 模态间关联：
   时序关系：<描述三种信号的时间对应关系，包括延迟>
   因果关系：<哪个信号是主因，哪些是响应>
   相关性：<量化描述相关程度>

4. 诊断结论：
   故障类型：<具体故障名称>
   严重程度：<轻微/一般/严重/危险>
   置信度：<0-100%>
   维护建议：<具体措施和时间窗口>
   
请基于专业知识和上述示例，提供详细的结构化分析："""
        
        return prompt
    
    def _create_attention_mask(self, batch, batch_size):
        """
        创建attention mask
        
        Mask规则：
        1. 文本部分：全部可见（causal attention）
        2. 信号部分：根据模态mask创建
        3. 特殊token：全部可见
        4. Padding：mask掉
        """
        input_ids = batch['input_ids']
        seq_length = input_ids.shape[1]
        
        # 基础attention mask（1表示可见，0表示mask）
        attention_mask = torch.ones(batch_size, seq_length, device=self.device)
        
        # 处理padding token
        pad_token_id = self.text_tokenizer.pad_token_id
        padding_mask = (input_ids != pad_token_id).long()
        attention_mask = attention_mask * padding_mask
        
        # 如果有模态mask，进一步处理
        if 'modality_mask' in batch:
            modality_mask = batch['modality_mask']
            
            # 为每个模态创建独立的attention模式
            # 文本模态（0）：causal attention
            # 信号模态（1,2,3）：bidirectional attention within modality
            
            # 获取各模态的位置
            text_positions = (modality_mask == 0).nonzero(as_tuple=True)[1]
            accel_positions = (modality_mask == 1).nonzero(as_tuple=True)[1]
            audio_positions = (modality_mask == 2).nonzero(as_tuple=True)[1]
            temp_positions = (modality_mask == 3).nonzero(as_tuple=True)[1]
            
            # 创建2D attention mask [batch_size, seq_length, seq_length]
            attention_mask_2d = torch.zeros(
                batch_size, seq_length, seq_length,
                device=self.device
            )
            
            for b in range(batch_size):
                # 所有位置可以attend到文本
                if len(text_positions) > 0:
                    attention_mask_2d[b, :, text_positions] = 1
                
                # 加速度模态内部可以互相attend
                if len(accel_positions) > 0:
                    attention_mask_2d[b, accel_positions[:, None], accel_positions] = 1
                
                # 声音模态内部可以互相attend
                if len(audio_positions) > 0:
                    attention_mask_2d[b, audio_positions[:, None], audio_positions] = 1
                
                # 温度模态内部可以互相attend
                if len(temp_positions) > 0:
                    attention_mask_2d[b, temp_positions[:, None], temp_positions] = 1
                
                # 跨模态attention（可选）
                if self.config.model.get('enable_cross_modal_attention', True):
                    # 信号模态之间可以互相attend（建模关联性）
                    signal_positions = torch.cat([accel_positions, audio_positions, temp_positions])
                    if len(signal_positions) > 0:
                        attention_mask_2d[b, signal_positions[:, None], signal_positions] = 1
            
            return attention_mask_2d
        
        return attention_mask
    
    def _compute_modality_specific_loss(self, batch, modality_mask):
        """
        计算特定模态的损失
        
        Args:
            batch: 包含预测和真实值的batch
            modality_mask: 该模态的mask [batch_size, seq_length]
            
        Returns:
            该模态的平均损失
        """
        # 1. 获取模型输出和目标
        if 'logits' in batch:
            logits = batch['logits']  # [batch_size, seq_length, vocab_size]
        else:
            # 如果没有预存的logits，需要重新forward
            raise ValueError("需要在batch中提供logits")
        
        if 'target_tokens' in batch:
            targets = batch['target_tokens']  # [batch_size, seq_length]
        else:
            targets = batch['input_ids']
        
        # 2. 只计算该模态的损失
        # modality_mask: True表示属于该模态
        masked_logits = logits[modality_mask]  # [num_tokens, vocab_size]
        masked_targets = targets[modality_mask]  # [num_tokens]
        
        if len(masked_targets) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 3. 计算交叉熵损失
        loss = F.cross_entropy(
            masked_logits,
            masked_targets,
            reduction='mean'
        )
        
        return loss
    
    def _create_default_modality_mask(self, batch_size):
        """
        创建默认的模态mask
        
        默认序列结构：
        [文本: 0-255] [加速度: 256-767] [声音: 768-1279] [温度: 1280-1535]
        """
        total_length = self.config.model.length
        
        # 根据配置确定各模态的长度
        text_length = self.config.model.get('text_length', 256)
        accel_length = self.config.model.get('accel_length', 512)
        audio_length = self.config.model.get('audio_length', 512)
        temp_length = self.config.model.get('temp_length', 256)
        
        # 验证长度
        assert text_length + accel_length + audio_length + temp_length == total_length, \
            f"模态长度之和({text_length + accel_length + audio_length + temp_length})不等于总长度({total_length})"
        
        # 创建mask
        modality_mask = torch.zeros(batch_size, total_length, device=self.device, dtype=torch.long)
        
        # 分配模态ID
        # 0: 文本
        # 1: 加速度
        # 2: 声音
        # 3: 温度
        
        start_idx = 0
        
        # 文本区域
        modality_mask[:, start_idx:start_idx + text_length] = 0
        start_idx += text_length
        
        # 加速度区域
        modality_mask[:, start_idx:start_idx + accel_length] = 1
        start_idx += accel_length
        
        # 声音区域
        modality_mask[:, start_idx:start_idx + audio_length] = 2
        start_idx += audio_length
        
        # 温度区域
        modality_mask[:, start_idx:start_idx + temp_length] = 3
        
        return modality_mask
    
    def _split_tokens_by_modality(self, tokens, modality_mask):
        """
        根据模态mask分离tokens
        
        Args:
            tokens: [batch_size, seq_length]
            modality_mask: [batch_size, seq_length]
            
        Returns:
            dict: 各模态的tokens
        """
        batch_size = tokens.shape[0]
        
        separated_tokens = {
            'text': [],
            'acceleration': [],
            'audio': [],
            'temperature': []
        }
        
        modality_mapping = {
            0: 'text',
            1: 'acceleration',
            2: 'audio',
            3: 'temperature'
        }
        
        for b in range(batch_size):
            for modality_id, modality_name in modality_mapping.items():
                mask = modality_mask[b] == modality_id
                modality_tokens = tokens[b][mask]
                separated_tokens[modality_name].append(modality_tokens)
        
        # 堆叠成tensor
        for key in separated_tokens:
            if len(separated_tokens[key]) > 0:
                # 需要padding到相同长度
                max_len = max(t.shape[0] for t in separated_tokens[key])
                padded_tokens = []
                for t in separated_tokens[key]:
                    if t.shape[0] < max_len:
                        padding = torch.full(
                            (max_len - t.shape[0],),
                            self.text_tokenizer.pad_token_id,
                            device=self.device,
                            dtype=t.dtype
                        )
                        t = torch.cat([t, padding])
                    padded_tokens.append(t)
                separated_tokens[key] = torch.stack(padded_tokens)
        
        return separated_tokens
    
    def _get_modality_statistics(self, batch):
        """
        获取batch中各模态的统计信息
        
        用于监控训练过程中各模态的分布
        """
        if 'modality_mask' not in batch:
            return {}
        
        modality_mask = batch['modality_mask']
        
        stats = {
            'text_ratio': (modality_mask == 0).float().mean().item(),
            'accel_ratio': (modality_mask == 1).float().mean().item(),
            'audio_ratio': (modality_mask == 2).float().mean().item(),
            'temp_ratio': (modality_mask == 3).float().mean().item(),
        }
        
        return stats
    
    def _validate_batch_structure(self, batch):
        """
        验证batch结构的正确性
        
        检查项：
        1. 必要字段存在
        2. 张量形状匹配
        3. 模态mask有效
        4. token ID在有效范围内
        """
        required_keys = ['input_ids', 'modality_mask']
        
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Batch缺少必要字段: {key}")
        
        input_ids = batch['input_ids']
        modality_mask = batch['modality_mask']
        
        # 检查形状匹配
        if input_ids.shape != modality_mask.shape:
            raise ValueError(
                f"input_ids shape {input_ids.shape} 与 "
                f"modality_mask shape {modality_mask.shape} 不匹配"
            )
        
        # 检查模态ID有效性
        unique_modalities = torch.unique(modality_mask)
        valid_modalities = torch.tensor([0, 1, 2, 3], device=self.device)
        
        if not all(m in valid_modalities for m in unique_modalities):
            raise ValueError(f"发现无效的模态ID: {unique_modalities.tolist()}")
        
        # 检查token ID范围
        max_token_id = self.signal_tokenizer.total_vocab_size + len(self.text_tokenizer)
        if (input_ids >= max_token_id).any():
            raise ValueError(
                f"发现超出词汇表范围的token ID: "
                f"max={input_ids.max()}, vocab_size={max_token_id}"
            )
        
        return True
    
    def get_modality_boundaries(self, batch):
        """
        获取各模态在序列中的边界位置
        
        Returns:
            dict: 各模态的起始和结束位置
        """
        modality_mask = batch.get('modality_mask', self._create_default_modality_mask(1))
        
        boundaries = {}
        
        for modality_id, modality_name in [(0, 'text'), (1, 'acceleration'), 
                                            (2, 'audio'), (3, 'temperature')]:
            positions = (modality_mask[0] == modality_id).nonzero(as_tuple=True)[0]
            
            if len(positions) > 0:
                boundaries[modality_name] = {
                    'start': positions[0].item(),
                    'end': positions[-1].item() + 1,
                    'length': len(positions)
                }
            else:
                boundaries[modality_name] = {
                    'start': -1,
                    'end': -1,
                    'length': 0
                }
        
        return boundaries


# ============= 使用示例 =============

def example_usage():
    """完整的使用示例"""
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from signal_tokenizer import MultiModalSignalTokenizer
    
    # 1. 加载配置
    config = OmegaConf.create({
        'model': {
            'llama_ckpt': './models/Llama-2-7b-hf',
            'hidden_dim': 768,
            'num_attention_heads': 12,
            'length': 1536,
            'text_length': 256,
            'accel_length': 512,
            'audio_length': 512,
            'temp_length': 256,
            'enable_cross_modal_attention': True
        },
        'trainer': {
            'loss_weights': {
                'modality_balance': 0.1,
                'temporal_alignment': 0.2,
                'class_balance': 0.15,
                'knowledge_consistency': 0.1
            }
        }
    })
    
    # 2. 初始化tokenizers
    text_tokenizer = AutoTokenizer.from_pretrained(config.model.llama_ckpt)
    signal_tokenizer = MultiModalSignalTokenizer(config)
    
    # 3. 创建模型
    model = IndustrialDiffusion(
        config=config,
        text_tokenizer=text_tokenizer,
        signal_tokenizer=signal_tokenizer,
        device='cuda'
    )
    
    # 4. 准备测试数据
    batch_size = 2
    test_batch = {
        'input_ids': torch.randint(0, 10000, (batch_size, 1536), device='cuda'),
        'text': torch.randint(0, 32000, (batch_size, 256), device='cuda'),
        'modality_mask': model._create_default_modality_mask(batch_size)
    }
    
    # 5. 验证batch
    model._validate_batch_structure(test_batch)
    
    # 6. 获取模态边界
    boundaries = model.get_modality_boundaries(test_batch)
    print("模态边界:", boundaries)
    
    # 7. 创建attention mask
    attention_mask = model._create_attention_mask(test_batch, batch_size)
    print("Attention mask shape:", attention_mask.shape)
    
    # 8. 获取统计信息
    stats = model._get_modality_statistics(test_batch)
    print("模态统计:", stats)
    
    print("\n✓ 所有组件测试通过!")


if __name__ == "__main__":
    example_usage()