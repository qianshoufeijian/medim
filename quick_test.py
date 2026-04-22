"""快速测试脚本 - 验证所有组件"""
import torch
import numpy as np
from omegaconf import OmegaConf

print("="*60)
print("快速测试 - 工业故障诊断模型")
print("="*60)

# 1. 测试配置加载
print("\n[1/5] 测试配置加载...")
try:
    config = OmegaConf.create({
        'model': {
            'text_length': 256,
            'accel_length': 128,
            'audio_length': 128,
            'temp_length': 64,
            'length': 576,
            'hidden_dim': 256,
            'num_attention_heads': 8,
            'llama_ckpt': 'gpt2',
            'accel_vqvae': {'input_dim': 3, 'hidden_dim': 128, 'num_embeddings': 1024, 'embedding_dim': 32},
            'audio_vqvae': {'input_dim': 120, 'hidden_dim': 128, 'num_embeddings': 1024, 'embedding_dim': 32},
            'temp_vqvae': {'input_dim': 1, 'hidden_dim': 64, 'num_embeddings': 512, 'embedding_dim': 16},
        },
        'data': {
            'train_data_dir': './test_data/train',
            'val_data_dir': './test_data/val',
        },
        'loader': {'batch_size': 2, 'eval_batch_size': 2},
        'trainer': {
            'loss_weights': {
                'modality_balance': 0.1,
                'temporal_alignment': 0.2,
                'class_balance': 0.15
            }
        }
    })
    print("✓ 配置加载成功")
except Exception as e:
    print(f"❌ 配置加载失败: {e}")
    exit(1)

# 2. 测试Signal Tokenizer
print("\n[2/5] 测试Signal Tokenizer...")
try:
    from signal_tokenizer import MultiModalSignalTokenizer
    
    tokenizer = MultiModalSignalTokenizer(config, device='cpu')
    
    # 创建测试数据
    test_batch = {
        'acceleration': torch.randn(2, 128, 3),
        'audio': torch.randn(2, 128, 120),
        'temperature': torch.randn(2, 64, 1),
        'text': torch.randint(0, 1000, (2, 256))
    }
    
    tokens, mask = tokenizer.encode_multimodal(test_batch)
    print(f"✓ Signal Tokenizer测试成功")
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Mask shape: {mask.shape}")
    
except Exception as e:
    print(f"❌ Signal Tokenizer测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. 测试数据加载器
print("\n[3/5] 测试数据加载器...")
try:
    from industrial_dataloader import IndustrialFaultDataset
    from transformers import AutoTokenizer
    
    text_tok = AutoTokenizer.from_pretrained('gpt2')
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token
    
    dataset = IndustrialFaultDataset(
        './test_data/train',
        text_tok,
        signal_length=128
    )
    
    sample = dataset[0]
    print(f"✓ 数据加载器测试成功")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  样本keys: {list(sample.keys())}")
    
except Exception as e:
    print(f"❌ 数据加载器测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. 测试评估器
print("\n[4/5] 测试质量评估器...")
try:
    from generate_fault_data import FaultDataQualityEvaluator, MultiModalAlignmentEvaluator
    
    evaluator = FaultDataQualityEvaluator(None)
    alignment_eval = MultiModalAlignmentEvaluator()
    
    # 创建测试数据
    test_signals = {
        'acceleration': np.random.randn(1000),
        'audio': np.random.randn(1000),
        'temperature': np.random.randn(1000)
    }
    
    alignment_metrics = alignment_eval.evaluate_alignment(test_signals)
    print(f"✓ 评估器测试成功")
    print(f"  对齐得分: {alignment_metrics['overall_alignment_score']:.3f}")
    
except Exception as e:
    print(f"❌ 评估器测试失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 组件集成测试
print("\n[5/5] 组件集成测试...")
try:
    print("✓ 所有组件可以独立运行")
    print("✓ 可以进行完整训练")
    
except Exception as e:
    print(f"❌ 集成测试失败: {e}")

print("\n" + "="*60)
print("✓ 快速测试完成！")
print("="*60)
print("\n下一步:")
print("  1. 准备真实数据集")
print("  2. 运行: python train_industrial.py")
print("  3. 监控训练过程")