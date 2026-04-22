import torch
import sys
import os
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# 添加当前目录到path
sys.path.insert(0, str(Path(__file__).parent))

from industrial_dataloader import IndustrialFaultDataset
from signal_tokenizer import MultiModalSignalTokenizer
from model_industrial import IndustrialDiffusion

def main():
    print("="*60)
    print("工业故障诊断多模态生成模型 - 训练脚本")
    print("="*60)
    
    # 1. 加载配置
    config_path = "configs/industrial_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print("请先创建配置文件！")
        return
    
    config = OmegaConf.load(config_path)
    print(f"\n✓ 加载配置: {config_path}")
    
    # 2. 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 使用设备: {device}")
    
    # 3. 初始化text tokenizer
    print(f"\n正在加载文本tokenizer: {config.model.llama_ckpt}")
    
    # 检查模型路径
    if not os.path.exists(config.model.llama_ckpt):
        print(f"⚠️  LLaMA模型路径不存在: {config.model.llama_ckpt}")
        print("使用GPT2作为备选...")
        config.model.llama_ckpt = "gpt2"
    
    text_tokenizer = AutoTokenizer.from_pretrained(
        config.model.llama_ckpt,
        trust_remote_code=True
    )
    
    # 添加特殊token
    special_tokens = {
        "additional_special_tokens": [
            "<bom_accel>", "<eom_accel>",
            "<bom_audio>", "<eom_audio>", 
            "<bom_temp>", "<eom_temp>",
            "<normal>", "<fault>",
            "<eos>"
        ]
    }
    text_tokenizer.add_special_tokens(special_tokens)
    
    # 确保有pad token
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    
    print(f"✓ 文本tokenizer加载完成 (vocab_size: {len(text_tokenizer)})")
    
    # 4. 初始化signal tokenizer
    print(f"\n正在初始化信号tokenizer...")
    signal_tokenizer = MultiModalSignalTokenizer(config, device=device)
    print(f"✓ 信号tokenizer初始化完成")
    
    # 5. 准备数据集
    print(f"\n正在加载数据集...")
    train_dataset = IndustrialFaultDataset(
        config.data.train_data_dir,
        text_tokenizer,
        signal_length=config.model.length
    )
    
    val_dataset = IndustrialFaultDataset(
        config.data.val_data_dir,
        text_tokenizer,
        signal_length=config.model.length
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.loader.batch_size, 
        shuffle=True,
        num_workers=0  # 设为0避免多进程问题
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.loader.eval_batch_size,
        num_workers=0
    )
    
    print(f"✓ 数据集加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    
    # 6. 初始化模型
    print(f"\n正在初始化模型...")
    model = IndustrialDiffusion(
        config=config,
        text_tokenizer=text_tokenizer,
        signal_tokenizer=signal_tokenizer,
        device=device
    )
    
    print(f"✓ 模型初始化完成")
    
    # 7. 测试一个batch
    print(f"\n测试数据加载...")
    try:
        batch = next(iter(train_loader))
        print(f"✓ Batch测试成功:")
        print(f"  acceleration shape: {batch['acceleration'].shape}")
        print(f"  audio shape: {batch['audio'].shape}")
        print(f"  temperature shape: {batch['temperature'].shape}")
        print(f"  text shape: {batch['text'].shape}")
        
        # 测试信号tokenizer
        print(f"\n测试信号编码...")
        unified_tokens, modality_mask = signal_tokenizer.encode_multimodal(batch)
        print(f"✓ 编码成功:")
        print(f"  unified_tokens shape: {unified_tokens.shape}")
        print(f"  modality_mask shape: {modality_mask.shape}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✓ 所有组件初始化成功！")
    print("="*60)
    
    # 开始训练
    print(f"\n开始训练...")
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision=config.trainer.precision)
    model.set_accelerator(accelerator, None)
    model.init_dataloader(train_loader, val_loader)
    model.train()

    # 训练完成后保存模型
    save_dir = Path(config.checkpoint.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = save_dir / config.checkpoint.best_model_name
    torch.save(
        {
            'model_state_dict': model.backbone.state_dict(),
            'config': dict(config),
        },
        best_model_path
    )
    print(f"\n✓ 模型已保存到: {best_model_path}")

    # 保存 signal tokenizer
    signal_tok_save_dir = config.checkpoint.get('signal_tokenizer_path', str(save_dir / 'signal_tokenizer'))
    signal_tokenizer.save_pretrained(signal_tok_save_dir)
    print(f"✓ Signal Tokenizer 已保存到: {signal_tok_save_dir}")

    print(f"\n下一步:")
    print(f"  运行数据生成: python generate_data.py --config configs/generation_config.yaml --checkpoint {best_model_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()