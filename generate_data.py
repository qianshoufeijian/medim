"""
正式数据生成脚本
用于从训练好的模型生成平衡的故障数据
"""
import torch
import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt

from signal_tokenizer import MultiModalSignalTokenizer
from model_industrial import IndustrialDiffusion
from generate_fault_data import (
    FaultDataQualityEvaluator, 
    MultiModalAlignmentEvaluator,
    visualize_alignment
)
from industrial_dataloader import IndustrialFaultDataset


class FaultDataGenerator:
    """故障数据生成器"""
    
    def __init__(self, config_path, checkpoint_path):
        print("="*70)
        print("工业故障诊断数据生成系统")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"用户: qianshoufeijian")
        print("="*70)
        
        # 加载配置
        self.config = OmegaConf.load(config_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n✓ 配置加载完成")
        print(f"✓ 使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(self.config.output.save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        print(f"\n正在加载训练好的模型...")
        self.model, self.text_tokenizer, self.signal_tokenizer = self._load_model(checkpoint_path)
        print(f"✓ 模型加载完成")
        
        # 初始化评估器
        self.quality_evaluator = FaultDataQualityEvaluator(None)
        self.alignment_evaluator = MultiModalAlignmentEvaluator()
        
        # 加载参考数据统计
        if self.config.evaluation.compute_fid:
            print(f"\n正在加载参考数据...")
            self.reference_stats = self._load_reference_data()
            print(f"✓ 参考数据加载完成")
    
    def _load_model(self, checkpoint_path):
        """加载训练好的模型"""
        # 加载text tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.llama_ckpt,
            trust_remote_code=True
        )
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token
        
        # 添加特殊token
        special_tokens = {
            "additional_special_tokens": [
                "<bom_accel>", "<eom_accel>",
                "<bom_audio>", "<eom_audio>", 
                "<bom_temp>", "<eom_temp>",
                "<normal>", "<fault>", "<eos>"
            ]
        }
        text_tokenizer.add_special_tokens(special_tokens)
        
        # 初始化signal tokenizer
        signal_tokenizer = MultiModalSignalTokenizer(self.config, device=self.device)
        
        # 如果有保存的signal tokenizer，加载它
        signal_tok_path = self.config.checkpoint.get('signal_tokenizer_path')
        if signal_tok_path and os.path.exists(signal_tok_path):
            signal_tokenizer.load_pretrained(signal_tok_path)
            print(f"  ✓ 加载Signal Tokenizer: {signal_tok_path}")
        
        # 初始化模型
        model = IndustrialDiffusion(
            config=self.config,
            text_tokenizer=text_tokenizer,
            signal_tokenizer=signal_tokenizer,
            device=self.device
        )
        
        # 加载检查点
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.backbone.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.backbone.load_state_dict(checkpoint['state_dict'])
            else:
                model.backbone.load_state_dict(checkpoint)
            print(f"  ✓ 加载模型检查点: {checkpoint_path}")
        else:
            print(f"  ⚠️  检查点不存在，使用随机初始化的模型")
        
        model.eval()
        return model, text_tokenizer, signal_tokenizer
    
    def _load_reference_data(self):
        """加载参考数据用于质量评估"""
        ref_dir = self.config.reference_data.real_data_dir
        num_samples = self.config.reference_data.num_reference_samples
        
        # 创建数据集
        dataset = IndustrialFaultDataset(
            ref_dir,
            self.text_tokenizer,
            signal_length=self.config.model.length
        )
        
        # 采样参考数据
        reference_data = {
            'acceleration': [],
            'audio': [],
            'temperature': []
        }
        
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        for idx in indices:
            sample = dataset[int(idx)]
            reference_data['acceleration'].append(sample['acceleration'].numpy())
            reference_data['audio'].append(sample['audio'].numpy())
            reference_data['temperature'].append(sample['temperature'].numpy())
        
        return reference_data
    
    def generate_for_class(self, fault_class, num_samples):
        """为特定故障类生成数据"""
        print(f"\n{'='*70}")
        print(f"正在生成故障类型: {fault_class}")
        print(f"目标样本数: {num_samples}")
        print(f"{'='*70}")
        
        generated_samples = []
        quality_scores = []
        alignment_scores = []
        rejected_count = 0
        
        # 质量阈值
        quality_threshold = self.config.generation.quality_threshold
        
        # 使用进度条
        pbar = tqdm(total=num_samples, desc=f"生成 {fault_class}")
        
        while len(generated_samples) < num_samples:
            # 批量生成
            batch_size = min(
                self.config.generation.batch_size,
                num_samples - len(generated_samples)
            )
            
            # 构建条件prompt
            prompts = []
            for _ in range(batch_size):
                prompt = self._construct_generation_prompt(fault_class)
                prompts.append(prompt)
            
            # Tokenize
            condition = self.text_tokenizer(
                prompts,
                return_tensors='pt',
                max_length=256,
                padding='max_length',
                truncation=True
            ).to(self.device)
            
            # 生成
            with torch.no_grad():
                generated_tokens = self.model.sample(
                    num_samples=batch_size,
                    condition=condition,
                    guidance_scale=self.config.generation.guidance_scale,
                    num_steps=self.config.generation.num_inference_steps
                )
            
            # 解码为信号
            for i in range(batch_size):
                try:
                    # 获取模态mask
                    modality_mask = self.model._create_default_modality_mask(1)
                    
                    # 解码
                    signals = self.signal_tokenizer.decode_multimodal(
                        generated_tokens[i:i+1],
                        modality_mask
                    )
                    
                    # 转为numpy
                    sample_data = {
                        'acceleration': signals['acceleration'].cpu().numpy().squeeze(),
                        'audio': signals['audio'].cpu().numpy().squeeze(),
                        'temperature': signals['temperature'].cpu().numpy().squeeze(),
                        'label': fault_class,
                        'generated_time': datetime.now().isoformat()
                    }
                    
                    # 质量评估
                    alignment_metrics = self.alignment_evaluator.evaluate_alignment(sample_data)
                    alignment_score = alignment_metrics['overall_alignment_score']
                    
                    # 物理合理性检查
                    physical_valid = self._check_comprehensive_validity(sample_data)
                    
                    # 根据阈值过滤
                    if (alignment_score >= quality_threshold.alignment_score and
                        physical_valid >= quality_threshold.physical_validity):
                        
                        sample_data['quality_metrics'] = {
                            'alignment_score': float(alignment_score),
                            'physical_validity': float(physical_valid),
                        }
                        
                        generated_samples.append(sample_data)
                        quality_scores.append(physical_valid)
                        alignment_scores.append(alignment_score)
                        
                        pbar.update(1)
                    else:
                        rejected_count += 1
                
                except Exception as e:
                    print(f"\n⚠️  样本生成失败: {e}")
                    rejected_count += 1
                    continue
        
        pbar.close()
        
        # 统计信息
        print(f"\n生成完成:")
        print(f"  ✓ 成功生成: {len(generated_samples)} 样本")
        print(f"  ✗ 拒绝样本: {rejected_count} 样本")
        print(f"  📊 平均质量分数: {np.mean(quality_scores):.3f}")
        print(f"  📊 平均对齐分数: {np.mean(alignment_scores):.3f}")
        
        return generated_samples, quality_scores, alignment_scores
    
    def _construct_generation_prompt(self, fault_class):
        """构建生成prompt"""
        # 根据故障类型生成详细描述
        prompts_map = {
            'bearing_outer_fault': "生成轴承外圈故障数据：振动信号包含BPFO特征频率及其谐波，声音信号有调制现象，温度略有上升。",
            'bearing_inner_fault': "生成轴承内圈故障数据：振动信号包含BPFI特征频率，高频共振明显，温度上升较快。",
            'gear_tooth_fault': "生成齿轮断齿故障数据：振动信号在啮合频率处幅值增大，声音信号有周期性冲击，温度异常升高。",
            'shaft_misalignment': "生成轴对中不良故障数据：2倍转频处振动突出，径向振动显著，温度上升。",
            'rotor_unbalance': "生成转子不平衡故障数据：1倍转频振动主导，径向振动为主，相位稳定。",
            'looseness': "生成机械松动故障数据：振动频谱出现多次谐波，非线性特征明显，间歇性冲击。",
        }
        
        return prompts_map.get(fault_class, f"生成{fault_class}故障数据。")
    
    def _check_comprehensive_validity(self, sample_data):
        """综合物理合理性检查"""
        accel = sample_data['acceleration']
        audio = sample_data['audio']
        temp = sample_data['temperature']
        
        validity_score = 0.0
        num_checks = 0
        
        # 1. 幅值范围检查
        if np.max(np.abs(accel)) < 100:  # 加速度合理范围
            validity_score += 1.0
        num_checks += 1
        
        if np.max(np.abs(audio)) <= 1.0:  # 音频归一化范围
            validity_score += 1.0
        num_checks += 1
        
        if -20 <= np.min(temp) <= 150:  # 温度合理范围
            validity_score += 1.0
        num_checks += 1
        
        # 2. 连续性检查
        accel_diff = np.abs(np.diff(accel.flatten()))
        if np.percentile(accel_diff, 95) < 10:  # 无异常突变
            validity_score += 1.0
        num_checks += 1
        
        return validity_score / num_checks
    
    def save_generated_data(self, fault_class, samples):
        """保存生成的数据"""
        class_dir = self.output_dir / fault_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n正在保存数据到: {class_dir}")
        
        for idx, sample in enumerate(tqdm(samples, desc="保存文件")):
            sample_id = f"generated_{idx:04d}"
            
            # 保存信号数据
            np.save(class_dir / f"{sample_id}_accel.npy", sample['acceleration'])
            np.save(class_dir / f"{sample_id}_audio.npy", sample['audio'])
            np.save(class_dir / f"{sample_id}_temp.npy", sample['temperature'])
            
            # 保存元信息
            metadata = {
                'label': sample['label'],
                'generated_time': sample['generated_time'],
                'quality_metrics': sample['quality_metrics']
            }
            with open(class_dir / f"{sample_id}_meta.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 可视化（每10个样本保存一次）
            if self.config.output.save_visualization and idx % 10 == 0:
                vis_path = class_dir / f"{sample_id}_visualization.png"
                visualize_alignment(sample, str(vis_path))
        
        print(f"✓ 数据保存完成")
    
    def generate_quality_report(self, all_results):
        """生成质量评估报告"""
        report_path = self.output_dir / "generation_report.json"
        
        report = {
            'generation_time': datetime.now().isoformat(),
            'config': OmegaConf.to_container(self.config),
            'summary': {},
            'details': {}
        }
        
        total_samples = 0
        total_quality = 0
        total_alignment = 0
        
        for fault_class, (samples, quality_scores, alignment_scores) in all_results.items():
            num_samples = len(samples)
            total_samples += num_samples
            
            avg_quality = float(np.mean(quality_scores))
            avg_alignment = float(np.mean(alignment_scores))
            
            total_quality += avg_quality * num_samples
            total_alignment += avg_alignment * num_samples
            
            report['details'][fault_class] = {
                'num_samples': num_samples,
                'avg_quality_score': avg_quality,
                'avg_alignment_score': avg_alignment,
                'std_quality': float(np.std(quality_scores)),
                'std_alignment': float(np.std(alignment_scores)),
                'min_quality': float(np.min(quality_scores)),
                'max_quality': float(np.max(quality_scores))
            }
        
        report['summary'] = {
            'total_samples_generated': total_samples,
            'overall_avg_quality': total_quality / total_samples if total_samples > 0 else 0,
            'overall_avg_alignment': total_alignment / total_samples if total_samples > 0 else 0,
            'num_fault_classes': len(all_results)
        }
        
        # 保存报告
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*70}")
        print("质量评估报告")
        print(f"{'='*70}")
        print(f"总生成样本数: {report['summary']['total_samples_generated']}")
        print(f"平均质量分数: {report['summary']['overall_avg_quality']:.3f}")
        print(f"平均对齐分数: {report['summary']['overall_avg_alignment']:.3f}")
        print(f"\n详细报告已保存到: {report_path}")
    
    def run(self):
        """执行完整生成流程"""
        print(f"\n{'='*70}")
        print("开始批量生成故障数据")
        print(f"{'='*70}")
        
        target_classes = self.config.generation.target_classes
        num_samples = self.config.generation.num_samples_per_class
        
        all_results = {}
        
        for fault_class in target_classes:
            # 生成数据
            samples, quality_scores, alignment_scores = self.generate_for_class(
                fault_class, 
                num_samples
            )
            
            # 保存数据
            self.save_generated_data(fault_class, samples)
            
            all_results[fault_class] = (samples, quality_scores, alignment_scores)
        
        # 生成报告
        if self.config.output.save_metrics:
            self.generate_quality_report(all_results)
        
        print(f"\n{'='*70}")
        print("✓ 所有数据生成完成！")
        print(f"{'='*70}")
        print(f"输出目录: {self.output_dir}")
        print(f"\n下一步:")
        print(f"  1. 查看生成的数据: {self.output_dir}")
        print(f"  2. 查看质量报告: {self.output_dir}/generation_report.json")
        print(f"  3. 使用生成数据训练下游模型")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="工业故障数据生成")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/generation_config.yaml',
        help='生成配置文件路径'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='训练好的模型检查点路径'
    )
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = FaultDataGenerator(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    # 运行生成
    generator.run()


if __name__ == "__main__":
    main()