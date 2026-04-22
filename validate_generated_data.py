"""验证生成数据的质量"""
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def validate_generated_data(data_dir):
    """验证生成的数据"""
    data_dir = Path(data_dir)
    
    print("="*70)
    print("验证生成数据")
    print("="*70)
    
    # 统计各类别
    for fault_class_dir in data_dir.iterdir():
        if not fault_class_dir.is_dir():
            continue
        
        fault_class = fault_class_dir.name
        
        # 统计样本数
        accel_files = list(fault_class_dir.glob("*_accel.npy"))
        num_samples = len(accel_files)
        
        print(f"\n故障类型: {fault_class}")
        print(f"  样本数: {num_samples}")
        
        # 检查数据完整性
        incomplete = 0
        for accel_file in accel_files:
            sample_id = accel_file.stem.replace('_accel', '')
            audio_file = fault_class_dir / f"{sample_id}_audio.npy"
            temp_file = fault_class_dir / f"{sample_id}_temp.npy"
            meta_file = fault_class_dir / f"{sample_id}_meta.json"
            
            if not (audio_file.exists() and temp_file.exists() and meta_file.exists()):
                incomplete += 1
        
        if incomplete > 0:
            print(f"  ⚠️  不完整样本: {incomplete}")
        else:
            print(f"  ✓ 所有样本完整")
        
        # 检查质量分数
        quality_scores = []
        for accel_file in accel_files[:100]:  # 抽样100个
            sample_id = accel_file.stem.replace('_accel', '')
            meta_file = fault_class_dir / f"{sample_id}_meta.json"
            
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    if 'quality_metrics' in meta:
                        quality_scores.append(meta['quality_metrics']['alignment_score'])
        
        if quality_scores:
            print(f"  平均质量分数: {np.mean(quality_scores):.3f}")
            print(f"  最低质量分数: {np.min(quality_scores):.3f}")

if __name__ == "__main__":
    validate_generated_data("./generated_data")