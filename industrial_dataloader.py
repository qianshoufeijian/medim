import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
import json
import os
from pathlib import Path

class IndustrialFaultDataset(Dataset):
    """工业故障诊断多模态数据集"""
    
    # 故障描述模板（同前面的代码）
    FAULT_DESCRIPTIONS = {
        'normal': {
            'templates': [
                "设备运行正常，各项指标平稳，无异常振动或噪声。",
            ],
            'severity': 0
        },
        'bearing_outer_fault': {
            'templates': [
                "轴承外圈故障，特征频率为{bpfo:.2f}Hz，伴随周期性冲击振动。",
            ],
            'severity': 3,
            'feature_freq': lambda rpm, nb=9: rpm / 60 * nb * 0.4
        },
        # ... 其他故障类型
    }
    
    def __init__(self, data_dir, tokenizer, signal_length=1024, 
                 accel_sr=10000, audio_sr=16000, temp_sr=100,
                 augmentation=False):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.signal_length = signal_length
        self.accel_sr = accel_sr
        self.audio_sr = audio_sr
        self.temp_sr = temp_sr
        self.augmentation = augmentation
        
        # 检查数据目录
        if not self.data_dir.exists():
            print(f"⚠️  数据目录不存在: {self.data_dir}")
            print(f"创建示例数据...")
            self._create_dummy_data()
        
        self.samples = self._load_samples()
        self.stats = self._compute_or_load_stats()
        
        print(f"✓ 加载了 {len(self.samples)} 个样本")
        self._print_class_distribution()
    
    def _create_dummy_data(self):
        """创建示例数据用于测试"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建两个类别
        for fault_type in ['normal', 'bearing_outer_fault']:
            class_dir = self.data_dir / fault_type
            class_dir.mkdir(exist_ok=True)
            
            # 每个类别创建10个样本
            for i in range(10):
                sample_id = f"sample_{i:03d}"
                
                # 生成随机信号
                accel = np.random.randn(self.signal_length, 3).astype(np.float32)
                audio = np.random.randn(self.signal_length).astype(np.float32)
                temp = np.random.randn(self.signal_length, 1).astype(np.float32) + 50
                
                # 保存
                np.save(class_dir / f"{sample_id}_accel.npy", accel)
                np.save(class_dir / f"{sample_id}_audio.npy", audio)
                np.save(class_dir / f"{sample_id}_temp.npy", temp)
                
                # 保存元信息
                metadata = {'rpm': 1800, 'load': 50}
                with open(class_dir / f"{sample_id}_meta.json", 'w') as f:
                    json.dump(metadata, f)
        
        print(f"✓ 创建了示例数据在 {self.data_dir}")
    
    def _load_samples(self):
        """加载样本路径"""
        samples = []
        
        for fault_class_dir in self.data_dir.iterdir():
            if not fault_class_dir.is_dir():
                continue
            
            fault_class = fault_class_dir.name
            
            # 获取样本组
            sample_files = {}
            for file_path in fault_class_dir.glob("*.npy"):
                parts = file_path.stem.split('_')
                sample_id = '_'.join(parts[:-1])
                modality = parts[-1]
                
                if sample_id not in sample_files:
                    sample_files[sample_id] = {}
                
                sample_files[sample_id][modality] = file_path
            
            # 检查完整性
            for sample_id, files in sample_files.items():
                if all(mod in files for mod in ['accel', 'audio', 'temp']):
                    meta_path = fault_class_dir / f"{sample_id}_meta.json"
                    if meta_path.exists():
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = {'rpm': 1800, 'load': 50}
                    
                    samples.append({
                        'accel_path': files['accel'],
                        'audio_path': files['audio'],
                        'temp_path': files['temp'],
                        'label': fault_class,
                        'metadata': metadata,
                        'sample_id': sample_id
                    })
        
        return samples
    
    def _compute_or_load_stats(self):
        """计算统计信息"""
        stats_file = self.data_dir / 'dataset_stats.json'
        
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        
        # 简单统计
        stats = {
            'accel': {'mean': 0.0, 'std': 1.0, 'min': -10.0, 'max': 10.0},
            'audio': {'mean': 0.0, 'std': 1.0, 'min': -1.0, 'max': 1.0},
            'temp': {'mean': 50.0, 'std': 10.0, 'min': 20.0, 'max': 100.0}
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        
        return stats
    
    def _print_class_distribution(self):
        """打印类别分布"""
        class_counts = {}
        for sample in self.samples:
            label = sample['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print("\n类别分布:")
        for label, count in sorted(class_counts.items()):
            print(f"  {label}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载信号
        accel_signal = np.load(sample['accel_path'])
        audio_signal = np.load(sample['audio_path'])
        temp_signal = np.load(sample['temp_path'])
        
        # 预处理
        accel_features = self._preprocess_acceleration(accel_signal)
        audio_features = self._preprocess_audio(audio_signal)
        temp_features = self._preprocess_temperature(temp_signal)
        
        # 生成描述
        fault_description = self._generate_fault_description(
            sample['label'], 
            sample['metadata'],
            accel_signal,
            temp_signal
        )
        
        # Tokenize
        text_tokens = self.tokenizer(
            fault_description,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).input_ids.squeeze(0)
        
        # 创建mask
        text_mask = (text_tokens != self.tokenizer.pad_token_id).long()
        
        return {
            'acceleration': torch.FloatTensor(accel_features),
            'audio': torch.FloatTensor(audio_features),
            'temperature': torch.FloatTensor(temp_features),
            'text': text_tokens,
            'mask': text_mask,
            'label': sample['label'],
            'metadata': sample['metadata']
        }
    
    def _preprocess_acceleration(self, signal):
        """加速度预处理"""
        # 去直流
        signal = signal - np.mean(signal, axis=0, keepdims=True)
        
        # 归一化
        signal = (signal - self.stats['accel']['mean']) / (self.stats['accel']['std'] + 1e-8)
        
        # 重采样
        if len(signal) != self.signal_length:
            if signal.ndim == 1:
                f = interp1d(np.linspace(0, 1, len(signal)), signal, kind='linear')
                signal = f(np.linspace(0, 1, self.signal_length))
            else:
                resampled = []
                for i in range(signal.shape[1]):
                    f = interp1d(np.linspace(0, 1, len(signal)), signal[:, i], kind='linear')
                    resampled.append(f(np.linspace(0, 1, self.signal_length)))
                signal = np.array(resampled).T
        
        return signal
    
    def _preprocess_audio(self, signal):
        """音频预处理"""
        # 预加重
        emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=emphasized, sr=self.audio_sr, n_mfcc=40, n_fft=512, hop_length=128)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2]).T
        
        # 归一化
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # 重采样
        if len(features) != self.signal_length:
            resampled = []
            for i in range(features.shape[1]):
                f = interp1d(np.linspace(0, 1, len(features)), features[:, i], kind='linear')
                resampled.append(f(np.linspace(0, 1, self.signal_length)))
            features = np.array(resampled).T
        
        return features
    
    def _preprocess_temperature(self, signal):
        """温度预处理"""
        # 平滑
        signal = np.convolve(signal.flatten(), np.ones(5)/5, mode='same')
        
        # 归一化
        signal = (signal - self.stats['temp']['mean']) / (self.stats['temp']['std'] + 1e-8)
        
        # 重采样
        if len(signal) != self.signal_length:
            f = interp1d(np.linspace(0, 1, len(signal)), signal, kind='linear')
            signal = f(np.linspace(0, 1, self.signal_length))
        
        return signal[:, np.newaxis]
    
    def _generate_fault_description(self, fault_type, metadata, accel_signal, temp_signal):
        """生成故障描述"""
        if fault_type not in self.FAULT_DESCRIPTIONS:
            return f"设备状态：{fault_type}"
        
        fault_info = self.FAULT_DESCRIPTIONS[fault_type]
        template = np.random.choice(fault_info['templates'])
        
        rpm = metadata.get('rpm', 1800)
        format_params = {'rpm': rpm}
        
        if 'feature_freq' in fault_info:
            if fault_type == 'bearing_outer_fault':
                format_params['bpfo'] = fault_info['feature_freq'](rpm)
        
        try:
            description = template.format(**format_params)
        except KeyError:
            description = template
        
        severity = ['正常', '轻微', '一般', '严重', '危险'][fault_info['severity']]
        return f"[故障诊断] {description} 严重程度：{severity}。"