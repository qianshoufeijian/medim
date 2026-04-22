import torch
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from dtaidistance import dtw
import torch.nn.functional as F

class FaultDataQualityEvaluator:
    """故障数据质量评估器"""
    
    def __init__(self, real_data_stats):
        """
        Args:
            real_data_stats: 真实数据的统计特征
        """
        self.real_stats = real_data_stats
    
    def evaluate_generation_quality(self, generated_data, real_data):
        """
        评判生成数据质量的综合指标
        
        评判标准：
        1. 频域特征相似度（FFT）
        2. 时域统计特征相似度
        3. Wasserstein距离（分布差异）
        4. Frechet Distance（FID for signals）
        5. 物理合理性检验
        """
        metrics = {}
        
        for modality in ['acceleration', 'audio', 'temperature']:
            gen_signals = generated_data[modality]
            real_signals = real_data[modality]
            
            # 1. 频域相似度
            freq_similarity = self._compute_frequency_similarity(gen_signals, real_signals)
            
            # 2. 时域统计特征
            stat_similarity = self._compute_statistical_similarity(gen_signals, real_signals)
            
            # 3. Wasserstein距离
            wasserstein_dist = self._compute_wasserstein_distance(gen_signals, real_signals)
            
            # 4. Frechet距离
            frechet_dist = self._compute_frechet_distance(gen_signals, real_signals)
            
            # 5. 物理合理性
            physical_validity = self._check_physical_validity(gen_signals, modality)
            
            metrics[modality] = {
                'frequency_similarity': freq_similarity,
                'statistical_similarity': stat_similarity,
                'wasserstein_distance': wasserstein_dist,
                'frechet_distance': frechet_dist,
                'physical_validity': physical_validity,
                'overall_score': self._compute_overall_score(
                    freq_similarity, stat_similarity, 
                    wasserstein_dist, frechet_dist, physical_validity
                )
            }
        
        return metrics
    
    def _compute_frequency_similarity(self, gen_signals, real_signals):
        """
        频域特征相似度
        使用功率谱密度(PSD)进行比较
        """
        # 计算功率谱密度
        gen_psd = self._compute_psd(gen_signals)
        real_psd = self._compute_psd(real_signals)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(
            torch.tensor(gen_psd), 
            torch.tensor(real_psd), 
            dim=0
        ).item()
        
        return similarity
    
    def _compute_psd(self, signals):
        """计算平均功率谱密度"""
        psds = []
        for sig in signals:
            freqs, psd = scipy_signal.welch(sig, fs=1000, nperseg=256)
            psds.append(psd)
        return np.mean(psds, axis=0)
    
    def _compute_statistical_similarity(self, gen_signals, real_signals):
        """
        时域统计特征相似度
        包括：均值、方差、偏度、峰度、RMS、峰值因子等
        """
        gen_features = self._extract_statistical_features(gen_signals)
        real_features = self._extract_statistical_features(real_signals)
        
        # 计算特征向量的欧氏距离（归一化）
        diff = np.abs(gen_features - real_features) / (np.abs(real_features) + 1e-8)
        similarity = 1.0 / (1.0 + np.mean(diff))
        
        return similarity
    
    def _extract_statistical_features(self, signals):
        """提取统计特征"""
        features = []
        for sig in signals:
            features.append([
                np.mean(sig),                          # 均值
                np.std(sig),                           # 标准差
                np.sqrt(np.mean(sig**2)),              # RMS
                np.max(np.abs(sig)),                   # 峰值
                np.max(np.abs(sig)) / (np.sqrt(np.mean(sig**2)) + 1e-8),  # 峰值因子
                self._compute_skewness(sig),           # 偏度
                self._compute_kurtosis(sig),           # 峰度
            ])
        return np.mean(features, axis=0)
    
    def _compute_skewness(self, signal):
        """计算偏度"""
        mean = np.mean(signal)
        std = np.std(signal)
        return np.mean(((signal - mean) / std) ** 3)
    
    def _compute_kurtosis(self, signal):
        """计算峰度"""
        mean = np.mean(signal)
        std = np.std(signal)
        return np.mean(((signal - mean) / std) ** 4)
    
    def _compute_wasserstein_distance(self, gen_signals, real_signals):
        """
        Wasserstein距离（Earth Mover's Distance）
        衡量两个分布的差异
        """
        # 展平信号
        gen_flat = np.concatenate([sig.flatten() for sig in gen_signals])
        real_flat = np.concatenate([sig.flatten() for sig in real_signals])
        
        # 计算距离
        distance = wasserstein_distance(gen_flat, real_flat)
        
        # 归一化到[0, 1]
        normalized_distance = 1.0 / (1.0 + distance)
        
        return normalized_distance
    
    def _compute_frechet_distance(self, gen_signals, real_signals):
        """
        Frechet距离（类似FID）
        假设信号特征服从正态分布
        """
        # 提取特征
        gen_features = np.array([self._extract_statistical_features([sig]) for sig in gen_signals])
        real_features = np.array([self._extract_statistical_features([sig]) for sig in real_signals])
        
        # 计算均值和协方差
        mu_gen = np.mean(gen_features, axis=0)
        mu_real = np.mean(real_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        sigma_real = np.cov(real_features, rowvar=False)
        
        # Frechet距离
        diff = mu_gen - mu_real
        covmean = scipy_signal.sqrtm(sigma_gen @ sigma_real)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fd = np.sum(diff**2) + np.trace(sigma_gen + sigma_real - 2*covmean)
        
        # 归一化
        normalized_fd = 1.0 / (1.0 + fd)
        
        return normalized_fd
    
    def _check_physical_validity(self, signals, modality):
        """
        物理合理性检验
        根据不同模态的物理特性检验
        """
        validity_scores = []
        
        for sig in signals:
            if modality == 'acceleration':
                # 加速度合理性检验
                # 1. 幅值范围合理性（±50g）
                max_accel = np.max(np.abs(sig))
                amplitude_valid = 1.0 if max_accel < 500 else 0.0
                
                # 2. 频率范围合理性（0-5kHz）
                freq_valid = self._check_frequency_range(sig, max_freq=5000)
                
                # 3. 连续性（无突变）
                continuity_valid = self._check_continuity(sig)
                
                validity = (amplitude_valid + freq_valid + continuity_valid) / 3.0
                
            elif modality == 'audio':
                # 声音合理性检验
                # 1. 幅值范围（归一化后-1到1）
                amplitude_valid = 1.0 if np.max(np.abs(sig)) <= 1.0 else 0.0
                
                # 2. 频率范围（20Hz-20kHz）
                freq_valid = self._check_frequency_range(sig, min_freq=20, max_freq=20000)
                
                # 3. 能量分布合理性
                energy_valid = self._check_energy_distribution(sig)
                
                validity = (amplitude_valid + freq_valid + energy_valid) / 3.0
                
            elif modality == 'temperature':
                # 温度合理性检验
                # 1. 幅值范围（-20°C到150°C）
                temp_range_valid = 1.0 if (-20 <= np.min(sig) and np.max(sig) <= 150) else 0.0
                
                # 2. 变化速率合理性（温度不能突变）
                rate_valid = self._check_temperature_rate(sig)
                
                # 3. 单调性或趋势合理性
                trend_valid = self._check_temperature_trend(sig)
                
                validity = (temp_range_valid + rate_valid + trend_valid) / 3.0
            
            validity_scores.append(validity)
        
        return np.mean(validity_scores)
    
    def _check_frequency_range(self, signal, min_freq=0, max_freq=5000, fs=10000):
        """检查频率范围合理性"""
        freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=min(256, len(signal)))
        
        # 检查主要能量是否在合理频率范围内
        valid_freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        energy_in_range = np.sum(psd[valid_freq_mask]) / (np.sum(psd) + 1e-8)
        
        return energy_in_range
    
    def _check_continuity(self, signal):
        """检查信号连续性（无异常突变）"""
        # 计算一阶差分
        diff = np.diff(signal)
        
        # 检查是否有异常大的跳变
        threshold = 3 * np.std(diff)
        abnormal_jumps = np.sum(np.abs(diff) > threshold)
        
        continuity_score = 1.0 - (abnormal_jumps / len(diff))
        return max(0.0, continuity_score)
    
    def _check_energy_distribution(self, signal):
        """检查能量分布合理性"""
        # 短时傅里叶变换
        f, t, Zxx = scipy_signal.stft(signal, fs=16000, nperseg=256)
        power = np.abs(Zxx) ** 2
        
        # 检查能量是否过于集中或分散
        # 使用熵来衡量
        power_flat = power.flatten()
        power_prob = power_flat / (np.sum(power_flat) + 1e-8)
        signal_entropy = entropy(power_prob + 1e-8)
        
        # 归一化熵到[0, 1]
        max_entropy = np.log(len(power_prob))
        normalized_entropy = signal_entropy / max_entropy
        
        # 合理的能量分布应该有适中的熵值
        return 1.0 - abs(normalized_entropy - 0.5) * 2
    
    def _check_temperature_rate(self, signal, max_rate=5.0):
        """检查温度变化速率（°C/秒）"""
        diff = np.abs(np.diff(signal))
        max_change = np.max(diff)
        
        if max_change <= max_rate:
            return 1.0
        else:
            return max(0.0, 1.0 - (max_change - max_rate) / max_rate)
    
    def _check_temperature_trend(self, signal):
        """检查温度趋势合理性"""
        # 使用线性拟合检查趋势
        x = np.arange(len(signal))
        z = np.polyfit(x, signal, 1)
        trend = z[0]
        
        # 温度不应该有过大的斜率
        max_trend = 0.1  # 每个采样点最大0.1度变化
        
        if abs(trend) <= max_trend:
            return 1.0
        else:
            return max(0.0, 1.0 - (abs(trend) - max_trend) / max_trend)
    
    def _compute_overall_score(self, freq_sim, stat_sim, wasserstein, frechet, physical):
        """
        计算综合评分
        权重分配：
        - 频域相似度: 20%
        - 统计特征相似度: 20%
        - Wasserstein距离: 20%
        - Frechet距离: 20%
        - 物理合理性: 20%
        """
        overall = (0.2 * freq_sim + 
                   0.2 * stat_sim + 
                   0.2 * wasserstein + 
                   0.2 * frechet + 
                   0.2 * physical)
        return overall


class MultiModalAlignmentEvaluator:
    """多模态对齐评估器"""
    
    def evaluate_alignment(self, generated_data):
        """
        评估多模态对齐质量
        
        评判标准：
        1. 时序同步性（DTW距离）
        2. 因果一致性（交叉相关）
        3. 频域一致性
        4. 物理关联性（基于领域知识）
        """
        metrics = {}
        
        accel = generated_data['acceleration']
        audio = generated_data['audio']
        temp = generated_data['temperature']
        
        # 1. DTW距离（时序对齐）
        dtw_scores = self._compute_dtw_alignment(accel, audio, temp)
        
        # 2. 交叉相关（因果一致性）
        cross_correlation = self._compute_cross_correlation(accel, audio, temp)
        
        # 3. 频域一致性
        frequency_coherence = self._compute_frequency_coherence(accel, audio)
        
        # 4. 物理关联性
        physical_consistency = self._check_physical_consistency(accel, audio, temp)
        
        # 5. 事件同步性（峰值对齐）
        event_synchronization = self._compute_event_synchronization(accel, audio, temp)
        
        metrics = {
            'dtw_alignment': dtw_scores,
            'cross_correlation': cross_correlation,
            'frequency_coherence': frequency_coherence,
            'physical_consistency': physical_consistency,
            'event_synchronization': event_synchronization,
            'overall_alignment_score': self._compute_alignment_score(
                dtw_scores, cross_correlation, frequency_coherence, 
                physical_consistency, event_synchronization
            )
        }
        
        return metrics
    
    def _compute_dtw_alignment(self, accel, audio, temp):
        """
        动态时间规整(DTW)距离
        衡量时序对齐程度
        """
        # 归一化信号
        accel_norm = self._normalize_signal(accel)
        audio_norm = self._normalize_signal(audio)
        temp_norm = self._normalize_signal(temp)
        
        # 重采样到相同长度
        target_len = min(len(accel_norm), len(audio_norm), len(temp_norm))
        accel_resampled = scipy_signal.resample(accel_norm, target_len)
        audio_resampled = scipy_signal.resample(audio_norm, target_len)
        temp_resampled = scipy_signal.resample(temp_norm, target_len)
        
        # 计算DTW距离
        dtw_accel_audio = dtw.distance(accel_resampled, audio_resampled)
        dtw_accel_temp = dtw.distance(accel_resampled, temp_resampled)
        dtw_audio_temp = dtw.distance(audio_resampled, temp_resampled)
        
        # 归一化到[0, 1]，距离越小越好
        dtw_scores = {
            'accel_audio': 1.0 / (1.0 + dtw_accel_audio),
            'accel_temp': 1.0 / (1.0 + dtw_accel_temp),
            'audio_temp': 1.0 / (1.0 + dtw_audio_temp),
            'average': 1.0 / (1.0 + (dtw_accel_audio + dtw_accel_temp + dtw_audio_temp) / 3)
        }
        
        return dtw_scores
    
    def _normalize_signal(self, signal):
        """信号归一化"""
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    def _compute_cross_correlation(self, accel, audio, temp):
        """
        交叉相关分析
        衡量模态间的因果关系和时延
        """
        # 归一化
        accel_norm = self._normalize_signal(accel)
        audio_norm = self._normalize_signal(audio)
        temp_norm = self._normalize_signal(temp)
        
        # 计算互相关
        corr_accel_audio = np.correlate(accel_norm, audio_norm, mode='full')
        corr_accel_temp = np.correlate(accel_norm, temp_norm, mode='full')
        corr_audio_temp = np.correlate(audio_norm, temp_norm, mode='full')
        
        # 找到最大相关值及其对应的时延
        max_corr_accel_audio = np.max(np.abs(corr_accel_audio))
        max_corr_accel_temp = np.max(np.abs(corr_accel_temp))
        max_corr_audio_temp = np.max(np.abs(corr_audio_temp))
        
        # 计算时延
        delay_accel_audio = np.argmax(np.abs(corr_accel_audio)) - len(accel_norm) + 1
        delay_accel_temp = np.argmax(np.abs(corr_accel_temp)) - len(accel_norm) + 1
        delay_audio_temp = np.argmax(np.abs(corr_audio_temp)) - len(audio_norm) + 1
        
        return {
            'accel_audio_correlation': max_corr_accel_audio / len(accel_norm),
            'accel_temp_correlation': max_corr_accel_temp / len(accel_norm),
            'audio_temp_correlation': max_corr_audio_temp / len(audio_norm),
            'accel_audio_delay': delay_accel_audio,
            'accel_temp_delay': delay_accel_temp,
            'audio_temp_delay': delay_audio_temp,
        }
    
    def _compute_frequency_coherence(self, accel, audio):
        """
        频域一致性分析
        使用相干函数衡量频域关联
        """
        # 计算相干函数
        f, Cxy = scipy_signal.coherence(accel, audio, fs=10000, nperseg=256)
        
        # 平均相干性
        mean_coherence = np.mean(Cxy)
        
        # 低频相干性（故障特征通常在低频）
        low_freq_mask = f < 1000
        low_freq_coherence = np.mean(Cxy[low_freq_mask])
        
        return {
            'mean_coherence': mean_coherence,
            'low_frequency_coherence': low_freq_coherence
        }
    
    def _check_physical_consistency(self, accel, audio, temp):
        """
        物理一致性检查
        基于工业故障诊断的领域知识
        """
        consistency_score = 0.0
        num_checks = 0
        
        # 检查1：振动-声音能量一致性
        # 振动越大，声音能量应该越大
        accel_energy = np.sum(accel ** 2)
        audio_energy = np.sum(audio ** 2)
        
        # 归一化能量
        accel_energy_norm = accel_energy / len(accel)
        audio_energy_norm = audio_energy / len(audio)
        
        # 能量应该正相关
        energy_consistency = 1.0 - abs(
            (accel_energy_norm - audio_energy_norm) / (accel_energy_norm + audio_energy_norm + 1e-8)
        )
        consistency_score += energy_consistency
        num_checks += 1
        
        # 检查2：温度-振动关联性
        # 故障时温度通常会上升，振动也会增大
        temp_trend = np.polyfit(np.arange(len(temp)), temp, 1)[0]  # 温度趋势
        accel_mean = np.mean(np.abs(accel))
        
        # 如果温度上升，振动应该较大
        if temp_trend > 0.01:  # 温度上升
            if accel_mean > np.median(np.abs(accel)):
                consistency_score += 1.0
            else:
                consistency_score += 0.5
        else:
            consistency_score += 0.8  # 中性情况
        num_checks += 1
        
        # 检查3：故障模式一致性
        # 检查峰值是否同时出现
        accel_peaks, _ = scipy_signal.find_peaks(np.abs(accel), height=np.std(accel))
        audio_peaks, _ = scipy_signal.find_peaks(np.abs(audio), height=np.std(audio))
        
        # 计算峰值时间的接近程度
        if len(accel_peaks) > 0 and len(audio_peaks) > 0:
            min_distances = []
            for ap in accel_peaks[:5]:  # 只检查前5个峰值
                distances = np.abs(audio_peaks - ap)
                min_distances.append(np.min(distances))
            
            avg_peak_distance = np.mean(min_distances)
            peak_alignment = 1.0 / (1.0 + avg_peak_distance / 10)  # 10个采样点内认为对齐
            consistency_score += peak_alignment
        else:
            consistency_score += 0.5
        num_checks += 1
        
        return consistency_score / num_checks
    
    def _compute_event_synchronization(self, accel, audio, temp):
        """
        事件同步性分析
        检查关键事件（如冲击、峰值）是否同步
        """
        # 检测事件（峰值）
        accel_events = self._detect_events(accel)
        audio_events = self._detect_events(audio)
        temp_events = self._detect_events(temp, threshold_factor=0.5)  # 温度变化较慢
        
        # 计算事件对齐程度
        sync_scores = []
        
        # 加速度-声音事件同步
        if len(accel_events) > 0 and len(audio_events) > 0:
            accel_audio_sync = self._calculate_event_sync(accel_events, audio_events, tolerance=50)
            sync_scores.append(accel_audio_sync)
        
        # 加速度-温度事件同步（温度有延迟是合理的）
        if len(accel_events) > 0 and len(temp_events) > 0:
            accel_temp_sync = self._calculate_event_sync(accel_events, temp_events, tolerance=200)
            sync_scores.append(accel_temp_sync)
        
        # 声音-温度事件同步
        if len(audio_events) > 0 and len(temp_events) > 0:
            audio_temp_sync = self._calculate_event_sync(audio_events, temp_events, tolerance=200)
            sync_scores.append(audio_temp_sync)
        
        return {
            'synchronization_score': np.mean(sync_scores) if sync_scores else 0.5,
            'num_accel_events': len(accel_events),
            'num_audio_events': len(audio_events),
            'num_temp_events': len(temp_events)
        }
    
    def _detect_events(self, signal, threshold_factor=2.0):
        """检测信号中的显著事件"""
        threshold = threshold_factor * np.std(signal)
        peaks, _ = scipy_signal.find_peaks(np.abs(signal), height=threshold, distance=20)
        return peaks
    
    def _calculate_event_sync(self, events1, events2, tolerance=50):
        """
        计算两组事件的同步程度
        tolerance: 允许的时间差（采样点数）
        """
        if len(events1) == 0 or len(events2) == 0:
            return 0.0
        
        matched_events = 0
        for e1 in events1:
            # 检查是否有events2中的事件在容差范围内
            if np.any(np.abs(events2 - e1) <= tolerance):
                matched_events += 1
        
        # 同步得分
        sync_score = matched_events / max(len(events1), len(events2))
        return sync_score
    
    def _compute_alignment_score(self, dtw, cross_corr, freq_coh, phys_cons, event_sync):
        """
        计算综合对齐得分
        权重分配：
        - DTW对齐: 25%
        - 交叉相关: 20%
        - 频域一致性: 15%
        - 物理一致性: 25%
        - 事件同步: 15%
        """
        dtw_score = dtw['average']
        corr_score = (cross_corr['accel_audio_correlation'] + 
                      cross_corr['accel_temp_correlation'] + 
                      cross_corr['audio_temp_correlation']) / 3
        freq_score = freq_coh['mean_coherence']
        phys_score = phys_cons
        event_score = event_sync['synchronization_score']
        
        overall = (0.25 * dtw_score + 
                   0.20 * corr_score + 
                   0.15 * freq_score + 
                   0.25 * phys_score + 
                   0.15 * event_score)
        
        return overall


def generate_balanced_fault_data(model, config, target_classes, num_samples):
    """生成平衡的故障数据（带评估）"""
    model.eval()
    generated_data = {cls: [] for cls in target_classes}
    
    # 初始化评估器
    evaluator = FaultDataQualityEvaluator(real_data_stats=None)
    alignment_evaluator = MultiModalAlignmentEvaluator()
    
    with torch.no_grad():
        for fault_class in target_classes:
            print(f"\n正在生成 {fault_class} 故障数据...")
            
            class_quality_scores = []
            class_alignment_scores = []
            
            for i in range(num_samples):
                # 生成数据（代码同前）
                fault_description = f"设备故障类型：{fault_class}，请生成相应的传感器数据。"
                condition = model.text_tokenizer(
                    fault_description,
                    return_tensors='pt',
                    max_length=256,
                    padding='max_length',
                    truncation=True
                ).to(model.device)
                
                generated_tokens = model.sample(
                    num_samples=1,
                    condition=condition,
                    guidance_scale=2.0
                )
                
                multimodal_signals = model.signal_tokenizer.decode_multimodal(
                    generated_tokens,
                    model._get_modality_mask(generated_tokens)
                )
                
                # 评估生成质量
                sample_data = {
                    'acceleration': multimodal_signals['acceleration'].cpu().numpy(),
                    'audio': multimodal_signals['audio'].cpu().numpy(),
                    'temperature': multimodal_signals['temperature'].cpu().numpy(),
                }
                
                # 对齐评估
                alignment_metrics = alignment_evaluator.evaluate_alignment(sample_data)
                alignment_score = alignment_metrics['overall_alignment_score']
                
                # 只保留高质量样本（对齐得分>0.7）
                if alignment_score > 0.7:
                    sample_data['label'] = fault_class
                    sample_data['alignment_score'] = alignment_score
                    generated_data[fault_class].append(sample_data)
                    
                    class_alignment_scores.append(alignment_score)
                
                if (i + 1) % 100 == 0:
                    avg_alignment = np.mean(class_alignment_scores[-100:])
                    print(f"  已生成 {i + 1}/{num_samples} 样本, 平均对齐得分: {avg_alignment:.3f}")
            
            # 类别汇总
            print(f"\n{fault_class} 生成完成:")
            print(f"  平均对齐得分: {np.mean(class_alignment_scores):.3f}")
            print(f"  高质量样本数: {len(generated_data[fault_class])}/{num_samples}")
    
    return generated_data


def visualize_alignment(sample_data, save_path='alignment_visualization.png'):
    """可视化多模态对齐情况"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    accel = sample_data['acceleration']
    audio = sample_data['audio']
    temp = sample_data['temperature']
    
    time_accel = np.arange(len(accel)) / 10000  # 10kHz采样率
    time_audio = np.arange(len(audio)) / 16000  # 16kHz采样率
    time_temp = np.arange(len(temp)) / 100      # 100Hz采样率
    
    # 加速度
    axes[0].plot(time_accel, accel, label='Acceleration')
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].set_title('Multi-Modal Signal Alignment')
    axes[0].legend()
    axes[0].grid(True)
    
    # 声音
    axes[1].plot(time_audio, audio, label='Audio', color='orange')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True)
    
    # 温度
    axes[2].plot(time_temp, temp, label='Temperature', color='red')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"对齐可视化已保存到: {save_path}")