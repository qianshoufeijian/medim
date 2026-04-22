#!/bin/bash

echo "安装工业故障诊断模型依赖..."

# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 核心依赖
pip install transformers accelerate omegaconf
pip install librosa scipy numpy
pip install scikit-learn matplotlib tqdm
pip install tensordict einops

# 评估指标
pip install dtaidistance  # DTW距离

# 可选：多进程
pip install peft  # LoRA

echo "✓ 依赖安装完成！"