#!/bin/bash

echo "======================================"
echo "工业故障数据生成系统"
echo "用户: qianshoufeijian"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"

# 检查检查点
if [ ! -f "outputs/best_model.pt" ]; then
    echo "❌ 错误: 找不到模型检查点"
    echo "请先运行训练: python train_industrial.py"
    exit 1
fi

# 执行生成
echo "开始生成数据..."
python generate_data.py \
    --config configs/generation_config.yaml \
    --checkpoint outputs/best_model.pt

# 验证
echo "验证生成数据..."
python validate_generated_data.py

echo "✓ 完成！"
echo "生成的数据位于: ./generated_data/"