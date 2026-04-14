#!/bin/bash
# 记忆增强依存解析系统使用示例

cd /home16T/ljj/4-biaffine-4paper/Memory_enhance

echo "=========================================="
echo "记忆增强依存解析系统"
echo "=========================================="
echo ""

# 检查是否已有记忆知识库
if [ ! -f "vi_memory_bank.json" ]; then
    echo "步骤1: 构建记忆知识库..."
    python run_memory_enhancement.py --build-bank
    echo ""
fi

# 运行记忆增强解析
echo "步骤2: 运行记忆增强解析..."
echo "处理文件: input/Qwen2.5-7B-vi.conllu"
echo "输出文件: output/Qwen2.5-7B-vi-enhanced.conllu"
echo ""

python run_memory_enhancement.py \
  --input input/Qwen2.5-7B-vi.conllu \
  --output output/Qwen2.5-7B-vi-enhanced.conllu \
  --model-path /home16T/ljj/model/Qwen2.5-7B-Instruct

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
echo "输出文件: output/Qwen2.5-7B-vi-enhanced.conllu"
echo ""
echo "查看结果:"
echo "  head -50 output/Qwen2.5-7B-vi-enhanced.conllu"
echo ""
echo "对比修改:"
echo "  diff input/Qwen2.5-7B-vi.conllu output/Qwen2.5-7B-vi-enhanced.conllu | head -100"
