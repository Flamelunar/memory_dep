# 记忆增强依存解析系统

基于论文《Memory-enhanced Large Language Model for Cross-lingual Dependency Parsing via Deep Hierarchical Syntax Understanding》实现的记忆增强依存解析系统。

## 系统概述

该系统通过构建依存标签记忆知识库，使用大语言模型（Qwen2.5-7B）根据记忆强度修正弱记忆标签，提高低资源语言的依存解析准确率。

## 文件结构

```
Memory_enhance/
├── build_memory_bank.py          # 记忆知识库构建模块
├── memory_enhanced_parser.py     # 记忆增强解析器
├── run_memory_enhancement.py     # 主程序入口
├── vi_memory_bank.json          # 越南语记忆知识库（生成）
├── input/                       # 输入目录
│   └── Qwen2.5-7B-vi.conllu    # 初次解析结果
└── output/                      # 输出目录
    └── Qwen2.5-7B-vi-enhanced.conllu  # 增强后的结果
```

## 使用方法

### 1. 构建记忆知识库

```bash
cd /home16T/ljj/4-biaffine-4paper/Memory_enhance
python run_memory_enhancement.py --build-bank
```

这将从训练数据 `/home16T/ljj/mydata/ud-2.13/tgt/vi-train.conllu` 中提取示例，构建记忆知识库 `vi_memory_bank.json`。

### 2. 运行记忆增强解析

```bash
python run_memory_enhancement.py \
  --input input/Qwen2.5-7B-vi.conllu \
  --output output/Qwen2.5-7B-vi-enhanced.conllu \
  --model-path /home16T/ljj/model/Qwen2.5-7B-Instruct
```

### 3. 完整流程（一次性执行）

```bash
python run_memory_enhancement.py \
  --build-bank \
  --input input/Qwen2.5-7B-vi.conllu \
  --output output/Qwen2.5-7B-vi-enhanced.conllu \
  --model-path /home16T/ljj/model/Qwen2.5-7B-Instruct
```

### 4. 测试模式（处理前N个句子）

```bash
python run_memory_enhancement.py \
  --input input/Qwen2.5-7B-vi.conllu \
  --output output/test.conllu \
  --model-path /home16T/ljj/model/Qwen2.5-7B-Instruct \
  --max-sentences 10
```

## 命令行参数

- `--build-bank`: 构建记忆知识库
- `--train-data`: 训练数据路径（默认：`/home16T/ljj/mydata/ud-2.13/tgt/vi-train.conllu`）
- `--memory-bank`: 记忆知识库文件路径（默认：`vi_memory_bank.json`）
- `--input`: 输入CoNLL-U文件路径（初次解析结果）
- `--output`: 输出CoNLL-U文件路径（增强后的结果）
- `--model-path`: Qwen2.5-7B模型路径（默认：`/home16T/ljj/model/Qwen2.5-7B-Instruct`）
- `--device`: 运行设备（默认：`cuda`，可选：`cpu`）
- `--max-sentences`: 最多处理的句子数量（用于测试）

## 记忆强度分类

根据论文方法，依存标签按记忆强度分为三类：

- **强记忆标签** (MS ≥ 0.6): 高准确率，无需修改
- **中等记忆标签** (0.3 ≤ MS < 0.6): 中等准确率，需要关注
- **弱记忆标签** (MS < 0.3): 低准确率，重点修改对象

## 输入格式

支持两种CoNLL-U格式：

### 标准10列格式
```
1   word_id
2   word_form
3   lemma
4   upos
5   xpos
6   feats
7   head
8   deprel
9   deps
10  misc
```

### 简化5列格式
```
1   word_id
2   word_form
3   upos
4   head
5   deprel
```

## 输出格式

输出为标准10列CoNLL-U格式，并在修改的句子前添加注释行：

```
# Modified: word_id=12, deprel: advmod:adj->advmod (memory_strength: 0.23, reason: weak memory label)
1   Thanh   _   PROPN   _   _   7   nsubj   _   _
2   bắt chuyện   _   VERB   _   _   7   xcomp   _   _
...
```

## 记忆知识库结构

```json
{
  "language_code": "vi",
  "language_name": "Vietnamese",
  "total_labels": 77,
  "labels": [
    {
      "deprel": "punct",
      "percentage": 14.57,
      "correct_rate": 99.82,
      "memory_strength": 1.00,
      "category": "strong_memory",
      "examples": [
        {
          "sentence": "Tôi nhớ lời anh chủ tịch ...",
          "word_id": 12,
          "word": ":",
          "head_id": 15,
          "head_word": "nhỏ",
          "context": "..."
        }
      ]
    }
  ]
}
```

## 统计信息

运行完成后会输出统计信息：

```
Statistics
============================================================
Total sentences: 150
Sentences with weak labels: 89
Sentences modified: 45
Total modifications: 67

Modifications by label:
  compound: 12
  obl: 8
  acl: 7
  advmod:adj: 6
  ...
```

## 依赖项

- Python 3.8+
- PyTorch
- Transformers
- Qwen2.5-7B-Instruct 模型

## 论文参考

Liu, J., Li, Y., Yu, Z., Su, S., Gao, S., & Huang, Y. (2025). Memory-enhanced Large Language Model for Cross-lingual Dependency Parsing via Deep Hierarchical Syntax Understanding. In Findings of the Association for Computational Linguistics: EMNLP 2025.

## 实现特点

1. **简化实现**：只使用目标语言（越南语）知识库，不需要源语言
2. **记忆强度数据**：直接使用用户提供的78个标签的记忆强度数据
3. **示例提取**：从训练数据中自动提取每个标签的示例句子
4. **批注说明**：在输出文件中添加注释行，说明每次修改的内容和原因
5. **灵活配置**：支持测试模式、自定义参数等

## 注意事项

1. 首次运行需要先构建记忆知识库（使用 `--build-bank`）
2. 模型推理较慢，建议先用 `--max-sentences` 测试
3. 需要GPU支持，建议使用CUDA设备
4. 输出文件会覆盖已存在的文件，请注意备份
