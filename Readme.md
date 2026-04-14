对应大论文第四个论文-EMNLP findings 2025

## 环境准备

### 1. 激活Conda环境

```
conda activate llama_factory
```

---

## 训练执行

### 2. 大模型微调（LLaMA Factory）

#### 2.1 启动WebUI界面

```
# 方式一：直接启动
llamafactory-cli webui

# 方式二：指定GPU设备启动
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webui
```

#### 2.2 配置参数

- **模型路径**：`/home16T/ljj/model/Qwen2.5-7B-Instruct`
- **数据集路径**：`/home16T/ljj/4-biaffine-4paper/data`
- **操作步骤**：
    1. 在WebUI界面中选择本地原始模型路径
    2. 选择监督微调LoRA数据集
    3. 从指定目录中挑选合适的数据集

#### 2.3 模型合并

微调完成后，将模型合并并存储至：

```
/home16T/ljj/model/Qwen2.5-7B-Instruct-lora
```

---

### 3. 知识库记忆增强（Memory Bank）

#### 3.1 输入处理

- **输入内容**：
    - 第一次解析的结果
    - 知识库数据
- **处理方式**：根据不同记忆强度进行检索增强

#### 3.2 输出结果

- **输出内容**：修改后的结果
- **增强方式**：基于知识库的记忆增强

#### cd /home16T/ljj/4-biaffine-4paper/Memory_enhance 运行

#### 完整流程（一次性执行）

```bash
python run_memory_enhancement.py \
  --build-bank \
  --input input/Qwen2.5-7B-vi.conllu \
  --output output/Qwen2.5-7B-vi-enhanced.conllu \
  --model-path /home16T/ljj/model/Qwen2.5-7B-Instruct
```

#### 测试模式（处理前N个句子）

```bash
python run_memory_enhancement.py \
  --input input/Qwen2.5-7B-vi.conllu \
  --output output/test.conllu \
  --model-path /home16T/ljj/model/Qwen2.5-7B-Instruct \
  --max-sentences 3
```
---

