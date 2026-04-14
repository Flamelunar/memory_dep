#!/usr/bin/env python3
"""
记忆知识库构建模块
根据论文方法构建越南语依存标签记忆知识库
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# 用户提供的记忆强度数据（78个依存标签）
MEMORY_STRENGTH_DATA = [
    ("punct", 14.57, 99.82, 1.00),
    ("nsubj", 8.19, 86.93, 0.86),
    ("obj", 7.42, 79.43, 0.79),
    ("root", 6.84, 81.14, 0.80),
    ("advmod", 6.43, 86.47, 0.85),
    ("case", 4.83, 83.14, 0.79),
    ("conj", 4.00, 75.47, 0.69),
    ("nmod", 3.94, 62.25, 0.56),
    ("xcomp", 2.81, 49.33, 0.40),
    ("mark", 2.51, 83.64, 0.65),
    ("obl", 2.22, 38.32, 0.28),
    ("nummod", 2.04, 88.57, 0.62),
    ("amod", 2.01, 59.69, 0.42),
    ("cc", 1.92, 64.16, 0.44),
    ("advcl", 1.80, 56.21, 0.37),
    ("obl:tmod", 1.77, 69.86, 0.46),
    ("clf:det", 1.74, 71.98, 0.47),
    ("det:pmod", 1.62, 78.22, 0.49),
    ("det", 1.62, 96.73, 0.60),
    ("advmod:neg", 1.56, 88.60, 0.54),
    ("obl:comp", 1.39, 48.89, 0.28),
    ("compound", 1.32, 25.91, 0.14),
    ("advmod:adj", 1.31, 41.54, 0.23),
    ("compound:vmod", 1.27, 64.86, 0.35),
    ("acl:subj", 1.21, 37.11, 0.19),
    ("discourse", 1.15, 64.08, 0.32),
    ("parataxis", 1.09, 47.55, 0.23),
    ("ccomp", 0.92, 48.03, 0.20),
    ("cop", 0.87, 81.08, 0.33),
    ("nmod:poss", 0.86, 87.21, 0.35),
    ("compound:svc", 0.77, 34.57, 0.13),
    ("aux", 0.74, 66.32, 0.24),
    ("acl", 0.73, 40.00, 0.14),
    ("advcl:objective", 0.55, 70.83, 0.20),
    ("acl:tmod", 0.44, 64.58, 0.15),
    ("obl:with", 0.42, 81.63, 0.18),
    ("clf", 0.40, 63.46, 0.14),
    ("aux:pass", 0.38, 77.36, 0.16),
    ("mark:pcomp", 0.33, 92.50, 0.17),
    ("compound:dir", 0.30, 29.85, 0.05),
    ("compound:prt", 0.30, 27.78, 0.05),
    ("nsubj:pass", 0.27, 71.88, 0.11),
    ("xcomp:adj", 0.27, 57.14, 0.08),
    ("appos", 0.23, 21.05, 0.03),
    ("obl:iobj", 0.23, 71.43, 0.09),
    ("appos:nmod", 0.22, 42.86, 0.05),
    ("compound:verbnoun", 0.22, 50.00, 0.06),
    ("csubj", 0.17, 4.76, 0.00),
    ("compound:pron", 0.15, 66.67, 0.06),
    ("obl:adj", 0.15, 0, 0.00),
    ("compound:amod", 0.15, 0, 0.00),
    ("obl:about", 0.13, 50, 0.04),
    ("acl:tonp", 0.13, 33.33, 0.02),
    ("nsubj:nn", 0.12, 0, 0.00),  # NaN -> 0
    ("csubj:vsubj", 0.11, 33.33, 0.02),
    ("iobj", 0.09, 25, 0.01),
    ("flat", 0.09, 0, 0.00),  # NaN -> 0
    ("dislocated", 0.08, 25, 0.01),
    ("flat:number", 0.08, 33.33, 0.02),
    ("advmod:dir", 0.08, 0, 0.00),  # NaN -> 0
    ("flat:date", 0.08, 50, 0.02),
    ("fixed", 0.07, 0, 0.00),
    ("csubj:asubj", 0.05, 40, 0.01),
    ("nsubj:xsubj", 0.04, 0, 0.00),  # NaN -> 0
    ("compound:atov", 0.03, 25, 0.01),
    ("dep", 0.03, 0, 0.00),
    ("obl:agent", 0.03, 50, 0.01),
    ("vocative", 0.02, 100, 0.01),
    ("flat:time", 0.02, 0, 0.00),  # NaN -> 0
    ("acl:relcl", 0.02, 0, 0.00),  # NaN -> 0
    ("xcomp:vcomp", 0.02, 0, 0.00),  # NaN -> 0
    ("flat:name", 0.02, 0, 0.00),
    ("compound:redup", 0.01, 0, 0.00),
    ("compound:apr", 0.01, 0, 0.00),  # NaN -> 0
    ("compound:adj", 0.01, 0, 0.00),  # NaN -> 0
    ("obl:adv", 0.01, 0, 0.00),  # NaN -> 0
    ("compound:z", 0.01, 100, 0.01),
]


def categorize_memory_strength(memory_strength: float) -> str:
    """根据记忆强度分类"""
    if memory_strength >= 0.6:
        return "strong_memory"
    elif memory_strength >= 0.3:
        return "moderate_memory"
    else:
        return "weak_memory"


def parse_conllu_file(file_path: str) -> List[List[Dict]]:
    """
    解析CoNLL-U文件，返回句子列表
    每个句子是一个token列表
    """
    sentences = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 空行表示句子结束
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # 跳过注释行
            if line.startswith('#'):
                continue

            # 解析token行
            parts = line.split('\t')
            if len(parts) != 10:
                continue

            word_id = parts[0]
            # 跳过多词标记和空节点
            if '-' in word_id or '.' in word_id:
                continue

            try:
                token = {
                    'id': int(word_id),
                    'form': parts[1],
                    'lemma': parts[2],
                    'upos': parts[3],
                    'xpos': parts[4],
                    'feats': parts[5],
                    'head': int(parts[6]) if parts[6] != '_' else 0,
                    'deprel': parts[7],
                    'deps': parts[8],
                    'misc': parts[9]
                }
                current_sentence.append(token)
            except (ValueError, IndexError):
                continue

        # 处理最后一个句子
        if current_sentence:
            sentences.append(current_sentence)

    return sentences


def extract_label_examples(sentences: List[List[Dict]], deprel: str, max_examples: int = 10) -> List[Dict]:
    """
    从句子中提取指定依存标签的示例
    """
    examples = []

    for sentence in sentences:
        if len(examples) >= max_examples:
            break

        for token in sentence:
            if token['deprel'] == deprel and len(examples) < max_examples:
                # 找到头词
                head_word = "ROOT" if token['head'] == 0 else ""
                for t in sentence:
                    if t['id'] == token['head']:
                        head_word = t['form']
                        break

                # 构建句子文本
                sentence_text = " ".join([t['form'] for t in sentence])

                # 构建CoNLL-U上下文（完整句子）
                context_lines = []
                for t in sentence:
                    context_lines.append(
                        f"{t['id']}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t{t['xpos']}\t"
                        f"{t['feats']}\t{t['head']}\t{t['deprel']}\t{t['deps']}\t{t['misc']}"
                    )
                context = "\n".join(context_lines)

                example = {
                    'sentence': sentence_text,
                    'word_id': token['id'],
                    'word': token['form'],
                    'head_id': token['head'],
                    'head_word': head_word,
                    'context': context
                }
                examples.append(example)
                break  # 每个句子只取一个示例

    return examples


def build_memory_bank(train_file: str, output_file: str):
    """
    构建记忆知识库
    """
    print("开始构建记忆知识库...")
    print(f"读取训练数据: {train_file}")

    # 解析训练数据
    sentences = parse_conllu_file(train_file)
    print(f"共读取 {len(sentences)} 个句子")

    # 构建知识库
    labels = []

    for deprel, percentage, correct_rate, memory_strength in MEMORY_STRENGTH_DATA:
        print(f"处理标签: {deprel} (记忆强度: {memory_strength:.2f})")

        # 提取示例
        examples = extract_label_examples(sentences, deprel, max_examples=10)

        # 分类记忆强度
        category = categorize_memory_strength(memory_strength)

        label_info = {
            'deprel': deprel,
            'percentage': percentage,
            'correct_rate': correct_rate,
            'memory_strength': memory_strength,
            'category': category,
            'example_count': len(examples),
            'examples': examples
        }
        labels.append(label_info)

    # 统计信息
    weak_count = sum(1 for l in labels if l['category'] == 'weak_memory')
    moderate_count = sum(1 for l in labels if l['category'] == 'moderate_memory')
    strong_count = sum(1 for l in labels if l['category'] == 'strong_memory')

    memory_bank = {
        'language_code': 'vi',
        'language_name': 'Vietnamese',
        'total_labels': len(labels),
        'weak_memory_count': weak_count,
        'moderate_memory_count': moderate_count,
        'strong_memory_count': strong_count,
        'labels': labels
    }

    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(memory_bank, f, ensure_ascii=False, indent=2)

    print(f"\n记忆知识库构建完成！")
    print(f"输出文件: {output_file}")
    print(f"总标签数: {len(labels)}")
    print(f"  - 强记忆标签 (≥0.6): {strong_count}")
    print(f"  - 中等记忆标签 (0.3-0.6): {moderate_count}")
    print(f"  - 弱记忆标签 (<0.3): {weak_count}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='构建记忆知识库')
    parser.add_argument('--train-file', type=str,
                        default='/home16T/ljj/mydata/ud-2.13/tgt/vi-train.conllu',
                        help='训练数据文件路径')
    parser.add_argument('--output', type=str,
                        default='/home16T/ljj/4-biaffine-4paper/Memory_enhance/vi_memory_bank.json',
                        help='输出文件路径')

    args = parser.parse_args()

    build_memory_bank(args.train_file, args.output)
