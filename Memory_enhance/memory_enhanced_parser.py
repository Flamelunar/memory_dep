#!/usr/bin/env python3
"""
记忆增强解析器
使用Qwen2.5-7B模型根据记忆强度修改依存解析结果
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Set
import re


class MemoryEnhancedParser:
    def __init__(self, model_path: str, memory_bank_path: str, device: str = "cuda"):
        """
        初始化记忆增强解析器

        Args:
            model_path: Qwen2.5-7B模型路径
            memory_bank_path: 记忆知识库JSON文件路径
            device: 设备（cuda或cpu）
        """
        self.device = device
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        print("Model loaded successfully.")

        print(f"Loading memory bank from {memory_bank_path}...")
        with open(memory_bank_path, 'r', encoding='utf-8') as f:
            self.memory_bank = json.load(f)

        # 构建标签到信息的映射
        self.label_info = {}
        for label_data in self.memory_bank['labels']:
            self.label_info[label_data['deprel']] = label_data

        print(f"Memory bank loaded: {len(self.label_info)} labels")

    def parse_sentence(self, conllu_lines: List[str]) -> Tuple[List[Dict], str]:
        """
        解析单个句子的CoNLL-U格式
        支持标准10列格式和简化5列格式

        Returns:
            (tokens列表, 句子文本)
        """
        tokens = []
        for line in conllu_lines:
            if not line.strip() or line.startswith('#'):
                continue

            parts = line.split('\t')

            # 跳过格式不正确的行
            if len(parts) not in [5, 10]:
                continue

            word_id = parts[0]
            if '-' in word_id or '.' in word_id:
                continue

            try:
                if len(parts) == 5:
                    # 简化格式: id, form, upos, head, deprel
                    token = {
                        'id': int(word_id),
                        'form': parts[1],
                        'lemma': '_',
                        'upos': parts[2],
                        'xpos': '_',
                        'feats': '_',
                        'head': int(parts[3]),
                        'deprel': parts[4],
                        'deps': '_',
                        'misc': '_'
                    }
                else:
                    # 标准10列格式
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
                tokens.append(token)
            except (ValueError, IndexError):
                continue

        sentence_text = " ".join([t['form'] for t in tokens])
        return tokens, sentence_text

    def identify_weak_labels(self, tokens: List[Dict]) -> List[Tuple[int, str, float]]:
        """
        识别句子中的弱记忆标签

        Returns:
            [(token_id, deprel, memory_strength), ...]
        """
        weak_labels = []
        for token in tokens:
            deprel = token['deprel']
            if deprel in self.label_info:
                label_data = self.label_info[deprel]
                if label_data['category'] == 'weak_memory':
                    weak_labels.append((token['id'], deprel, label_data['memory_strength']))

        return weak_labels

    def build_enhancement_prompt(self, tokens: List[Dict], sentence_text: str,
                                 weak_labels: List[Tuple[int, str, float]]) -> str:
        """
        构建记忆增强提示词
        """
        # 构建当前句子的CoNLL-U格式
        conllu_lines = []
        for t in tokens:
            conllu_lines.append(
                f"{t['id']}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t{t['xpos']}\t"
                f"{t['feats']}\t{t['head']}\t{t['deprel']}\t{t['deps']}\t{t['misc']}"
            )
        current_conllu = "\n".join(conllu_lines)

        # 构建弱记忆标签信息
        weak_label_info = []
        for token_id, deprel, memory_strength in weak_labels:
            label_data = self.label_info[deprel]

            # 获取示例
            examples_text = []
            for i, ex in enumerate(label_data['examples'][:3], 1):  # 最多3个示例
                examples_text.append(
                    f"  Example {i}: {ex['word']} -> {ex['head_word']} ({deprel})\n"
                    f"  Context: {ex['sentence'][:100]}..."
                )

            weak_label_info.append(
                f"- Label: {deprel}\n"
                f"  Memory Strength: {memory_strength:.2f}\n"
                f"  Correct Rate: {label_data['correct_rate']:.2f}%\n"
                f"  Category: {label_data['category']}\n"
                f"  Examples from training data:\n" + "\n".join(examples_text)
            )

        prompt = f"""You are a Vietnamese dependency parsing expert. Given a sentence with its initial parsing result, please review and correct the dependency relations based on the memory strength of each label.

[Memory Knowledge Base]
The following dependency labels in this sentence have weak memory (low accuracy) and need special attention:

{chr(10).join(weak_label_info)}

[Input Sentence]
{sentence_text}

[Initial Parsing Result (CoNLL-U Format)]
{current_conllu}

[Task]
Review the dependency relations, especially those with weak memory labels listed above. Based on the examples provided, correct any errors in the dependency relations (head and deprel).

[Important Notes]
1. Each sentence must have exactly one root node (head=0)
2. Keep all words in the original order
3. Only modify the head (column 7) and deprel (column 8) if necessary
4. Output ONLY the corrected CoNLL-U format (10 columns, tab-separated)
5. Do not add explanations or comments

[Output Format]
Output the corrected parsing result in CoNLL-U format:
"""

        return prompt

    def call_llm(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """
        调用大模型生成修改建议
        """
        messages = [
            {"role": "system", "content": "You are a Vietnamese dependency parsing expert."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def parse_llm_output(self, llm_output: str) -> List[Dict]:
        """
        解析LLM输出的CoNLL-U格式
        """
        tokens = []
        lines = llm_output.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) != 10:
                continue

            word_id = parts[0]
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
                tokens.append(token)
            except (ValueError, IndexError):
                continue

        return tokens

    def compare_and_get_modifications(self, original_tokens: List[Dict],
                                      corrected_tokens: List[Dict]) -> List[Dict]:
        """
        比较原始和修改后的tokens，返回修改列表
        """
        modifications = []

        # 确保两个列表长度相同
        if len(original_tokens) != len(corrected_tokens):
            return modifications

        for orig, corr in zip(original_tokens, corrected_tokens):
            if orig['id'] != corr['id']:
                continue

            changes = {}
            if orig['head'] != corr['head']:
                changes['head'] = (orig['head'], corr['head'])
            if orig['deprel'] != corr['deprel']:
                changes['deprel'] = (orig['deprel'], corr['deprel'])

            if changes:
                # 获取记忆强度信息
                memory_strength = 0.0
                if orig['deprel'] in self.label_info:
                    memory_strength = self.label_info[orig['deprel']]['memory_strength']

                modifications.append({
                    'word_id': orig['id'],
                    'word': orig['form'],
                    'changes': changes,
                    'memory_strength': memory_strength
                })

        return modifications

    def process_sentence(self, conllu_lines: List[str]) -> Tuple[List[str], List[Dict]]:
        """
        处理单个句子

        Returns:
            (输出的CoNLL-U行列表, 修改列表)
        """
        # 解析句子
        tokens, sentence_text = self.parse_sentence(conllu_lines)
        if not tokens:
            return conllu_lines, []

        # 识别弱记忆标签
        weak_labels = self.identify_weak_labels(tokens)

        # 如果没有弱记忆标签，直接返回原始内容
        if not weak_labels:
            return conllu_lines, []

        print(f"  Found {len(weak_labels)} weak memory labels: {[l[1] for l in weak_labels]}")

        # 构建增强提示词
        prompt = self.build_enhancement_prompt(tokens, sentence_text, weak_labels)

        # 调用LLM
        llm_output = self.call_llm(prompt)

        # 解析LLM输出
        corrected_tokens = self.parse_llm_output(llm_output)

        # 如果解析失败，返回原始内容
        if not corrected_tokens or len(corrected_tokens) != len(tokens):
            print("  Warning: LLM output parsing failed, keeping original")
            return conllu_lines, []

        # 比较并获取修改
        modifications = self.compare_and_get_modifications(tokens, corrected_tokens)

        if not modifications:
            return conllu_lines, []

        # 构建输出行
        output_lines = []

        # 添加修改注释
        for mod in modifications:
            changes_desc = []
            if 'head' in mod['changes']:
                old_head, new_head = mod['changes']['head']
                changes_desc.append(f"head: {old_head}->{new_head}")
            if 'deprel' in mod['changes']:
                old_deprel, new_deprel = mod['changes']['deprel']
                changes_desc.append(f"deprel: {old_deprel}->{new_deprel}")

            output_lines.append(
                f"# Modified: word_id={mod['word_id']}, {', '.join(changes_desc)} "
                f"(memory_strength: {mod['memory_strength']:.2f}, reason: weak memory label)"
            )

        # 添加修改后的CoNLL-U行
        for token in corrected_tokens:
            output_lines.append(
                f"{token['id']}\t{token['form']}\t{token['lemma']}\t{token['upos']}\t{token['xpos']}\t"
                f"{token['feats']}\t{token['head']}\t{token['deprel']}\t{token['deps']}\t{token['misc']}"
            )

        return output_lines, modifications

    def process_file(self, input_path: str, output_path: str, max_sentences: int = None) -> Dict:
        """
        处理整个CoNLL-U文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            max_sentences: 最多处理的句子数（用于测试）

        Returns:
            统计信息字典
        """
        print(f"Processing {input_path}...")

        # 读取输入文件
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 分割成句子
        sentences = []
        current_sentence = []
        for line in lines:
            if line.strip():
                current_sentence.append(line.rstrip('\n'))
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
        if current_sentence:
            sentences.append(current_sentence)

        # 限制处理的句子数
        if max_sentences:
            sentences = sentences[:max_sentences]
            print(f"Processing first {len(sentences)} sentences (max_sentences={max_sentences})")
        else:
            print(f"Total sentences: {len(sentences)}")

        # 处理每个句子
        output_lines = []
        all_modifications = []
        sentences_modified = 0
        sentences_with_weak_labels = 0

        for i, sentence_lines in enumerate(sentences, 1):
            print(f"Processing sentence {i}/{len(sentences)}...")

            # 先检查是否有弱记忆标签
            tokens, _ = self.parse_sentence(sentence_lines)
            weak_labels = self.identify_weak_labels(tokens)

            if weak_labels:
                sentences_with_weak_labels += 1

            processed_lines, modifications = self.process_sentence(sentence_lines)

            if modifications:
                sentences_modified += 1
                all_modifications.extend(modifications)

            output_lines.extend(processed_lines)
            output_lines.append("")  # 空行分隔句子

        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_path}")

        # 统计修改的标签类型
        label_counts = {}
        for mod in all_modifications:
            if 'deprel' in mod['changes']:
                old_label = mod['changes']['deprel'][0]
                label_counts[old_label] = label_counts.get(old_label, 0) + 1

        # 返回统计信息
        return {
            'total_sentences': len(sentences),
            'sentences_with_weak_labels': sentences_with_weak_labels,
            'sentences_modified': sentences_modified,
            'total_modifications': len(all_modifications),
            'label_modifications': label_counts
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Memory-Enhanced Dependency Parser")
    parser.add_argument("--input", required=True, help="Input CoNLL-U file")
    parser.add_argument("--output", required=True, help="Output CoNLL-U file")
    parser.add_argument("--memory-bank", required=True, help="Memory bank JSON file")
    parser.add_argument("--model-path", required=True, help="Qwen2.5-7B model path")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    # 创建解析器
    parser = MemoryEnhancedParser(
        model_path=args.model_path,
        memory_bank_path=args.memory_bank,
        device=args.device
    )

    # 处理文件
    parser.process_file(args.input, args.output)


if __name__ == "__main__":
    main()
