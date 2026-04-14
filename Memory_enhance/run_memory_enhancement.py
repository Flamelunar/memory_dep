#!/usr/bin/env python3
"""
记忆增强依存解析主程序
整合知识库构建和记忆增强解析
"""

import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Memory-Enhanced Dependency Parsing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 构建知识库
  python run_memory_enhancement.py --build-bank

  # 运行记忆增强解析
  python run_memory_enhancement.py \\
    --input input/Qwen2.5-7B-vi.conllu \\
    --output output/Qwen2.5-7B-vi-enhanced.conllu \\
    --model-path /home16T/ljj/model/Qwen2.5-7B-Instruct

  # 完整流程（先构建知识库，再运行增强）
  python run_memory_enhancement.py \\
    --build-bank \\
    --input input/Qwen2.5-7B-vi.conllu \\
    --output output/Qwen2.5-7B-vi-enhanced.conllu \\
    --model-path /home16T/ljj/model/Qwen2.5-7B-Instruct
        """
    )

    parser.add_argument(
        '--build-bank',
        action='store_true',
        help='构建记忆知识库'
    )

    parser.add_argument(
        '--train-data',
        type=str,
        default='/home16T/ljj/mydata/ud-2.13/tgt/vi-train.conllu',
        help='训练数据路径（用于提取示例）'
    )

    parser.add_argument(
        '--memory-bank',
        type=str,
        default='vi_memory_bank.json',
        help='记忆知识库文件路径'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='输入CoNLL-U文件路径（初次解析结果）'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='输出CoNLL-U文件路径（增强后的结果）'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='/home16T/ljj/model/Qwen2.5-7B-Instruct',
        help='Qwen2.5-7B模型路径'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='运行设备'
    )

    parser.add_argument(
        '--max-sentences',
        type=int,
        default=None,
        help='最多处理的句子数量（用于测试）'
    )

    args = parser.parse_args()

    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent.absolute()

    # 步骤1：构建知识库
    if args.build_bank:
        print("="*60)
        print("Step 1: Building Memory Bank")
        print("="*60)

        # 导入构建模块
        sys.path.insert(0, str(script_dir))
        from build_memory_bank import build_memory_bank

        memory_bank_path = script_dir / args.memory_bank
        train_data_path = args.train_data

        if not os.path.exists(train_data_path):
            print(f"Error: Training data not found: {train_data_path}")
            return 1

        print(f"Training data: {train_data_path}")
        print(f"Output: {memory_bank_path}")

        try:
            build_memory_bank(
                train_data_path=train_data_path,
                output_path=str(memory_bank_path)
            )
            print(f"\n✓ Memory bank built successfully: {memory_bank_path}")
        except Exception as e:
            print(f"\n✗ Error building memory bank: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # 步骤2：运行记忆增强解析
    if args.input and args.output:
        print("\n" + "="*60)
        print("Step 2: Memory-Enhanced Parsing")
        print("="*60)

        # 检查输入文件
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = script_dir / input_path

        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return 1

        # 检查知识库
        memory_bank_path = Path(args.memory_bank)
        if not memory_bank_path.is_absolute():
            memory_bank_path = script_dir / memory_bank_path

        if not memory_bank_path.exists():
            print(f"Error: Memory bank not found: {memory_bank_path}")
            print("Please run with --build-bank first to create the memory bank.")
            return 1

        # 检查模型路径
        if not os.path.exists(args.model_path):
            print(f"Error: Model not found: {args.model_path}")
            return 1

        # 准备输出路径
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Memory Bank: {memory_bank_path}")
        print(f"Model: {args.model_path}")
        print(f"Device: {args.device}")

        # 导入解析器
        sys.path.insert(0, str(script_dir))
        from memory_enhanced_parser import MemoryEnhancedParser

        try:
            # 初始化解析器
            parser = MemoryEnhancedParser(
                model_path=args.model_path,
                memory_bank_path=str(memory_bank_path),
                device=args.device
            )

            # 运行增强
            stats = parser.process_file(
                input_path=str(input_path),
                output_path=str(output_path),
                max_sentences=args.max_sentences
            )

            # 输出统计信息
            print("\n" + "="*60)
            print("Statistics")
            print("="*60)
            print(f"Total sentences: {stats['total_sentences']}")
            print(f"Sentences with weak labels: {stats['sentences_with_weak_labels']}")
            print(f"Sentences modified: {stats['sentences_modified']}")
            print(f"Total modifications: {stats['total_modifications']}")

            if stats['label_modifications']:
                print("\nModifications by label:")
                for label, count in sorted(stats['label_modifications'].items(),
                                          key=lambda x: x[1], reverse=True):
                    print(f"  {label}: {count}")

            print(f"\n✓ Enhanced parsing completed: {output_path}")

        except Exception as e:
            print(f"\n✗ Error during parsing: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # 如果没有指定任何操作
    if not args.build_bank and not (args.input and args.output):
        parser.print_help()
        print("\nError: Please specify --build-bank and/or --input/--output")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
