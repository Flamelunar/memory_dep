import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def get_model_result(base_model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    device = "cuda"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    sentence = "Thanh\tbắt chuyện\tvới\tHùng\tvà\tnói\t:\t\"\tTôi\ttrông\tông\tquen quen\t?\t\"\t."
    CoNLL_U = f""" [CoNLL-U Format]:
    Each parsed word has ten columns separated by TABs, should follow the below structure:
    1 Word index (starts from 1)
    2 Original word form
    3 _
    4 POS tags(**UPPERCASE letters**)
    5 _
    6 _
    7 Headword indices
    8 Dependency type (**lowercase letters**)
    9 _
    10 _

    [Important Notes]:
    1.Each sentence only has one root node.
    2.Words and characters in the original sentence word cannot be missing, keep the word number.
    3.Only output the final [CoNLL-U format].
    
    
    [Example]:        
    [Input]: Tôi\tnhớ\tlời\tanh\tchủ tịch\txã\tBùi Văn Luyến\tnhắc\tđi\tnhắc\tlại\t:\t\"\tCoi bộ\tnhỏ\tnhưng\tquan trọng\tlắm\t!
    [word number]: 19
    [Output]: 
    1	Tôi	_	PRON	_	_	2	nsubj	_	_
    2	nhớ	_	VERB	_	_	0	root	_	_
    3	lời	_	NOUN	_	_	2	obj	_	_
    4	anh	_	NOUN	_	_	5	clf:det	_	_
    5	chủ tịch	_	NOUN	_	_	8	nsubj	_	_
    6	xã	_	NOUN	_	_	5	nmod	_	_
    7	Bùi Văn Luyến	_	PROPN	_	_	5	appos	_	_
    8	nhắc	_	VERB	_	_	3	acl	_	_
    9	đi	_	ADV	_	_	8	flat:redup	_	_
    10	nhắc	_	VERB	_	_	8	flat:redup	_	_
    11	lại	_	ADV	_	_	8	flat:redup	_	_
    12	:	_	PUNCT	_	_	15	punct	_	_
    13	"	_	PUNCT	_	_	15	punct	_	_
    14	Coi bộ	_	ADV	_	_	15	advmod	_	_
    15	nhỏ	_	ADJ	_	_	2	parataxis	_	_
    16	nhưng	_	SCONJ	_	_	17	mark	_	_
    17	quan trọng	_	ADJ	_	_	15	conj	_	_
    18	lắm	_	PART	_	_	17	advmod	_	_
    19	!	_	PUNCT	_	_	15	punct	_	_
    """

    messages = [
        {"role": "system", "content": f"Parse the sentence {sentence} into CoNLL-U format {CoNLL_U}, with each word separated by a tab ('\t')."},
        {"role": "user", "content": "Output sentence's CoNLL-U format"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    def get_result(model_inputs, model):
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            # eos_token_id=tokenizer.get_vocab()["<|c|>"]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    base_model_response = get_result(model_inputs, base_model)
    # fintune_model_response = get_result(model_inputs, fintune_model)

    print("\n" + "="*50)
    print(f"{'待解析的句子':^50}")
    print("="*50)
    print(sentence + "\n")
    
    print("-"*50)
    print(f"{'微调前结果':^50}")
    print("-"*50)
    print(base_model_response + "\n")



if __name__ == '__main__':
    # 示例：使用前4个可见GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 根据实际设备调整
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 新增这行
    base_path = "/home/shared/LLM/Meta-Llama-3.1-8B-Instruct"

    get_model_result(base_path)


# 答案、
# Thanh\tbắt chuyện\tvới\tHùng\tvà\tnói\t:\t\"\tTôi\ttrông\tông\tquen quen\t?\t\"\t.
# 1	Thanh	thanh	PROPN	NNP	_	2	nsubj	_	_
# 2	bắt chuyện	bắt chuyện	VERB	V	_	0	root	_	_
# 3	với	với	ADP	Pre	_	4	case	_	_
# 4	Hùng	Hùng	PROPN	NNP	_	2	obl:with	_	_
# 5	và	và	CCONJ	CC	_	6	cc	_	_
# 6	nói	nói	VERB	V	_	2	conj	_	_
# 7	:	:	PUNCT	:	_	10	punct	_	_
# 8	"	"	PUNCT	``	_	10	punct	_	_
# 9	Tôi	tôi	PRON	Pro	_	10	nsubj	_	_
# 10	trông	trông	VERB	V	_	2	parataxis	_	_
# 11	ông	ông	NOUN	N	_	10	obj	_	_
# 12	quen quen	quen quen	VERB	V	_	10	compound:redup	_	_
# 13	?	?	PUNCT	?	_	10	punct	_	_
# 14	"	"	PUNCT	``	_	10	punct	_	_
# 15	.	.	PUNCT	.	_	2	punct	_	_

