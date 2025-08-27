from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import jsonlines
import json
from tqdm import tqdm

original_model_path = "/home/qianwenhao/LLM/Qwen3-1.7B"

prompts = []
mapping = []
sql_prompt_template = """Given the database schema and the user question, generate the corresponding SQL query.
\[SCHEMA]
{schema}
\[QUESTION]
{question}
"""

with jsonlines.open('data/infer_data_and_gt/dev_infer_data.jsonl', 'r') as reader:
	for obj in reader:
		schema = obj['table_creating']
		question = obj['input']
		prompt = sql_prompt_template.format(schema=schema, question=question)
		prompts.append(prompt)
		mapping.append(obj)

engine_args = {
    "model": original_model_path,
    "trust_remote_code": True,
    "dtype": "bfloat16",               # 优先 bf16（4090 支持），速度快且稳定
    "max_model_len": 1152,  # 2560 或更大
    "tensor_parallel_size": 2,         # 两张卡 → tensor parallel
    "pipeline_parallel_size": 1,       # 不用 pipeline，保持简单
    "disable_log_stats": True,
    "enable_lora": True,				# 启用 LoRA
    "gpu_memory_utilization": 0.9,     # 90% 显存利用
    "max_num_seqs": 32,                # 批处理容量，提升吞吐
}

llm = LLM(**engine_args)

tokenizer = AutoTokenizer.from_pretrained(original_model_path)
sampling_params = SamplingParams(
    repetition_penalty=1.1,          		# 轻度惩罚重复
    temperature=0.2,                		# 低温度，输出更稳定
    top_p=0.9,                      		# 保持多样性，但不完全贪心
    top_k=-1,                       		# 不限制 top_k
    stop_token_ids=[tokenizer.eos_token_id], # 以 EOS 停止
    max_tokens=1024,                		# SQL 语句够用
    skip_special_tokens=True,
    seed=42                         		# 保证结果复现
)

lora_request = LoRARequest("default", 1, 'saves/qwen3-1.7B_no_filter_data/lora/sft')

results = []
batch_size = 16
for i in tqdm(range(0, len(prompts), batch_size)):
    batch_prompts = prompts[i: i + batch_size]
    outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
    for j, out in enumerate(outputs):
        item = mapping[i + j]
        item['pred'] = out.outputs[0].text
        results.append(item)

with open('data/infer_data_and_gt/infer_res/infer_out_qwen3_1p7B_no_filter_data.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
