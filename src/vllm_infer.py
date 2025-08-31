from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import jsonlines
import json
import yaml
import sys
import os
from tqdm import tqdm


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default config path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'vllm_infer_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Extract configuration
    model_path = config['model_name_or_path']
    input_file = config['input_file'] if not config['use_reference_prompt'] else config['input_file_with_reference']
    output_file = config['output_file']
    batch_size = config['batch_size']
    
    prompts = []
    mapping = []

    def get_prompt(item):
        schema = item['table_creating']
        question = item['input']
        if config['use_reference_prompt']:
            if config['reference_type'] == 'normal':
                reference_items = item['normal_reference_prompt'] 
            elif config['reference_type'] == 'reranked':
                reference_items = item['reranked_reference_prompt']
            elif config['reference_type'] == 'reranked_sqlkey':
                reference_items = item['ranked_reference_prompt_sql_as_key']
            return config['prompt_reference_template'].format(
                schema=schema, question=question, reference_list=reference_items)
        return config['prompt_template'].format(schema=schema, question=question)


    with jsonlines.open(input_file, 'r') as reader:
        for obj in reader:
            prompts.append(get_prompt(obj))
            mapping.append(obj)

    # Engine arguments from config
    engine_args = {
        "model": model_path,
        "trust_remote_code": config['trust_remote_code'],
        "dtype": config['dtype'],
        "max_model_len": config['max_model_len'],
        "tensor_parallel_size": config['tensor_parallel_size'],
        "pipeline_parallel_size": config['pipeline_parallel_size'],
        "disable_log_stats": config['disable_log_stats'],
        "enable_lora": config['enable_lora'],
        "gpu_memory_utilization": config['gpu_memory_utilization'],
        "max_num_seqs": config['max_num_seqs'],
        "max_lora_rank": config['max_lora_rank'],
    }

    llm = LLM(**engine_args)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Sampling parameters from config
    sampling_params = SamplingParams(
        repetition_penalty=config['repetition_penalty'],
        temperature=config['temperature'],
        top_p=config['top_p'],
        top_k=config['top_k'],
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=config['max_tokens'],
        skip_special_tokens=config['skip_special_tokens'],
        seed=config['seed']
    )

    # LoRA request from config
    lora_adapter_path = config.get('lora_adapter_path')
    if lora_adapter_path and lora_adapter_path != 'null':
        lora_request = LoRARequest("default", 1, lora_adapter_path)
    else:
        lora_request = None

    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i: i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
        for j, out in enumerate(outputs):
            item = mapping[i + j]
            item['pred'] = out.outputs[0].text
            results.append(item)

    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Inference completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
