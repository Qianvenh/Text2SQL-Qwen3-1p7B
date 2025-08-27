import jsonlines
import ijson
from tqdm import tqdm


with jsonlines.open('./dev_sql_llm_respnose.jsonl', 'r') as f:
	description_map = {}
	for item in f:
		description_map[item['instruction']] = item['table_creating']

with open('./dev_sql.json', 'rt') as reader:
	with jsonlines.open('../infer_data_and_gt/dev_infer_data.jsonl', 'w') as writer:
		for item in tqdm(ijson.items(reader, 'item')):
			if item['instruction'] in description_map:
				item['table_creating'] = description_map[item['instruction']]
				writer.write(item)
			else:
				print(f"Instruction not found: {item['instruction']}, {item['idx']}")
