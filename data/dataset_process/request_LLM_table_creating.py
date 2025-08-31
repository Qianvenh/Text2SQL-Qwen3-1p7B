from openai import OpenAI
import os
import ijson
import jsonlines

clinet = OpenAI(
	base_url='https://api.moonshot.cn/v1',
	api_key=os.getenv('KIMI_API_KEY')
)

system_prompt = """## Role
You are a world class SQL expert.
## task
You will be provided with a description of database schema. Your task is to generate the accurate table-creating SQL and adheres to the provided schema.
## output format
<sql>{{your_sql}}</sql>
"""

schema_set = set()
with open('./dev_sql.json', 'rt') as f:
	for idx, item in enumerate(ijson.items(f, 'item')):
		try:
			if item['instruction'] in schema_set:
				continue
			response = clinet.chat.completions.create(
				model="kimi-k2-0711-preview",
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": f"Description of Database Schema: {item['instruction']}"}
				],
				temperature=0.15,
				max_tokens=1024
			)
			
			item['table_creating'] = response.choices[0].message.content.strip()
			item['idx'] = idx
			
			with jsonlines.open('./dev_sql_llm_respnose.jsonl', mode='a') as writer:
				writer.write(item)
			schema_set.add(item['instruction'])
			
			if (idx + 1) % 10 == 0:
				print(f"Processed {idx + 1} items")
		except Exception as e:
			print(f"Error processing item {idx}: {e}")
		