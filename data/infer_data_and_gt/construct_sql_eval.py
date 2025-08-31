import ijson
import os
import pandas as pd

# input_file = './infer_res/qwen3_1p7B_tp_dp_P2trainP2.json'
# input_file = './infer_res/qwen3_1p7B_rag_only_embedding.json'
# input_file = './infer_res/qwen3_1p7B_rag_embeddingNranker.json'
input_file = './infer_res/qwen3_1p7B_rag_embeddingNreranker_sqlkey.json'
# exp_name = 'qwen3_1p7B_tp_dp_P2trainP2'
# exp_name = 'qwen3_1p7B_rag_only_embedding'
# exp_name = 'qwen3_1p7B_rag_embeddingNreranked'
exp_name = 'qwen3_1p7B_rag_embeddingNreranked_sqlkey'
target_path = '../../test-suite-sql-eval/evaluation_examples/'
target_path = os.path.join(target_path, exp_name)
if not os.path.exists(target_path):
	os.makedirs(target_path)

df = pd.read_parquet('./val_db_info.parquet', engine='pyarrow')

with open(input_file, 'rt') as f:
	with open(os.path.join(target_path, 'predict.txt'), 'w') as pred_w:
		with open(os.path.join(target_path, 'gold.txt'), 'w') as gold_w:
			for item, db_info in zip(ijson.items(f, 'item'), df.iterrows()):
				gold_w.write(f"{item['output'].strip()}\t{db_info[1]['db_id']}\n\n")
				try:
					# pre_sql = item['pred'].split('</think>\n\n')[-1].replace('\n', ' ').strip()
					pre_sql = item['pred'].replace('\n', ' ').strip()
					if pre_sql == '':
						pre_sql = 'None'
						print(item['pred'])
					pred_w.write(pre_sql + '\n\n')
				except Exception as e:
					pred_w.write('None \n\n')
					