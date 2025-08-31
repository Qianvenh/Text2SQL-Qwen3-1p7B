#!/bin/bash

# Exp_name="qwen3_1p7B_zs(baseline)"
# Exp_name="qwen3_1p7B_tp_dp_P2trainP2"
# Exp_name="qwen3_1p7B_rag_only_embedding"
# Exp_name="qwen3_1p7B_rag_embeddingNreranked"
Exp_name="qwen3_1p7B_rag_embeddingNreranked_sqlkey"

cd test-suite-sql-eval

python evaluation.py --gold evaluation_examples/$Exp_name/gold.txt \
	--pred evaluation_examples/$Exp_name/predict.txt \
	--table tables.json \
	--db database \
	--plug_value
