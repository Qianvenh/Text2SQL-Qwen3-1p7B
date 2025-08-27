#!/bin/bash

cd "$(dirname $0)/test-suite-sql-eval"
python evaluation.py --gold evaluation_examples/gold.txt \
	--pred evaluation_examples/predict.txt \
	--table tables.json \
	--db database \
	--plug_value
