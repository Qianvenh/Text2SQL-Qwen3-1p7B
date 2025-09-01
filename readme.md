# Text2SQL-Qwen3-1.7B

Fine-tuning and context engineering of **Qwen3-1.7B** for the Text-to-SQL task.

## 📖 训练设置

* 基于框架：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
* 集成模块：[ROLL/MCoreAdapter](https://github.com/alibaba/ROLL/tree/main/mcore_adapter)，用于使 LLaMA-Factory 兼容 Megatron
* 设备：**NVIDIA GeForce RTX 4090 (24GB) × 4**

### 训练方式

* **LoRA SFT**：仅微调 `Q_proj` 与 `V_proj`，采用 **DeepSpeed ZeRO-3** 配置
* **Full Parameters SFT**：全参数微调，采用 **DeepSpeed ZeRO-2 + Megatron Tensor Parallelism (TP)** 配置

### 数据集

* 训练集：[fahmiaziz/text2sql-dataset](https://huggingface.co/datasets/fahmiaziz/text2sql-dataset)
* 数据预处理：移除了包含测试样本的部分，详见 [dataset\_process.ipynb](https://github.com/Qianvenh/Text2SQL-Qwen3-1p7B/blob/main/data/dataset_process/dataset_process.ipynb)

### 数据格式示例

```text
<|im_start|>system
Given the database schema and the user question, generate the corresponding SQL query. 
The output must be only a valid SQL query, without explanations, comments, or extra text.
<|im_end|>
<|im_start|>user

[SCHEMA]
CREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT);
INSERT INTO salesperson (salesperson_id, name, region) VALUES 
(1, 'John Doe', 'North'), 
(2, 'Jane Smith', 'South');

CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE);
INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES 
(1, 1, 120, '2021-01-01'), 
(2, 1, 150, '2021-02-01'), 
(3, 2, 180, '2021-01-01');

[QUESTION]
What is the total volume of timber sold by each salesperson, sorted by salesperson?
<|im_end|>
<|im_start|>assistant
SELECT salesperson_id, name, SUM(volume) AS total_volume 
FROM timber_sales 
JOIN salesperson ON timber_sales.salesperson_id = salesperson.salesperson_id 
GROUP BY salesperson_id, name 
ORDER BY total_volume DESC;
<|im_end|>
```

## 🧩 上下文工程方案

### 知识库构建

- 从 `data/text2sql.json` 中读取训练集数据
- 使用 `ijson` 流式处理，提取 `[QUESTION]` 字段作为向量索引
- 通过 `Qwen3-Embedding` 将 `[QUESTION]` 向量化
- 存储于 `Milvus` 向量数据库

### 检索与排序

- 基于用户查询进行相似性搜索，获得粗排候选样本
- 使用 `Qwen3-Reranker` 对结果进行精排
- 取 Top-k 样本

### 上下文增强

- 将检索到的相关样本的 `[QUESTION]` 和 `[ANSWER]` 加入模型上下文，构建为 Few-shot Examples
- 生成最终 SQL 答案

### 数据格式实例

``` text
 <|im_start|>system
Given the database schema and the user question, generate the corresponding SQL query.
The output must be only a valid SQL query, without explanations, comments, or extra text.
Here are some reference examples that might help you:
<reference_list> 
  <reference_item id="0"> 
  [QUESTION] What is the total number of speakers for each language? 
  [ANSWER] SELECT Language, SUM(SpeakerCount) FROM LanguageSpeakers GROUP BY Language; 
  </reference_item> 
  <reference_item id="1"> 
  [QUESTION] How many cinema do we have? 
  [ANSWER] SELECT COUNT(*) FROM cinema; 
  </reference_item> 
  <reference_item id="2"> 
  [QUESTION] How many tracks do we have? 
  [ANSWER] SELECT COUNT(*) FROM track; 
  </reference_item> 
  <reference_item id="3"> 
  [QUESTION] How many faculty do we have? 
  [ANSWER] SELECT COUNT(*) FROM Faculty; 
  </reference_item> 
</reference_list>
<|im_end|>
<|im_start|>user
[SCHEMA]
<sql> CREATE TABLE stadium ( stadium_id INT PRIMARY KEY, location VARCHAR(255), name VARCHAR(255), capacity INT, highest INT, lowest INT, average INT ); CREATE TABLE singer ( singer_id INT PRIMARY KEY, name VARCHAR(255), country VARCHAR(255), song_name VARCHAR(255), song_release_year INT, age INT, is_male BOOLEAN ); CREATE TABLE concert ( concert_id INT PRIMARY KEY, concert_name VARCHAR(255), theme VARCHAR(255), stadium_id INT, year INT, FOREIGN KEY (stadium_id) REFERENCES stadium(stadium_id) ); CREATE TABLE singer_in_concert ( concert_id INT, singer_id INT, PRIMARY KEY (concert_id, singer_id), FOREIGN KEY (concert_id) REFERENCES concert(concert_id), FOREIGN KEY (singer_id) REFERENCES singer(singer_id) ); </sql>
[QUESTION]
How many singers do we have?
<|im_end|>
<|im_start|>assistant
<think>

</think>

[ANSWER]
SELECT COUNT(*) FROM singer;
```

## 🛠️ 项目结构与工具说明

### Scripts 目录脚本说明

项目的 `scripts/` 目录包含了完整的训练、推理和评估流程脚本：

* `qwen3_1p7B_train.sh`: LoRA 微调训练脚本，使用 LLaMA-Factory 框架
* `megatron_deepspeed_train.sh`: Megatron + DeepSpeed 全参数微调训练脚本
* `merge_hf_lora.sh`: LoRA 权重合并脚本，将训练好的 LoRA 权重合并到基础模型
* `mg2hf.sh`: Megatron 格式模型转换为 HuggingFace 格式的脚本
* `infer_vllm.sh`: 使用 vLLM 框架进行模型推理的脚本
* `evaluate_sql.sh`: SQL 评估脚本，计算模型在测试集上的执行准确率

### RAG 上下文工程模块

位于 `src/RAG/` 目录的向量检索与重排序系统：

`vector_db_builder_qwen.py`: 向量数据库构建工具

- 使用 `ijson` 流式处理训练数据 `text2sql.json`
基于 `Qwen3-Embedding` 模型生成问题向量
构建 `Milvus` 向量数据库索引，存储问题-SQL对应关系

`vector_search_with_rerank.py`: 检索与重排序工具

- 基于向量相似度进行粗排检索
- 使用 `Qwen3-Reranker` 进行精排序
- 支持基于问题或 SQL 答案的重排序策略
- 生成结构化的 Few-shot 示例上下文

## 📊 模型表现

* 评估框架：[test-suite-sql-eval](https://github.com/Qianvenh/Text2SQL-Qwen3-1p7B/tree/main/test-suite-sql-eval)
* 测试集：[Spider dev dataset](https://github.com/eosphoros-ai/DB-GPT-Hub/blob/main/src/dbgpt-hub-sql/dbgpt_hub_sql/data/eval_data/dev_sql.json)
* 评价指标：**Execution Accuracy (执行准确率)**

### 🔧 训练-测试差异缓解

* **问题**：训练集中的 SCHEMA 用 `CREATE TABLE` 表示，而测试集中的 SCHEMA 为自然语言描述
* **解决方案**：利用 SoTA 大模型将测试集 SCHEMA 转换为 `CREATE TABLE` 语句

  * 脚本：[request\_LLM\_table\_creating.py](https://github.com/Qianvenh/Text2SQL-Qwen3-1p7B/blob/main/data/dataset_process/request_LLM_table_creating.py)

### 📈 结果对比（Execution Accuracy）

| 模型                      | Easy      | Medium    | Hard      | Extra     | All       |
| ----------------------- | --------- | --------- | --------- | --------- | --------- |
| baseline                | **85.9%** | **58.7%** | 43.1%     | 15.1%     | 55.6%     |
| RAG_only_embedding	    | 83.9%	    | 57.0%	    | 39.1%	    | 12.7%	    | 53.3%     | 
| RAG_embedding+rerank	  | 83.5%	    | 56.7%	    | 41.4%	    | 16.3%	    | 54.1%     |
| qwen3\_1p7B\_lora\_32r  | 79.8%     | 47.3%     | 37.4%     | 19.3%     | 48.9%     |
| qwen3\_1p7B\_lora\_128r | 73.4%     | 42.6%     | 32.2%     | 15.1%     | 43.8%     |
| qwen3\_1p7B\_full       | 85.5%     | 57.6%     | **52.3%** | **21.7%** | **57.6%** |


## 🔍 关键观察

* **全参数微调 (qwen3\_1p7B\_full)** 整体表现最佳，尤其在 Medium、Hard、Extra 难度上
* **baseline** 在 Easy 难度领先，但在复杂查询上性能下降明显
* **LoRA rank 增大 (32r → 128r)** 反而导致性能下降
* **Extra 难度** 对所有模型仍然是显著挑战\
* **RAG_only_embedding** 在所有难度上表现中等，整体准确率为53.3%
* **RAG_embedding+rerank** 通过重排序机制在Extra难度上有所提升，但整体准确率仍不高
* **RAG方法 和 LoRA微调** 在当前训练集和模型上相比全参数微调模型仍有不足，但都能一定程度提升 Extra 难度上的性能


## 📊 模型表现可视化

![难度级别对比柱状图](figs/model_performance_comparison.png)
