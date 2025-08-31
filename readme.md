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

---

## 📊 模型表现

* 评估框架：[test-suite-sql-eval](https://github.com/Qianvenh/Text2SQL-Qwen3-1p7B/tree/main/test-suite-sql-eval)
* 测试集：[Spider dev dataset](https://github.com/eosphoros-ai/DB-GPT-Hub/blob/main/src/dbgpt-hub-sql/dbgpt_hub_sql/data/eval_data/dev_sql.json)
* 评价指标：**Execution Accuracy (执行准确率)**

### 🔧 训练-测试差异缓解

* **问题**：训练集中的 SCHEMA 用 `CREATE TABLE` 表示，而测试集中的 SCHEMA 为自然语言描述
* **解决方案**：利用 SoTA 大模型将测试集 SCHEMA 转换为 `CREATE TABLE` 语句

  * 脚本：[response\_table\_creating.py](https://github.com/Qianvenh/Text2SQL-Qwen3-1p7B/blob/main/data/dataset_process/response_table_creating.py)

### 📈 结果对比（Execution Accuracy）

| 模型                      | Easy      | Medium    | Hard      | Extra     | All       |
| ----------------------- | --------- | --------- | --------- | --------- | --------- |
| baseline                | **85.9%** | **58.7%** | 43.1%     | 15.1%     | 55.6%     |
| qwen3\_1p7B\_lora\_32r  | 79.8%     | 47.3%     | 37.4%     | 19.3%     | 48.9%     |
| qwen3\_1p7B\_lora\_128r | 73.4%     | 42.6%     | 32.2%     | 15.1%     | 43.8%     |
| qwen3\_1p7B\_full       | 85.5%     | 57.6%     | **52.3%** | **21.7%** | **57.6%** |

---

## 🔍 关键观察

* **全参数微调 (qwen3\_1p7B\_full)** 整体表现最佳，尤其在 Medium、Hard、Extra 难度上
* **baseline** 在 Easy 难度领先，但在复杂查询上性能下降明显
* **LoRA rank 增大 (32r → 128r)** 反而导致性能下降
* **Extra 难度** 对所有模型仍然是显著挑战

---

## 📊 可视化表现

（图表待补充）

---
