# Text2SQL RAG系统

这是一个基于向量数据库的Text2SQL检索增强生成(RAG)系统，使用本地Qwen3-Embedding-0.6B模型进行文本嵌入。

## 系统架构

```
data/text2sql.json (原始数据)
    ↓ (ijson流式处理)
提取 [SCHEMA] 和 [QUESTION]
    ↓ (Qwen3-Embedding编码)
向量化 → Milvus向量数据库
    ↓ (相似性搜索)
检索相关SQL示例
```

## 功能特性

1. **流式数据处理**: 使用ijson处理大型JSON文件，内存效率高
2. **本地嵌入模型**: 使用Qwen3-Embedding-0.6B进行文本向量化
3. **向量数据库**: 基于Milvus构建高效的向量搜索系统
4. **智能检索**: 支持基于问题和模式的相似性搜索
5. **批量处理**: 支持批量搜索和测试功能

## 文件结构

```
RAG/
├── config.py                    # 配置文件
├── vector_db_builder_qwen.py    # 向量数据库构建脚本
├── vector_search_qwen.py        # 向量数据库搜索脚本
├── run_rag_system.sh           # 启动脚本
├── requirements.txt            # 依赖包列表
└── README.md                   # 说明文档
```

## 安装依赖

系统已在llamafactory环境中安装以下依赖包：

- ijson >= 3.4.0
- langchain >= 0.3.0
- langchain-milvus >= 0.2.0
- sentence-transformers >= 5.0.0
- pymilvus >= 2.6.0
- transformers >= 4.30.0

## 使用方法

### 1. 快速启动

```bash
cd RAG
./run_rag_system.sh
```

### 2. 构建向量数据库

```bash
cd RAG
python3 vector_db_builder_qwen.py
```

该脚本会：
- 使用ijson流式处理`../data/text2sql.json`
- 从每条记录的`conversations[0]['value']`中提取`[SCHEMA]`和`[QUESTION]`
- 使用Qwen3嵌入模型生成向量
- 将`conversations[1]['value']`作为SQL答案存储
- 在Milvus中建立向量索引

### 3. 搜索向量数据库

```bash
cd RAG
python3 vector_search_qwen.py
```

支持以下功能：
- 交互式问题搜索
- 批量测试搜索
- 相似度评分显示

## 配置说明

### 嵌入模型配置
```python
EMBEDDING_CONFIG = {
    "model_name": "/home/qianwenhao/LLM/Qwen3-Embedding-0.6B",
    "device": "cpu",  # 可改为 "cuda"
    "normalize_embeddings": True,
    "trust_remote_code": True
}
```

### Milvus配置
```python
MILVUS_CONFIG = {
    "host": "localhost",
    "port": 19530,
    "collection_name": "text2sql_collection"
}
```

## 数据格式

### 输入数据格式 (text2sql.json)
```json
[
  {
    "conversations": [
      {
        "from": "user",
        "value": "\n[SCHEMA]\nCREATE TABLE...\n[QUESTION]\nWhat is...?\n"
      },
      {
        "from": "assistant",
        "value": "SELECT * FROM..."
      }
    ]
  }
]
```

### 向量数据库存储格式
```python
{
    'id': 'text2sql_0',
    'schema': 'CREATE TABLE...',
    'question': 'What is...?',
    'sql_answer': 'SELECT * FROM...',
    'combined_text': 'Schema: ... Question: ...'
}
```

## 使用示例

### 搜索示例
```python
from vector_search_qwen import Text2SQLSearcherQwen

# 初始化搜索器
searcher = Text2SQLSearcherQwen()

# 搜索相似问题
results = searcher.search_by_question("What is the total sales?", k=5)

# 获取格式化示例用于提示工程
examples = searcher.get_similar_examples("Count customers", k=3)
```

### 批量搜索
```python
questions = [
    "What is the total sales by region?",
    "Show all employees",
    "Count customers"
]
batch_results = searcher.batch_search(questions, k=3)
```

## 性能优化

1. **GPU加速**: 修改配置中的`device`为`"cuda"`
2. **批量处理**: 使用`max_records`参数限制处理数量进行测试
3. **内存优化**: ijson流式处理避免一次性加载大文件

## 故障排除

### 常见问题

1. **模型路径错误**
   ```
   FileNotFoundError: 模型路径不存在
   ```
   确保Qwen3模型路径正确：`/home/qianwenhao/LLM/Qwen3-Embedding-0.6B`

2. **Milvus连接失败**
   ```
   连接向量数据库失败
   ```
   确保Milvus服务正在运行，或使用内存模式

3. **依赖包问题**
   ```bash
   pip install -r requirements.txt
   ```

### 日志调试
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

1. **多语言支持**: 扩展到支持中文问题
2. **高级搜索**: 添加过滤条件和排序选项
3. **API接口**: 提供REST API服务
4. **缓存机制**: 添加查询结果缓存

## 许可证

本项目基于MIT许可证开源。
