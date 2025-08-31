#!/usr/bin/env python3
"""
Configuration file for Text2SQL RAG system
"""

import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
LOGS_DIR = BASE_DIR / "logs"

# 数据文件配置
TEXT2SQL_JSON_FILE = DATA_DIR / "text2sql.json"

# Milvus配置 - 使用本地内存模式(Milvus Lite)
MILVUS_CONFIG = {
    "uri": "./milvus_lite.db",  # 使用本地文件数据库
    "collection_name": "text2sql_qwen_collection"
}

# 嵌入模型配置 - 使用本地Qwen3-Embedding模型
EMBEDDING_CONFIG = {
    "model_name": "/home/qianwenhao/LLM/Qwen3-Embedding-0.6B",
    "device": "cuda",  # 使用GPU处理
    "normalize_embeddings": True,
    "trust_remote_code": True  # Qwen模型需要此参数
}

RERANKER_CONFIG = {
    "model_name": "/home/qianwenhao/LLM/Qwen3-Reranker-0.6B",
	"device": "cuda"
}

# 文本分割配置
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 100
}

# 搜索配置
SEARCH_CONFIG = {
    "default_k": 4,  # 默认返回结果数量
    "max_k": 30      # 最大返回结果数量
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "rag_system.log"
}

# 确保日志目录存在
LOGS_DIR.mkdir(exist_ok=True)
