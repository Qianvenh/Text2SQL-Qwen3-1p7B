#!/usr/bin/env python3
"""
Text2SQL Vector Database Builder using Milvus with Qwen3-Embedding
This script processes text2sql.json file using ijson to extract SCHEMA and QUESTION,
then builds a vector database with langchain-milvus using local Qwen3-Embedding model.
"""

import ijson
import re
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from tqdm import tqdm
from pymilvus import connections, utility
from config import EMBEDDING_CONFIG, MILVUS_CONFIG, TEXT_SPLITTER_CONFIG, LOGGING_CONFIG

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]), 
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class Text2SQLVectorDBQwen:
    def __init__(self, 
                 data_file_path: str,
                 collection_name: str = None):
        """
        初始化Text2SQL向量数据库构建器（使用Qwen3嵌入模型）
        
        Args:
            data_file_path: text2sql.json文件路径
            collection_name: Milvus集合名称
        """
        self.data_file_path = Path(data_file_path)
        self.collection_name = collection_name or MILVUS_CONFIG["collection_name"]
        
        # 初始化本地Qwen3嵌入模型
        logger.info(f"初始化本地Qwen3嵌入模型: {EMBEDDING_CONFIG['model_name']}")
        
        # 检查模型路径是否存在
        model_path = Path(EMBEDDING_CONFIG['model_name'])
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_CONFIG['model_name'],
                model_kwargs={
                    'device': EMBEDDING_CONFIG['device'],
                    'trust_remote_code': EMBEDDING_CONFIG['trust_remote_code']
                },
                encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG['normalize_embeddings']}
            )
            logger.info("Qwen3嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"初始化Qwen3嵌入模型失败: {e}")
            raise
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
            chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
            length_function=len,
        )
    
    def extract_schema_and_question(self, content: str) -> Tuple[str, str]:
        """
        从conversations[0]['value']中提取[SCHEMA]和[QUESTION]
        
        Args:
            content: conversations[0]['value']的内容
            
        Returns:
            Tuple[schema, question]: 提取的schema和question
        """
        schema_pattern = r'\[SCHEMA\]\s*(.*?)\s*\[QUESTION\]'
        question_pattern = r'\[QUESTION\]\s*(.*?)$'
        
        schema_match = re.search(schema_pattern, content, re.DOTALL)
        question_match = re.search(question_pattern, content, re.DOTALL)
        
        schema = schema_match.group(1).strip() if schema_match else ""
        question = question_match.group(1).strip() if question_match else ""
        
        return schema, question
    
    def process_json_data(self, max_records: int = None) -> List[Dict[str, Any]]:
        """
        使用ijson流式处理JSON文件，提取数据
        
        Args:
            max_records: 最大处理记录数（用于测试）
            
        Returns:
            List[Dict]: 处理后的数据列表
        """
        logger.info(f"开始处理文件: {self.data_file_path}")
        processed_data = []
        
        try:
            with open(self.data_file_path, 'rb') as file:
                # 使用ijson解析JSON数组中的每个item
                parser = ijson.items(file, 'item')
                
                for idx, item in enumerate(parser):
                    # 如果设置了最大记录数限制
                    if max_records and idx >= max_records:
                        logger.info(f"达到最大处理记录数限制: {max_records}")
                        break
                        
                    try:
                        conversations = item.get('conversations', [])
                        if len(conversations) >= 2:
                            # 提取用户输入 (conversations[0]['value'])
                            user_value = conversations[0].get('value', '')
                            # 提取助手回复 (conversations[1]['value']) 作为SQL答案
                            assistant_value = conversations[1].get('value', '')
                            
                            # 从用户输入中提取SCHEMA和QUESTION
                            schema, question = self.extract_schema_and_question(user_value)
                            
                            if schema and question and assistant_value:
                                processed_data.append({
                                    'id': f"text2sql_{idx}",
                                    'schema': schema,
                                    'question': question,
                                    'sql_answer': assistant_value
                                })
                        
                        # 每处理1000条记录打印进度
                        if (idx + 1) % 1000 == 0:
                            logger.info(f"已处理 {idx + 1} 条记录，有效数据: {len(processed_data)} 条")
                            
                    except Exception as e:
                        logger.warning(f"处理第 {idx} 条记录时出错: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"读取文件时出错: {e}")
            raise
        
        logger.info(f"总共处理了 {len(processed_data)} 条有效记录")
        return processed_data
    
    def create_documents(self, processed_data: List[Dict[str, Any]]) -> List[Document]:
        """
        创建Langchain文档对象
        
        Args:
            processed_data: 处理后的数据
            
        Returns:
            List[Document]: Langchain文档对象列表
        """
        logger.info("创建Langchain文档对象")
        documents = []
        
        for data in processed_data:
            # 用问题作为文档内容，即数据库的索引embedding
            doc = Document(
                page_content=data['question'],
                metadata={
                    'id': data['id'],
                    'schema': data['schema'],
                    'question': data['question'],
                    'sql_answer': data['sql_answer']
                }
            )
            documents.append(doc)
        
        logger.info(f"创建了 {len(documents)} 个文档")
        return documents
    
    def build_vector_database(self, documents: List[Document]) -> Milvus:
        """
        构建向量数据库
        
        Args:
            documents: 文档列表
            
        Returns:
            Milvus: 向量数据库实例
        """
        logger.info("开始使用Qwen3嵌入模型构建向量数据库")
        
        try:
            # 连接Milvus服务器, 新建前删除同名集合
            connections.connect(uri=MILVUS_CONFIG["uri"])
            if utility.has_collection(self.collection_name):
                choice = input(f"集合 {self.collection_name} 已存在，要继续删除吗？(y/n): ")
                if choice.lower() != 'y':
                    logger.info("用户取消操作")
                    exit(0)
                else:
                    utility.drop_collection(self.collection_name)
                    print(f"Dropped existing collection: {self.collection_name}")

            # 创建Milvus向量存储
            vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": MILVUS_CONFIG["uri"]}
            )

            batch_size = 2048  # 按显存情况调整
            for i in tqdm(range(0, len(documents), batch_size)):
                batch = documents[i:i+batch_size]
                vector_store.add_documents(batch)

            
            logger.info(f"向量数据库构建完成，集合名称: {self.collection_name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"构建向量数据库时出错: {e}")
            raise
    
    def search_similar(self, vector_store: Milvus, query: str, k: int = 5) -> List[Document]:
        """
        搜索相似文档
        
        Args:
            vector_store: 向量数据库实例
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Document]: 相似文档列表
        """
        logger.info(f"搜索查询: {query}")
        results = vector_store.similarity_search(query, k=k)
        
        for i, doc in enumerate(results, 1):
            logger.info(f"结果 {i}:")
            logger.info(f"  问题: {doc.metadata.get('question', '')[:100]}...")
            logger.info(f"  SQL: {doc.metadata.get('sql_answer', '')[:100]}...")
        
        return results
    
    def run(self, max_records: int = None) -> Milvus:
        """
        运行完整的向量数据库构建流程
        
        Args:
            max_records: 最大处理记录数（用于测试）
            
        Returns:
            Milvus: 构建好的向量数据库实例
        """
        logger.info("开始Text2SQL向量数据库构建流程（使用Qwen3嵌入模型）")
        
        # 1. 处理JSON数据
        processed_data = self.process_json_data(max_records)
        
        if not processed_data:
            raise ValueError("没有处理到有效数据")
        
        # 2. 创建文档
        documents = self.create_documents(processed_data)
        
        # 3. 构建向量数据库
        vector_store = self.build_vector_database(documents)
        
        logger.info("Text2SQL向量数据库构建完成")
        return vector_store


def main():
    """主函数"""
    # 配置参数
    data_file = "../data/text2sql.json"
    
    # 检查数据文件是否存在
    if not Path(data_file).exists():
        logger.error(f"数据文件不存在: {data_file}")
        return
    
    try:
        # 创建向量数据库构建器
        db_builder = Text2SQLVectorDBQwen(
            data_file_path=data_file,
            collection_name="text2sql_qwen_collection"
        )
        
        logger.info("开始构建向量数据库")
        vector_store = db_builder.run()
        
        # 测试搜索功能
        test_queries = [
            "What is the total sales by region?",
            "Show all employees in the company",
            "Count the number of customers"
        ]
        
        for query in test_queries:
            logger.info(f"\n测试搜索功能，查询: {query}")
            results = db_builder.search_similar(vector_store, query, k=3)
        
        logger.info("向量数据库构建和测试完成")
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise


if __name__ == "__main__":
    main()
