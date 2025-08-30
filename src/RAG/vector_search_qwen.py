#!/usr/bin/env python3
"""
Text2SQL Vector Database Search with Qwen3-Embedding
This script provides search functionality for the built vector database using Qwen3 embedding model.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_milvus import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from config import EMBEDDING_CONFIG, MILVUS_CONFIG, SEARCH_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Text2SQLSearcherQwen:
    def __init__(self, collection_name: str = None):
        """
        初始化Text2SQL搜索器（使用Qwen3嵌入模型）
        
        Args:
            collection_name: Milvus集合名称
        """
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
        
        # 连接到现有的Milvus集合
        try:
            self.vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args={
                    "uri": MILVUS_CONFIG["uri"]
            }
            )
            logger.info(f"已连接到向量数据库集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"连接向量数据库失败: {e}")
            raise
    
    def search_by_question(self, question: str, k: int = None) -> List[Dict[str, Any]]:
        """
        根据问题搜索相似的SQL示例
        
        Args:
            question: 用户问题
            k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        k = k or SEARCH_CONFIG["default_k"]
        logger.info(f"搜索问题: {question}")
        
        try:
            # 使用问题进行相似性搜索
            results = self.vector_store.similarity_search_with_score(question, k=k)
            
            search_results = []
            for doc, score in results:
                result = {
                    'score': score,
                    'question': doc.metadata.get('question', ''),
                    'schema': doc.metadata.get('schema', ''),
                    'sql_answer': doc.metadata.get('sql_answer', ''),
                    'id': doc.metadata.get('id', '')
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索时出错: {e}")
            return []
    
    def search_by_schema(self, schema_keyword: str, k: int = None) -> List[Dict[str, Any]]:
        """
        根据模式关键词搜索相关的SQL示例
        
        Args:
            schema_keyword: 模式关键词
            k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        query = f"Schema: {schema_keyword}"
        return self.search_by_question(query, k)
    
    def get_similar_examples(self, question: str, k: int = 3) -> str:
        """
        获取相似示例的格式化字符串，用于提示工程
        
        Args:
            question: 用户问题
            k: 返回结果数量
            
        Returns:
            str: 格式化的示例字符串
        """
        results = self.search_by_question(question, k)
        
        if not results:
            return "没有找到相关示例。"
        
        examples = []
        for i, result in enumerate(results, 1):
            example = f"""
示例 {i} (相似度: {result['score']:.4f}):
问题: {result['question']}
模式: {result['schema'][:200]}...
SQL: {result['sql_answer']}
"""
            examples.append(example)
        
        return "\n".join(examples)
    
    def display_search_results(self, results: List[Dict[str, Any]]):
        """
        显示搜索结果
        
        Args:
            results: 搜索结果列表
        """
        if not results:
            print("没有找到相关结果。")
            return
        
        print(f"\n找到 {len(results)} 个相关结果:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i} (相似度分数: {result['score']:.4f}):")
            print(f"ID: {result['id']}")
            print(f"问题: {result['question']}")
            print(f"模式: {result['schema'][:300]}...")
            print(f"SQL: {result['sql_answer']}")
            print("-" * 80)
    
    def batch_search(self, questions: List[str], k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量搜索多个问题
        
        Args:
            questions: 问题列表
            k: 每个问题返回的结果数量
            
        Returns:
            Dict: 以问题为键的搜索结果字典
        """
        results = {}
        for question in questions:
            results[question] = self.search_by_question(question, k)
        return results


def main():
    """主函数 - 提供交互式搜索"""
    try:
        # 初始化搜索器
        searcher = Text2SQLSearcherQwen(collection_name="text2sql_qwen_collection")
        
        print("Text2SQL向量数据库搜索工具 (使用Qwen3嵌入模型)")
        print("输入 'quit' 或 'exit' 退出程序")
        print("输入 'batch' 进行批量测试")
        print("=" * 60)
        
        while True:
            # 获取用户输入
            user_input = input("\n请输入您的问题或关键词: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见!")
                break
            
            if user_input.lower() == 'batch':
                # 批量测试
                test_questions = [
                    "What is the total sales by region?",
                    "Show all employees in the company",
                    "Count the number of customers",
                    "Find the highest salary",
                    "List all products with price greater than 100"
                ]
                
                print("\n执行批量测试...")
                batch_results = searcher.batch_search(test_questions, k=2)
                
                for question, results in batch_results.items():
                    print(f"\n查询: {question}")
                    searcher.display_search_results(results)
                continue
            
            if not user_input:
                continue
            
            # 执行搜索
            results = searcher.search_by_question(user_input, k=5)
            
            # 显示结果
            searcher.display_search_results(results)
            
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")


if __name__ == "__main__":
    main()
