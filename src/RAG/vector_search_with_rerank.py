#!/usr/bin/env python3
"""
Text2SQL Vector Database Search with Reranking
最终修复版本：完全解决padding token问题
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


class Text2SQLSearcherWithRerank:
    def __init__(self, 
                 collection_name: str = None,
                 use_reranker: bool = True,
                 reranker_model: str = "/home/qianwenhao/LLM/Qwen3-Reranker-0.6B"):
        """
        初始化Text2SQL搜索器（带重排序功能）
        
        Args:
            collection_name: Milvus集合名称
            use_reranker: 是否使用重排序
            reranker_model: 重排序模型名称
        """
        self.collection_name = collection_name or MILVUS_CONFIG["collection_name"]
        self.use_reranker = use_reranker
        
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
        
        # 初始化重排序器 - 完全修复padding token问题
        if self.use_reranker:
            try:
                logger.info(f"初始化重排序模型: {reranker_model}")
                
                # 直接使用sentence_transformers的CrossEncoder
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(reranker_model)
                
                # 完全修复padding token问题
                # 1. 设置tokenizer的pad_token
                self.cross_encoder.tokenizer.pad_token = self.cross_encoder.tokenizer.eos_token
                self.cross_encoder.tokenizer.pad_token_id = self.cross_encoder.tokenizer.eos_token_id
                
                # 2. 设置模型配置的pad_token_id
                self.cross_encoder.model.config.pad_token_id = self.cross_encoder.tokenizer.eos_token_id
                
                logger.info(f"设置pad_token_id为: {self.cross_encoder.tokenizer.eos_token_id}")
                logger.info("重排序器初始化成功")
            except Exception as e:
                logger.warning(f"重排序器初始化失败，将使用普通搜索: {e}")
                self.use_reranker = False
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """
        使用CrossEncoder对文档进行重排序
        
        Args:
            query: 查询字符串
            documents: 候选文档列表
            top_k: 返回top-k个文档
            
        Returns:
            重排序后的文档列表
        """
        if not hasattr(self, 'cross_encoder') or not documents:
            return documents[:top_k]
        
        try:
            # 准备文本对 (query, document_content)
            text_pairs = []
            for doc in documents:
                # 使用问题作为文档内容进行重排序
                doc_text = doc.metadata.get('question', '') + ' ' + doc.page_content
                text_pairs.append((query, doc_text))
            
            # 计算重排序分数
            scores = self.cross_encoder.predict(text_pairs)
            
            # 将分数与文档配对并排序
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 返回top-k个重排序后的文档
            reranked_docs = []
            for doc, score in doc_scores[:top_k]:
                # 将重排序分数添加到metadata中
                doc.metadata['rerank_score'] = float(score)
                reranked_docs.append(doc)
            
            logger.info(f"重排序完成，处理了{len(documents)}个文档，返回{len(reranked_docs)}个结果")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"重排序过程中出错: {e}")
            return documents[:top_k]
    
    def search_by_question(self, question: str, k: int = None, use_rerank: bool = None) -> List[Dict[str, Any]]:
        """
        根据问题搜索相似的SQL示例（支持重排序）
        
        Args:
            question: 用户问题
            k: 返回结果数量
            use_rerank: 是否使用重排序（覆盖默认设置）
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        k = k or SEARCH_CONFIG["default_k"]
        use_rerank = use_rerank if use_rerank is not None else self.use_reranker
        
        logger.info(f"搜索问题: {question} (重排序: {'开启' if use_rerank else '关闭'})")
        
        try:
            # 如果使用重排序，先获取更多候选结果
            search_k = k * 3 if use_rerank and hasattr(self, 'cross_encoder') else k
            
            # 进行向量搜索
            results = self.vector_store.similarity_search_with_score(question, k=search_k)
            documents = [doc for doc, score in results]
            
            if use_rerank and hasattr(self, 'cross_encoder') and len(documents) > 1:
                # 使用重排序
                reranked_docs = self.rerank_documents(question, documents, k)
                
                search_results = []
                for doc in reranked_docs:
                    result = {
                        'score': doc.metadata.get('rerank_score', 0.0),
                        'question': doc.metadata.get('question', ''),
                        'schema': doc.metadata.get('schema', ''),
                        'sql_answer': doc.metadata.get('sql_answer', ''),
                        'id': doc.metadata.get('id', ''),
                        'reranked': True
                    }
                    search_results.append(result)
            else:
                # 使用普通向量搜索结果
                search_results = []
                for doc, score in results[:k]:
                    result = {
                        'score': score,
                        'question': doc.metadata.get('question', ''),
                        'schema': doc.metadata.get('schema', ''),
                        'sql_answer': doc.metadata.get('sql_answer', ''),
                        'id': doc.metadata.get('id', ''),
                        'reranked': False
                    }
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索时出错: {e}", exc_info=True)
            return []
    
    def compare_search_methods(self, question: str, k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        比较普通搜索和重排序搜索的结果
        
        Args:
            question: 查询问题
            k: 返回结果数量
            
        Returns:
            Dict: 包含两种方法结果的字典
        """
        logger.info(f"比较搜索方法: {question}")
        
        # 普通搜索
        normal_results = self.search_by_question(question, k, use_rerank=False)
        
        # 重排序搜索
        rerank_results = self.search_by_question(question, k, use_rerank=True)
        
        return {
            'normal_search': normal_results,
            'reranked_search': rerank_results
        }
    
    def display_comparison(self, question: str, k: int = 3):
        """
        显示搜索方法比较结果
        
        Args:
            question: 查询问题
            k: 返回结果数量
        """
        results = self.compare_search_methods(question, k)
        
        print(f"\n🔍 查询: {question}")
        print("=" * 80)
        
        print("\n📊 普通向量搜索结果:")
        print("-" * 40)
        for i, result in enumerate(results['normal_search'], 1):
            print(f"{i}. (向量分数: {result['score']:.4f}) {result['question'][:60]}...")
        
        print("\n🎯 重排序搜索结果:")
        print("-" * 40)
        for i, result in enumerate(results['reranked_search'], 1):
            score_type = "重排序分数" if result['reranked'] else "向量分数"
            print(f"{i}. ({score_type}: {result['score']:.4f}) {result['question'][:60]}...")
        
        print("\n" + "=" * 80)


def demo_reranking():
    """演示重排序功能"""
    
    print("🚀 Text2SQL RAG 重排序演示 - 最终修复版本")
    print("=" * 60)
    
    try:
        # 初始化带重排序的搜索器
        searcher = Text2SQLSearcherWithRerank(
            collection_name='text2sql_test_collection',
            use_reranker=True
        )
        
        # 测试查询
        test_queries = [
            "Count the total number of customers",
            "Find the highest salary",
            "Show sales by region"
        ]
        
        for query in test_queries:
            searcher.display_comparison(query, k=3)
            print()
        
        print("🎉 重排序演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")


if __name__ == "__main__":
    demo_reranking()
