#!/usr/bin/env python3
"""
Text2SQL Vector Database Search with Reranking
最终修复版本：完全解决padding token问题
"""

import logging
import re
import jsonlines
from typing import List, Dict, Any
from pathlib import Path

from langchain_milvus import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from tqdm import tqdm
from config import EMBEDDING_CONFIG, MILVUS_CONFIG, SEARCH_CONFIG, RERANKER_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Text2SQLSearcherWithRerank:
    def __init__(self, 
                 collection_name: str = None,
                 use_reranker: bool = True):
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
                connection_args={"uri": MILVUS_CONFIG["uri"]})
            logger.info(f"已连接到向量数据库集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"连接向量数据库失败: {e}")
            raise
        
        # 初始化重排序器 - 完全修复padding token问题
        if self.use_reranker:
            try:
                logger.info(f"初始化重排序模型: {RERANKER_CONFIG['model_name']}")
                
                # 直接使用sentence_transformers的CrossEncoder
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(RERANKER_CONFIG['model_name'],
                                                  device=RERANKER_CONFIG['device'])
                
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
    
    def rerank_documents(self, query, schema: str, documents: List[Document], top_k: int, rank_key: str) -> List[Document]:
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
                text_pairs.append((query, doc.metadata[rank_key]))
            
            # 计算重排序分数
            scores = self.cross_encoder.predict(text_pairs)
            
            # 将分数与文档配对并排序
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 返回top-k个重排序后的文档
            reranked_docs = []
            # 如果doc.metadata['schema']长度大于1000，则跳过
            for doc, score in doc_scores[:top_k]:
                # 将重排序分数添加到metadata中
                doc.metadata['rerank_score'] = float(score)
                reranked_docs.append(doc)
            
            logger.info(f"重排序完成，处理了{len(documents)}个文档，返回{len(reranked_docs)}个结果")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"重排序过程中出错: {e}")
            return documents[:top_k]
    
    def search_by_question(self, question: str, schema: str, use_rerank: bool = None, rank_key='question') -> List[Dict[str, Any]]:
        """
        根据问题搜索相似的SQL示例（支持重排序）
        
        Args:
            question: 用户问题
            k: 返回结果数量
            use_rerank: 是否使用重排序（覆盖默认设置）
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        k = SEARCH_CONFIG["default_k"]
        use_rerank = use_rerank if use_rerank is not None else self.use_reranker
        
        logger.info(f"搜索问题: {question} (重排序: {'开启' if use_rerank else '关闭'})")
        
        try:
            # 如果使用重排序，先获取更多候选结果
            search_k = SEARCH_CONFIG['max_k'] if use_rerank and hasattr(self, 'cross_encoder') else k
            
            # 进行向量搜索
            results = self.vector_store.similarity_search_with_score(question, k=search_k)
            documents = [doc for doc, score in results]
            print(len(documents))
            
            if use_rerank and hasattr(self, 'cross_encoder') and len(documents) > 1:
                # 使用重排序
                reranked_docs = self.rerank_documents(question, schema, documents, k, rank_key)
                
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
    
    def compare_search_methods(self, question: str, schema: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        比较普通搜索和重排序搜索的结果
        
        Args:
            question: 查询问题
            k: 返回结果数量
            
        Returns:
            Dict: 包含三种方法结果的字典
        """
        logger.info(f"比较搜索方法: {question}")
        
        # 普通搜索
        normal_results = self.search_by_question(question, schema, use_rerank=False)
        
        # 重排序搜索
        rerank_results = self.search_by_question(question, schema, use_rerank=True)

        # 重排序搜索，使用sql_answer作为重排序依据
        rerank_results_sql_as_key = self.search_by_question(question, schema, use_rerank=True, rank_key='sql_answer')
        
        return {
            'normal_search': normal_results,
            'reranked_search': rerank_results,
            'reranked_search_sql_as_key': rerank_results_sql_as_key
        }
    
    def display_comparison(self, question: str, schema: str):
        """
        显示搜索方法比较结果
        
        Args:
            question: 查询问题
            k: 返回结果数量
        """
        results = self.compare_search_methods(question, schema)
        
        print(f"\n🔍 查询: {question}")
        print("=" * 80)
        
        print("\n📊 普通向量搜索结果:")
        print("-" * 40)
        for i, result in enumerate(results['normal_search'], 1):
            print(f"{i}. (向量分数: {result['score']:.4f}) {result['question'][:200]}...")
        
        print("\n🎯 重排序搜索结果:")
        print("-" * 40)
        for i, result in enumerate(results['reranked_search'], 1):
            score_type = "重排序分数" if result['reranked'] else "向量分数"
            print(f"{i}. ({score_type}: {result['score']:.4f}) {result['question'][:200]}...")
        
        print("\n🎯 重排序搜索结果 (以 SQL 作为重排序依据):")
        print("-" * 40)
        for i, result in enumerate(results['reranked_search_sql_as_key'], 1):
            score_type = "重排序分数" if result['reranked'] else "向量分数"
            print(f"{i}. ({score_type}: {result['score']:.4f}) {result['question'][:200]}...")

        print("\n" + "=" * 80)


reference_prompt_template = """
<reference_list description>{reference_items}
</reference_list>
"""

reference_item_template = """
<reference_item id="{id}">
[QUESTION]
{question}
[ANSWER]
{sql_answer}
</reference_item>
"""

def general_compression(text: str) -> str:
    """
    通用文本压缩，去除多余空白和换行
    
    Args:
        text: 原始文本字符串
        
    Returns:
        压缩后的文本字符串
    """
    # 多个空格替换为单个空格
    compressed_text = re.sub(r'\s+', ' ', text).strip()
    return compressed_text

def get_reference_prompt(results: List[Dict[str, Any]]) -> str:
    """
    生成参考提示
    
    Args:
        results: 搜索结果列表
        
    Returns:
        参考提示字符串
    """
    
    reference_items = "\n".join([
        reference_item_template.format(
            id=idx,
            question=res['question'],
            sql_answer=res['sql_answer'])
        for idx, res in enumerate(results)])
    
    reference_prompt = reference_prompt_template.format(reference_items=reference_items)
    return general_compression(reference_prompt)

def save_results_to_file(input_filename: str, output_filename: str):
    """
    将搜索结果保存到文件
    
    Args:
        results: 搜索结果字典
        filename: 保存的文件名
    """
    searcher = Text2SQLSearcherWithRerank(
        collection_name=MILVUS_CONFIG["collection_name"],
        use_reranker=True
    )
    try:
        with jsonlines.open(input_filename, mode='r') as reader:
            with jsonlines.open(output_filename, mode='w') as writer:
                for item in tqdm(reader):
                    question = item.get('input', '')
                    schema = item.get('table_creating', '')
                    results = searcher.compare_search_methods(question, schema)
                    normal_reference_prompt = get_reference_prompt(results['normal_search'])
                    reranked_reference_prompt = get_reference_prompt(results['reranked_search'])
                    ranked_reference_prompt_sql_as_key = get_reference_prompt(results['reranked_search_sql_as_key'])
                    save_item = {
                        'input': question,
                        'output': item.get('output', ''),
                        'table_creating': general_compression(schema),
                        'normal_reference_prompt': normal_reference_prompt,
                        'reranked_reference_prompt': reranked_reference_prompt,
                        'ranked_reference_prompt_sql_as_key': ranked_reference_prompt_sql_as_key,
                    }
                    writer.write(save_item)

        logger.info(f"搜索结果已保存到文件: {output_filename}")
    except Exception as e:
        logger.error(f"保存结果到文件时出错: {e}")


def demo_reranking():
    """演示重排序功能"""
    
    print("🚀 Text2SQL RAG 重排序演示 - 最终修复版本")
    print("=" * 60)
    
    try:
        # 初始化带重排序的搜索器
        searcher = Text2SQLSearcherWithRerank(
            collection_name=MILVUS_CONFIG["collection_name"],
            use_reranker=True
        )
        
        # 测试查询
        test_queries = [
            "Show name, country, age for all singers ordered by age from the oldest to the youngest",
            "How many paragraphs for the document with name 'Summer Show'?",
            "What are the towns from which at least two teachers come from?"
        ]

        test_schemas = [
            "<sql>\nCREATE TABLE stadium (\n    stadium_id INT PRIMARY KEY,\n    location VARCHAR(255),\n    name VARCHAR(255),\n    capacity INT,\n    highest INT,\n    lowest INT,\n    average INT\n);\n\nCREATE TABLE singer (\n    singer_id INT PRIMARY KEY,\n    name VARCHAR(255),\n    country VARCHAR(255),\n    song_name VARCHAR(255),\n    song_release_year INT,\n    age INT,\n    is_male BOOLEAN\n);\n\nCREATE TABLE concert (\n    concert_id INT PRIMARY KEY,\n    concert_name VARCHAR(255),\n    theme VARCHAR(255),\n    stadium_id INT,\n    year INT,\n    FOREIGN KEY (stadium_id) REFERENCES stadium(stadium_id)\n);\n\nCREATE TABLE singer_in_concert (\n    concert_id INT,\n    singer_id INT,\n    PRIMARY KEY (concert_id, singer_id),\n    FOREIGN KEY (concert_id) REFERENCES concert(concert_id),\n    FOREIGN KEY (singer_id) REFERENCES singer(singer_id)\n);\n</sql>",
            "<sql>\nCREATE TABLE ref_template_types (\n    template_type_code      VARCHAR(10)  PRIMARY KEY,\n    template_type_description VARCHAR(255) NOT NULL\n);\n\nCREATE TABLE templates (\n    template_id          INT          PRIMARY KEY,\n    version_number       INT          NOT NULL,\n    template_type_code   VARCHAR(10)  NOT NULL,\n    date_effective_from  DATE,\n    date_effective_to    DATE,\n    template_details     TEXT,\n    CONSTRAINT fk_templates_type\n        FOREIGN KEY (template_type_code)\n        REFERENCES ref_template_types(template_type_code)\n);\n\nCREATE TABLE documents (\n    document_id        INT          PRIMARY KEY,\n    template_id        INT          NOT NULL,\n    document_name      VARCHAR(255),\n    document_description TEXT,\n    other_details      TEXT,\n    CONSTRAINT fk_documents_template\n        FOREIGN KEY (template_id)\n        REFERENCES templates(template_id)\n);\n\nCREATE TABLE paragraphs (\n    paragraph_id   INT          PRIMARY KEY,\n    document_id    INT          NOT NULL,\n    paragraph_text TEXT,\n    other_details  TEXT,\n    CONSTRAINT fk_paragraphs_document\n        FOREIGN KEY (document_id)\n        REFERENCES documents(document_id)\n);\n</sql>",
            "<sql>\nCREATE TABLE course (\n    course_id      VARCHAR(50) PRIMARY KEY,\n    starting_date  DATE,\n    course         VARCHAR(100)\n);\n\nCREATE TABLE teacher (\n    teacher_id VARCHAR(50) PRIMARY KEY,\n    name       VARCHAR(100),\n    age        INT,\n    hometown   VARCHAR(100)\n);\n\nCREATE TABLE course_arrange (\n    course_id  VARCHAR(50) PRIMARY KEY,\n    teacher_id VARCHAR(50),\n    grade      VARCHAR(10),\n    FOREIGN KEY (course_id)  REFERENCES course(course_id),\n    FOREIGN KEY (teacher_id) REFERENCES teacher(teacher_id)\n);\n</sql>"
        ]
        
        for query, schema in zip(test_queries, test_schemas):
            searcher.display_comparison(query, schema)
            print()
        
        print("🎉 重排序演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")


if __name__ == "__main__":
    # demo_reranking()
    save_results_to_file(
        input_filename="../../data/infer_data_and_gt/dev_infer_data.jsonl",
        output_filename="../../data/infer_data_and_gt/dev_infer_data_with_reference.json")
