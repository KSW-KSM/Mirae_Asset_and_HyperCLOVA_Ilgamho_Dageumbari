from langchain.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain.memory import ConversationKGMemory
from langchain.chains import ConversationChain

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from concurrent.futures import ThreadPoolExecutor
from llm.clova import Clova

from typing import List
from dotenv import load_dotenv
import os
import json


class ArticleToKeyword: 
    def __init__(self): 
        load_dotenv() 
        data = None 
        with open('/Users/seong-ugang/Desktop/학교/공모전/미래에셋_2500/dev/data.json', 'r') as f: 
            data = json.load(f) 
        api_key = os.getenv('CLOVA_API_KEY') 
        print(api_key) 
        api_key_primary_val = os.getenv('API_KEY_PRI_VAL') 
        print(api_key_primary_val) 
        request_id = os.getenv('REQUEST_ID') 
        print(request_id) 
        self.llm = Clova( 
            api_key=api_key, 
            api_key_primary_val=api_key_primary_val, 
            request_id=request_id, 
            model_name="HyperX", 
            temperature=0.5, 
            system_message=""" 
                너는 기사 내용을 이해하고, 기사에 등장하지않는 주요 키워드를 제공하는 역할을 해. 
            """ 
        ).setup()
        text_splitter = CharacterTextSplitter() 
        #self.llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o")
        self.llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'))
        prompt = PromptTemplate( 
        # 주제에 대한 다섯 가지를 나열하라는 템플릿
        #        관련 이슈 및 사건
        template="""
        {keyword}를 분석하여 다음 지시사항에 따라 키워드를 추출해주세요:

        기사에 등장하지 않는 주요 키워드를 한국어로 추출해주세요.
        추상적인 개념보다는 구체적이고 세부적인 용어를 선택해주세요.
        다음 카테고리별로 최소 3개 이상의 키워드를 제시해주세요:

        관련 기술 및 제품
        관련 기업 및 기관
        관련 법률 및 정책
        관련 산업 및 시장 동향
        관련 국가 및 지역


        기사의 주제와 관련된 최신 트렌드나 이슈를 반영하는 키워드도 포함해주세요.
        전문 용어나 약어가 있다면 그것도 포함해주세요.
        총 20개 이상의 다양한 키워드를 제시해주세요.

        결과는 다음 JSON 형식으로 출력해주세요:
        
        "keywords": [
        "category": "관련 기술 및 제품", "items": ["키워드1", "키워드2", "키워드3"],
        "category": "관련 기업 및 기관", "items": ["키워드4", "키워드5", "키워드6"],
        ...
        ]
        
        주의: 기사 내용과 직접적으로 언급되지 않은 키워드만을 선별해주세요.
        """,
        input_variables=["keyword"],  
        )

        chain = prompt | self.llm
        print(data['data'][2]['text']) 
        #re = chain.invoke({"keyword": data['data'][2]['text']}) 
        re = chain.invoke({"keyword": "급발진논란에 페달 블랙박스 의무화 법안 나왔다"}) 
        print(re.content) 

ArticleToKeyword()