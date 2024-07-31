import os
import json
from langchain.vectorstores import PGVector
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Dict, List, Optional, cast
from langchain_community.embeddings import ClovaEmbeddings
import time
import requests
from pydantic import SecretStr
from langchain_core.embeddings import Embeddings
from llm.clova import Clova
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class CustomClovaEmbeddings(ClovaEmbeddings):
    """
    Custom Clova's embedding service with enhanced error handling and debugging.
    """

    def _embed_text(self, text: str) -> List[float]:
        """
        Internal method to call the embedding API and handle the response with enhanced error handling.
        """
        payload = {"text": text}

        # HTTP headers for authorization
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": cast(
                SecretStr, self.clova_emb_api_key
            ).get_secret_value(),
            "X-NCP-APIGW-API-KEY": cast(
                SecretStr, self.clova_emb_apigw_api_key
            ).get_secret_value(),
            "Content-Type": "application/json",
        }

        # Debugging output
        print(f"Sending request to {self.endpoint_url}/{self.model}/{cast(SecretStr, self.app_id).get_secret_value()} with payload: {payload}")

        app_id = cast(SecretStr, self.app_id).get_secret_value()
        url = f"https://clovastudio.apigw.ntruss.com/testapp/v1/api-tools/embedding/v2/58554d44ee52497c8172968297b30fcd"

        while True:
            response = requests.post(url, headers=headers, json=payload)

            # Debugging output
            print(f"Received response with status code {response.status_code}: {response.text}")

            # check for errors
            if response.status_code == 200:
                response_data = response.json()
                print(f"Response data: {response_data}")
                if "result" in response_data and "embedding" in response_data["result"]:
                    return response_data["result"]["embedding"]
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                print(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
            else:
                raise ValueError(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

class CausalKeywordGenerator:
    def __init__(self):
        load_dotenv()
        
        # Clova 임베딩 초기화
        self.embeddings = CustomClovaEmbeddings(
            clova_emb_api_key=os.getenv('CLOVA_API_KEY'),
            clova_emb_apigw_api_key=os.getenv('CLOVA_AMD_API_KEY'),
            app_id=os.getenv('REQUEST_ID')
        )
        
        # 벡터 데이터베이스 초기화
        connection_string = os.getenv('POSTGRES_USER')
        self.vectorstore = PGVector(
            connection_string=connection_string,
            embedding_function=self.embeddings,
            collection_name="your_collection_name"
        )
        
        # Clova LLM 초기화
        self.llm = Clova(
            api_key=os.getenv('CLOVA_API_KEY'),
            api_key_primary_val=os.getenv('API_KEY_PRI_VAL'),
            request_id=os.getenv('REQUEST_ID'),
            model_name="HyperX",
            temperature=0.0,
            system_message="""
            You are an AI assistant specialized in generating causal keywords based on given topics and contexts.
            """,
            max_tokens=2000,
        ).setup()
        
        # Self Query Retriever 설정
        metadata_field_info = [
            AttributeInfo(
                name="category",
                description="The category of the document",
                type="string",
            ),
            AttributeInfo(
                name="date",
                description="The date of the document",
                type="string",
            ),
        ]
        document_content_description = "News article about various topics"
        self.retriever = SelfQueryRetriever.from_llm(
            self.llm,
            self.vectorstore,
            document_content_description,
            metadata_field_info,
            verbose=True
        )
        
        # 프롬프트 템플릿 설정
        self.prompt = PromptTemplate(
            template="""
        기사: "{topic}"
        를 분석하여 다음 지시사항에 따라 키워드를 추출해주세요:

        기사에 등장하지 않는 주요 키워드를 한국어로 추출해주세요.
        추상적인 개념보다는 구체적이고 세부적인 용어를 선택해주세요.
        다음 카테고리별로 최소 3개 이상의 키워드를 제시해주세요:

        관련기사정보: {context}

        관련 기술 및 제품
        관련 기업 및 기관
        관련 법률 및 정책
        관련 산업 및 시장 동향
        관련 국가 및 지역


        기사의 주제와 관련된 최신 트렌드나 이슈를 반영하는 키워드도 포함해주세요.
        전문 용어나 약어가 있다면 그것도 포함해주세요.
        총 20개 이상의 다양한 키워드를 제시해주세요.
        단, 기사 내용과 직접적으로 언급되지 않은 키워드만을 선별해주세요. 만약 애플 제품에 대한 기사라면, 애플 제품명은 제외해주세요. 다른 관련 기술이나 기업, 산업등에 대한 키워드를 선별해주세요.
        주식과 관련된 키워드로 예를 들면, 삼성전자, 삼성전자 주가, 삼성전자 주식 등은 제외하고, 삼성전자의 경쟁사, 삼성전자의 제품, 삼성전자의 기술 등을 선별해주세요.
        최대한 디테일한 키워드를 선별해주세요. 예를 들어, 삼성전자의 스마트폰이라는 키워드가 나왔다면, 스마트폰이라는 키워드는 제외하고, 삼성전자의 스마트폰과 관련된 기술이나 제품명, 서비스명 등을 선별해주세요.
        
        결과는 다음 JSON 형식으로 출력해주세요:
        
        {{
            "keywords": [
                {{"category": "관련 기술 및 제품", "items": ["키워드1", "키워드2", "키워드3"]}},
                {{"category": "관련 기업 및 기관", "items": ["키워드4", "키워드5", "키워드6"]}},
                ...
            ]
        }}
        
        주의1: 기사 내용과 직접적으로 언급되지 않은 키워드만을 선별해주세요. 
        주의2: 이 키워드는 주제와 관련된 인과관계를 중심으로 생성해야 합니다.
        주의3: DB에 저장된 문서를 적극 활용해주세요.
        주의4: json 형식을 꼭 지켜주세요. json 형식을 벗어나는 출력은 절대 허용되지 않습니다.
        주의5: 중괄호 범위 밖에 있는 내용은 삭제하고 출력해주세요. 
        주의6: 추가적인 도움이 필요하시면 언제든지 말씀해주세요. 라는 문장은 삭제하고 출력해주세요.
        """,
            input_variables=["topic", "context"]
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

        # 기사 요약 프롬프트 추가
        self.summary_prompt = PromptTemplate(
            template="다음 기사를 100자 이내로 요약해주세요:\n\n{article}\n\n요약:",
            input_variables=["article"]
        )
        self.summary_chain = LLMChain(llm=self.llm, prompt=self.summary_prompt)
        print("summary_chain\n\n\n")
        # 긍정/부정 분석 프롬프트
        self.sentiment_prompt = PromptTemplate(
            template="""
            다음 키워드 리스트에 대해 각 키워드의 긍정/부정 비율을 분석하고, 
            60~70% 비중(긍정 또는 부정)을 가진 키워드만 선별해주세요.
            
            키워드 리스트:
            {keywords}
            
            결과는 다음 JSON 형식으로 출력해주세요:
            {{
                "selected_keywords": [
                    {{
                        "keyword": "키워드1",
                        "sentiment": "긍정" 또는 "부정",
                        "ratio": 비율(60~70 사이의 숫자)
                    }},
                    ...
                ]
            }}
            
            주의: JSON 형식을 정확히 지켜주세요. 추가 설명이나 주석은 포함하지 마세요.
            """,
            input_variables=["keywords"]
        )
        self.sentiment_chain = LLMChain(llm=self.llm, prompt=self.sentiment_prompt)
        print("sentiment_chain\n\n")

    def generate_causal_keywords(self, article):
        # 기사 요약
        summary = self.summary_chain.run(article=article)
        
        # Self Query Retriever를 사용하여 관련 문서 검색
        query = f"""Find documents related to {summary} focusing on cause and effect relationships"""
        docs = self.retriever.get_relevant_documents(query)
        
        # 검색된 문서의 내용을 컨텍스트로 결합
        context = "\n".join([doc.page_content for doc in docs])
        print(f"Context: {context}")
        
        # LLM을 사용하여 인과관계 중심의 키워드 생성
        response = self.chain.run(topic=summary, context=context)
        
        try:
            keywords = json.loads(response)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 LLM을 사용하여 수정
            fix_prompt = PromptTemplate(
                template="다음 텍스트를 유효한 JSON 형식으로 수정해주세요:\n\n{text}\n\n수정된 JSON:",
                input_variables=["text"]
            )
            fix_chain = LLMChain(llm=self.llm, prompt=fix_prompt)
            fixed_response = fix_chain.run(text=response)
            keywords = json.loads(fixed_response)

        # 긍정/부정 분석 및 필터링
        sentiment_response = self.sentiment_chain.run(keywords=json.dumps(keywords))
        
        try:
            filtered_keywords = json.loads(sentiment_response)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 LLM을 사용하여 수정
            fix_prompt = PromptTemplate(
                template="다음 텍스트를 유효한 JSON 형식으로 수정해주세요:\n\n{text}\n\n수정된 JSON:",
                input_variables=["text"]
            )
            fix_chain = LLMChain(llm=self.llm, prompt=fix_prompt)
            fixed_response = fix_chain.run(text=sentiment_response)
            filtered_keywords = json.loads(fixed_response)

        return filtered_keywords

# FastAPI 서버 설정
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleRequest(BaseModel):
    article: str

import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/generate_keywords")
async def generate_keywords(request: ArticleRequest):
    generator = CausalKeywordGenerator()
    try:
        logger.info(f"Received article: {request.article[:100]}...")  # 첫 100자만 로그에 기록
        keywords = generator.generate_causal_keywords(request.article)
        logger.info(f"Generated keywords: {keywords}")
        return keywords
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON format in the response")
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)