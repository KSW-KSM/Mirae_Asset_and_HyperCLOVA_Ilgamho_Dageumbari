import os
import json
from langchain.document_loaders import PyPDFLoader, JSONLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import ClovaEmbeddings
from langchain.vectorstores import PGVector
from langchain.schema import Document
from llm.hyper_embed import CompletionExecutor, PrecomputedEmbeddings
from dotenv import load_dotenv
from typing import List
import pdfplumber
from typing import Dict, List, Optional, cast
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

import time
from typing import List, cast
from pydantic import SecretStr
import requests

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
            
def clean_text(text):
    # 불필요한 공백 및 특수문자 제거
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # 공백만 있는 줄 제거
        if line.strip():
            cleaned_lines.append(line.strip())
    return '\n'.join(cleaned_lines)

def extract_text_with_pdfplumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            # 페이지의 모든 텍스트 추출
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
        
        # 텍스트 정리 및 후처리
        cleaned_text = clean_text(text)
        return cleaned_text

pdf_path = "./dev/data/20240724_[배터리 메탈_소재] 고려아연 (010130_매수_신규).pdf"
extracted_text = extract_text_with_pdfplumber(pdf_path)



def extract_data(record: dict):
    return record.get("text", "")

json_loader = JSONLoader(
    file_path="./dev/data2.json",
    jq_schema='.data[]',
    content_key="text",
    text_content=False,
    
)
json_docs = json_loader.load()

load_dotenv()
# 모든 문서 합치기
all_docs =  [Document(extracted_text)] + json_docs

# 텍스트 스플리터 정의
text_splitter = RecursiveCharacterTextSplitter()

# 문서 청킹
chunks = text_splitter.split_documents(all_docs)

embeddings = CustomClovaEmbeddings(
    clova_emb_api_key=os.getenv('CLOVA_API_KEY'),
    clova_emb_apigw_api_key=os.getenv('CLOVA_AMD_API_KEY'),
    app_id=os.getenv('REQUEST_ID'),
)

connection_string = os.getenv('POSTGRES_USER')

# 기존 테이블 삭제 (필요한 경우)

from sqlalchemy import create_engine, text
engine = create_engine(connection_string)
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding"))
    conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection"))

# PGVector 초기화 및 문서 저장
db = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string=connection_string,
    collection_name="your_collection_name"
)

# 임베딩 함수 초기화
#embedding_func = PrecomputedEmbeddings(embeddings, hyper_clova)

query = "요즘 집값"  
docs = db.similarity_search(query, k=5)

print(f"'{query}'에 대한 검색 결과:")
for doc in docs:
    #print(doc)
    print(doc.page_content[:100]) 