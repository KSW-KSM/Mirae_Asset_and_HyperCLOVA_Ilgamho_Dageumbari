import os
import json
from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import PGVector
from langchain.schema import Document
from llm.hyper_embed import CompletionExecutor, PrecomputedEmbeddings

# PDF 로더
#pdf_loader = PyPDFLoader("/Users/seong-ugang/Desktop/학교/공모전/미래에셋_2500/dev/data/20240724_[배터리 메탈_소재] 고려아연 (010130_매수_신규).pdf")
#pdf_docs = pdf_loader.load()

def extract_data(record: dict):
    return record.get("text", "")

json_loader = JSONLoader(
    file_path="./dev/data.json",
    jq_schema='.data[]',
    content_key="text",
    text_content=False,
    
)
json_docs = json_loader.load()

# 모든 문서 합치기
all_docs =  json_docs #+ pdf_docs

# 텍스트 스플리터 정의
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 문서 청킹
chunks = text_splitter.split_documents(all_docs)

# 임베딩 설정
#embeddings = OpenAIEmbeddings()
hyper_clova = CompletionExecutor(
    host="hyper-embed.com",
    api_key="YOUR_API_KEY",
    api_key_primary="YOUR_API_KEY",
    request_id="YOUR_REQUEST"
)

# 임베딩 실행
embeddings_pre = [hyper_clova.execute({"text": doc.page_content}) for doc in all_docs]

embeddings = PrecomputedEmbeddings(embeddings_pre, hyper_clova)

# PGVector 설정 및 문서 저장
connection_string = "postgresql://myuser:mypassword@localhost:5432/mydatabase"
db = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string=connection_string,
    collection_name="your_collection_name"
)



# 임베딩 함수 초기화
embedding_func = PrecomputedEmbeddings(embeddings, hyper_clova)

print("docs saved to db")
print(all_docs[0].page_content)
query = "그지같은 떡락기사"  
docs = db.similarity_search(query)

print(f"'{query}'에 대한 검색 결과:")
for doc in docs:
    print(doc.page_content[:100]) 



