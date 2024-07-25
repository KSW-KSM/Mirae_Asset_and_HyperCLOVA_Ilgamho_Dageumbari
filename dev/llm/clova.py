# -*- coding: utf-8 -*-

import requests


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        with requests.post(self._host + '/testapp/v1/chat-completions/HCX-DASH-001',
                           headers=headers, json=completion_request, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    print(line.decode("utf-8"))


if __name__ == '__main__':
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key='NTA0MjU2MWZlZTcxNDJiY4ujR7nHhC0y9c3wY9dKJkxTX8LeN+C/uXtZCG/qZFc1',
        api_key_primary_val='BHkaWCrCwI5zz3eWmUOVQVT3OqJpCTQV5KwfctcD',
        request_id='6b83f15e-efae-403e-8d28-4c11249e248b'
    )

    preset_text = [{"role":"system","content":""},{"role":"user","content":""}]

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 256,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }

    print(preset_text)
    completion_executor.execute(request_data)


import requests

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
#from langchain.schema.retriever import BaseRetriever, Document
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


import os
import json
import dotenv
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Mapping,
)
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_core.language_models.llms import LLM, BaseLLM #https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.language_models
import requests, time, json
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
#0.1. 클로바 X 체인 생성
#https://python.langchain.com/docs/modules/model_io/llms/custom_llm
class HyperClovaX(LLM):
    host: str = None
    api_key: str = None
    api_key_primary_val: str = None
    request_id: str = None
    system_message: str = None
    top_p: float = None
    top_k: int = None
    temperature: float = None
    max_tokens: int = None

    def execute(self, completion_request) -> str:
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key, 
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val, 
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id, 
            'Content-Type': 'application/json; charset=utf-8', 
            #'Accept': 'text/event-stream'
        }

        res = None # 이전 모델 HCX-002
        with requests.post(self.host + '/testapp/v1/chat-completions/HCX-DASH-001', 
                           headers=headers, json=completion_request, stream=True) as r:
            #for line in r.iter_lines():
            #   if line:
            print("STATUS CODE:", r.status_code)
            data = r.content.decode("utf-8")
            print("DATA:", data)
            data = json.loads(data)
            print("--INPUT:", completion_request)
            print("--RESPONSE:", data)
            assert type(data) is dict
            assert type(completion_request['messages']) is list
            completion_request['messages'].append(data['result']['message'])

            while data['result']['stopReason'] == 'length':
                r = requests.post(self.host + '/testapp/v1/chat-completions/HCX-DASH-001', headers=headers, json=completion_request, stream=True)
                data_recv = r.content.decode("utf-8")
                print("--RESPONSE:", data)
                data_recv = json.loads(data_recv)
                data['result']['message']['content'] += data_recv['result']['message']['content']
                data['result']['outputLength'] += data_recv['result']['outputLength']
                data['result']['stopReason'] = data_recv['result']['stopReason']

                #결과 더함
                completion_request['messages'].append(data_recv['result']['message'])
                print("-------Add More: STOP REASON is 'LENGTH'---------")
                print(data)
            
            try:
                assert type(data) is not None
                assert data['result']['message']['content'] is not None
                print("FINAL RESULT:", data)
                res = data['result']['message']['content']
                return res
            except:
                print("ERROR", data)
                return "ERROR"

    @property
    def _llm_type(self):
        return "ClovaHyperX"

    def _call(
      self,
      prompt: str,
      stop: Optional[List[str]]= None,
      run_manager: Optional[CallbackManagerForLLMRun] = None,
      **kwargs: Any,
    ) -> str:
        # Call HyperX
        
        preset_text = [
            {"role":"system","content":self.system_message},
            {"role":"user","content":f"{prompt}"}
        ]
        
        request_data  = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 256,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }
        result = self.execute(request_data)
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return result
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "host": self.host,
            "api_key": self.api_key,
            "api_key_primary_val": self.api_key_primary_val,
            "request_id": self.request_id,
            "system_message": self.system_message,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    


class Clova:
    def __init__(self, api_key:str, 
                api_key_primary_val:str, 
                request_id:str, 
                model_name="HyperX",
                temperature:float = 0, 
                system_message:str = """
                너는 기사 내용을 이해하고, 기사에 등장하지않는 주요 키워드를 제공하는 역할을 해.
            """, 
                top_p:float = None,
                top_k:int = None,
                Frequency_Penalty: float = None,
                Presence_Penalty: float = None,
                max_tokens: int = None,
                ):
        self.api_key = api_key
        self.model_name = model_name
        self.api_key_primary_val = api_key_primary_val
        self.request_id = request_id
        self.temperature = temperature
        self.system_message = system_message
        self.top_p = top_p
        self.top_k = top_k
        self.Frequency_Penalty = Frequency_Penalty
        self.Presence_Penalty = Presence_Penalty

        if max_tokens == -1 :
            max_tokens = None
        self.max_tokens = max_tokens
    
    def setup(self):
        print(f"model_name: {self.model_name}")
        """
        chat = HyperClovaX(
            host='https://clovastudio.stream.ntruss.com',
            api_key=self.api_key,
            api_key_primary_val=self.api_key_primary_val,
            request_id=self.request_id
        )
        """
        if(self.temperature > 1.0):
            print("\t- Warning: High temperature may result in incoherent responses. (=setting temperature to 1.0)")
            self.temperature = 1.0
        chat = HyperClovaX(
            host='https://clovastudio.stream.ntruss.com',
            api_key=self.api_key,
            api_key_primary_val=self.api_key_primary_val,
            request_id=self.request_id,
            temperature=self.temperature,
            system_message=self.system_message,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens
        )
        return chat
