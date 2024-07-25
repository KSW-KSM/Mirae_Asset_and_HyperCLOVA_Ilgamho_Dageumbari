# -*- coding: utf-8 -*-
import base64
import json
import http.client
from langchain.embeddings.base import Embeddings

class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/api-tools/embedding/v2/a2601a175a8d4951a59928411ec22d61', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['embedding']
        else:
            return 'Error'

class PrecomputedEmbeddings(Embeddings):
    def __init__(self, embeddings, hyper_clova_instance):
        self._embeddings = embeddings
        self._hyper_clova = hyper_clova_instance
        
    def embed_documents(self, texts):
        return self._embeddings
    
    def embed_query(self, text):
        request_data = {"text": text}
        return self._hyper_clova.execute(request_data)