import requests
import json

url = "http://localhost:8100/generate_keywords"

article = """
미국 바이든 정부가 중국의 반도체 산업 육성을 저지하기 위해 새로운 수출 규제를 도입했습니다. 
이번 조치로 인해 미국 기업들은 중국 기업에 첨단 반도체 제조 장비를 판매할 때 정부의 허가를 받아야 합니다. 
이는 중국의 기술 발전을 막고 미국의 기술 우위를 유지하려는 전략의 일환으로 보입니다. 
이로 인해 글로벌 반도체 산업 전반에 큰 영향이 있을 것으로 예상됩니다.
"""

payload = {"article": article}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # 오류 발생 시 예외를 발생시킵니다.
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    print(f"Response status code: {response.status_code}")
    print(f"Response text: {response.text}")