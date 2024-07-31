## 가이드

이 프로젝트를 실행하기 위한 주요 단계는 다음과 같습니다. :
0. `pip install -r requirements.txt` 명령어로 코드 실행에 필요한 의존성 설치

1. dev/.env파일을 생성하고 
```
CLOVA_API_KEY = 
API_KEY_PRI_VAL =
REQUEST_ID =
POSTGRES_USER = 
CLOVA_AMD_API_KEY = 
```
위와 같이 키 등록

1. `dev/vector_DB_init.py` 파일을 열고 실행하여 db을 초기화합니다.

2. `dev/server.py` 파일을 실행하여 fast api 서버를 구동

3. `cd mirae-asset-news-analysis` 명령어로 프론트 폴더로 이동후 `npm install` 명령실행

4. 마지막으로 `npm start`

5. 임베딩, LLM은 모두 CLOVA model을 사용하여 구현되었습니다.
