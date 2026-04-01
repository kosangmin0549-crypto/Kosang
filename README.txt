KRX Live Bundle

구성
- backend/app.py : FastAPI 백엔드
- frontend/index.html : 실데이터 스크리너 프런트엔드
- requirements.txt : 설치 패키지

실행
1) Python 설치 후 터미널에서:
   pip install -r requirements.txt

2) 서버 실행:
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

3) frontend/index.html 을 브라우저에서 열기

기본값
- API Base URL: http://127.0.0.1:8000
- 최근 365일 일봉
- 종목 80개 우선 분석

주의
- 분석 종목 수를 크게 올리면 시간이 오래 걸릴 수 있음
- 이 번들에는 실시간 틱/분봉이 아니라 최근 1년 일봉 기반 스크리닝이 들어 있음
- 손실 비발생은 보장하지 않음
