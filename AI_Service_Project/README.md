# ⚖️ 법률 사건 Q&A 챗봇 (Self-RAG)

> 2026 인공지능서비스개발I 팀 프로젝트  
> **주제**: 사건 발생 시 즉시 관련 법률·판례·대처 절차를 확인하는 챗봇

---

## 📌 서비스 소개

### 대상 사용자
**일반 시민** — 교통사고, 임금체불, 사기, 폭행 등 갑작스러운 사건 발생 시 즉시 법률 정보가 필요한 사람

### RAG가 필요한 이유
| 구분 | 순수 LLM | **우리 챗봇 (Self-RAG)** |
|------|----------|--------------------------|
| 최신 법령 반영 | ❌ 훈련 데이터 기준 | ✅ 업로드된 법령 PDF 기준 |
| 특정 법 조항 인용 | ⚠️ 부정확 가능 | ✅ 실제 문서 근거 인용 |
| 할루시네이션 | ⚠️ 높은 위험 | ✅ 지원도 평가로 제어 |
| 신뢰도 | 낮음 | 높음 (문서 근거 명시) |

---

## 🏗️ 기술 스택
- **LLM**: GPT-4o / GPT-4o-mini
- **임베딩**: text-embedding-3-large
- **벡터 DB**: FAISS (로컬 저장/로드)
- **프레임워크**: LangChain
- **RAG 전략**: Self-RAG (6단계 파이프라인)
- **Memory**: ConversationBufferWindowMemory (최근 5턴)
- **UI**: Streamlit (다크 테마)
- **성능 평가**: RAGAS (Ground Truth 12개)

---

## 🔁 Self-RAG 파이프라인
```
사건 설명 / 질문
       ↓
[1] 법률 문서 검색 필요 여부 판단
   ├─ No  → 일반 LLM 답변
   └─ Yes ↓
[2] FAISS 법률 문서 검색 (Top-K)
       ↓
[3] 법률 관련성 필터링
       ↓
[4] 관련 문서별 법률 답변 생성
       ↓
[5] 지원도 평가 (Fully/Partially/No)
       ↓
[6] 유용성 점수 (1~5)
       ↓
최적 답변 선택 → 출력
       ↓
[Memory] 대화 이력 저장
```

---

## 🚀 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 환경변수 설정 (.env)
OPENAI_API_KEY=sk-...

# 3. 앱 실행
streamlit run app.py

# 4. RAGAS 성능 평가
python ragas_eval.py 형법.pdf
```

---

## 📁 파일 구조
```
📦 rag_project/
├── app.py           # Streamlit 메인 앱 (Self-RAG 챗봇)
├── ragas_eval.py    # RAGAS 성능 평가 (GT 12개)
├── requirements.txt
├── .env             # API 키 (gitignore 필수)
└── README.md
```

---

## ⚠️ 면책 고지
본 서비스는 법률 정보 제공 목적이며, 법적 효력이 있는 법률 자문이 아닙니다.  
실제 법적 분쟁은 반드시 자격 있는 변호사와 상담하시기 바랍니다.
