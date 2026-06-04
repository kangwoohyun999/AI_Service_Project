"""
================================================
RAGAS 완전 가이드: 데이터셋 생성 → 평가까지
================================================
RAGAS = Retrieval Augmented Generation Assessment
RAG 시스템의 품질을 자동으로 측정하는 라이브러리

설치 명령어:
    pip install ragas langchain langchain-openai langchain-community faiss-cpu
"""

# ─────────────────────────────────────────
# STEP 0. 필요한 패키지 설치 확인
# ─────────────────────────────────────────
# 터미널에서 실행:
# pip install ragas==0.2.0 langchain langchain-openai langchain-community faiss-cpu

import os

# OpenAI API 키 설정 (환경변수 권장)
os.environ["OPENAI_API_KEY"] = "sk-..."   # 본인 키로 교체하세요


# ═══════════════════════════════════════════════════════════════
# PART 1. 📚 테스트 데이터셋 자동 생성 (TestsetGenerator)
# ═══════════════════════════════════════════════════════════════
"""
[핵심 아이디어]
- 원래는 사람이 직접 질문/정답 쌍을 만들어야 했음 (매우 힘들고 비용이 큼)
- RAGAS는 문서를 주면 LLM이 자동으로 질문/정답 쌍을 생성해줌!
- 생성된 데이터셋을 "골든 셋(Golden Set)" 이라고 부름
"""

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator  # RAGAS의 테스트셋 생성 도구
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms import default_transforms

# ── 1-1. 문서 로드 ────────────────────────────────────────────
# 방법 A: 텍스트 파일 1개 로드
loader = TextLoader("my_document.txt", encoding="utf-8")
documents = loader.load()

# 방법 B: 폴더 내 모든 .txt 파일 한꺼번에 로드
# loader = DirectoryLoader("./docs", glob="**/*.txt")
# documents = loader.load()

# 방법 C: PDF 파일 로드 (pip install pypdf 필요)
# from langchain_community.document_loaders import PyPDFLoader
# loader = PyPDFLoader("my_paper.pdf")
# documents = loader.load()

print(f"문서 로드 완료: {len(documents)}개")

# ── 1-2. 문서를 작은 청크로 나누기 ───────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 각 청크의 최대 글자 수
    chunk_overlap=200,    # 청크 간 겹치는 글자 수 (문맥 보존용)
)
chunks = splitter.split_documents(documents)
print(f"청크 분할 완료: {len(chunks)}개 청크")

# ── 1-3. 테스트셋 자동 생성 ──────────────────────────────────
# LLM이 문서를 읽고 다양한 유형의 질문을 자동으로 만들어줌
generator_llm = ChatOpenAI(model="gpt-4o-mini")     # 질문 생성용 LLM
critic_llm    = ChatOpenAI(model="gpt-4o-mini")     # 질문 품질 평가용 LLM
embeddings    = OpenAIEmbeddings()                   # 임베딩 모델

generator = TestsetGenerator.from_langchain(
    llm=generator_llm,
    embedding_model=embeddings,
)

# 테스트셋 생성! (testset_size: 생성할 질문 개수)
testset = generator.generate_with_langchain_docs(
    documents=chunks,
    testset_size=10,     # 처음엔 작게 시작 (비용 절약)
)

# 결과를 pandas DataFrame으로 변환
df = testset.to_pandas()
print("\n📋 생성된 테스트셋 미리보기:")
print(df[["user_input", "reference"]].head(3))
#  user_input          : 자동 생성된 질문
#  reference           : 자동 생성된 정답 (ground_truth)  # LLM이 문서를 참고해서 만들어낸 답변으로 완벽하지 않을 수 있어, 확인과 수정 필요

# CSV로 저장 (나중에 재사용 가능)
df.to_csv("golden_testset.csv", index=False, encoding="utf-8-sig")
print("✅ 테스트셋 저장 완료: golden_testset.csv")


# ═══════════════════════════════════════════════════════════════
# PART 2. 🤖 간단한 RAG 시스템 만들기
# ═══════════════════════════════════════════════════════════════
"""
[핵심 아이디어]
1. 문서를 벡터로 변환해서 벡터 DB에 저장
2. 질문이 들어오면 → 유사한 문서 조각을 찾아옴 (Retrieval)
3. 찾아온 문서를 참고해서 LLM이 답변 생성 (Generation)
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# ── 2-1. 벡터 DB 구축 ────────────────────────────────────────
# 문서 청크를 임베딩(숫자 벡터)으로 변환해서 저장
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever   = vectorstore.as_retriever(
    search_kwargs={"k": 3}   # 질문과 가장 유사한 문서 3개 검색
)
print("✅ 벡터 DB 구축 완료")

# ── 2-2. RAG 프롬프트 설정 ────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 참고하여 질문에 답하세요.
컨텍스트에 없는 내용은 "모르겠습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:""")

# ── 2-3. RAG 체인 구성 ───────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    """검색된 문서 리스트를 하나의 문자열로 합치기"""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
)

# ── 2-4. 테스트 질문으로 답변 생성 ───────────────────────────
def get_rag_answer(question: str) -> dict:
    """
    질문을 받아서 RAG 답변과 참고한 컨텍스트를 반환
    Returns:
        answer   : LLM이 생성한 답변
        contexts : 검색된 문서 조각들 (리스트)
    """
    # 관련 문서 검색
    relevant_docs = retriever.get_relevant_documents(question)
    contexts = [doc.page_content for doc in relevant_docs]

    # LLM으로 답변 생성
    answer = rag_chain.invoke(question).content

    return {"answer": answer, "contexts": contexts}

# 테스트
sample = get_rag_answer("한국의 수도는 어디인가요?")
print(f"\n질문: 한국의 수도는 어디인가요?")
print(f"답변: {sample['answer']}")
print(f"참고 문서 수: {len(sample['contexts'])}개")


# ═══════════════════════════════════════════════════════════════
# PART 3. 📊 RAGAS로 평가하기
# ═══════════════════════════════════════════════════════════════
"""
[4가지 핵심 지표]
┌──────────────────────────────────────────────────────────────┐
│ 1. Faithfulness    │ 답변이 컨텍스트에 근거하는가? (환각 탐지)│
│ 2. AnswerRelevancy │ 답변이 질문과 관련 있는가?               │
│ 3. ContextPrecision│ 검색된 문서 중 관련 문서 비율            │
│ 4. ContextRecall   │ 필요한 정보를 모두 검색했는가?           │
└──────────────────────────────────────────────────────────────┘
"""

import pandas as pd
from datasets import Dataset
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# ── 3-1. 저장해둔 테스트셋 불러오기 ──────────────────────────
df = pd.read_csv("golden_testset.csv")

# ── 3-2. 각 질문에 대해 RAG 답변 수집 ────────────────────────
print("\n🔄 RAG 답변 생성 중... (잠시 기다려 주세요)")
samples = []

for _, row in df.iterrows():
    question     = row["user_input"]    # 질문
    ground_truth = row["reference"]     # 자동 생성된 정답

    # RAG 시스템으로 답변 생성
    result = get_rag_answer(question)

    # RAGAS 평가용 샘플 생성
    sample = SingleTurnSample(
        user_input=question,
        response=result["answer"],
        retrieved_contexts=result["contexts"],
        reference=ground_truth,
    )
    samples.append(sample)

eval_dataset = EvaluationDataset(samples=samples)
print(f"✅ 평가 데이터셋 준비 완료: {len(samples)}개 샘플")

# ── 3-3. RAGAS 평가 실행 ─────────────────────────────────────
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

result = evaluate(
    dataset=eval_dataset,
    metrics=[
        Faithfulness(llm=evaluator_llm),          # 사실 일치도
        AnswerRelevancy(llm=evaluator_llm),       # 답변 관련성
        ContextPrecision(llm=evaluator_llm),      # 컨텍스트 정밀도
        ContextRecall(llm=evaluator_llm),         # 컨텍스트 재현율
    ],
)

# ── 3-4. 결과 출력 ───────────────────────────────────────────
print("\n" + "="*50)
print("📊 RAGAS 평가 결과")
print("="*50)
print(f"  Faithfulness     (사실 일치도) : {result['faithfulness']:.3f}")
print(f"  Answer Relevancy (답변 관련성) : {result['answer_relevancy']:.3f}")
print(f"  Context Precision(검색 정밀도) : {result['context_precision']:.3f}")
print(f"  Context Recall   (검색 재현율) : {result['context_recall']:.3f}")
print("="*50)
print("  점수는 0~1 사이, 1에 가까울수록 좋음")

# 결과를 DataFrame으로도 볼 수 있음
result_df = result.to_pandas()
print("\n세부 결과 (질문별):")
print(result_df[["user_input", "faithfulness", "answer_relevancy"]].to_string())

# 결과 저장
result_df.to_csv("ragas_results.csv", index=False, encoding="utf-8-sig")
print("\n✅ 결과 저장 완료: ragas_results.csv")







# ═══════════════════════════════════════════════════════════════
# PART 4. 🎯 가장 쉬운 방법: 직접 데이터셋 만들기
# ═══════════════════════════════════════════════════════════════
"""
TestsetGenerator가 복잡하다면?
→ 직접 소수의 샘플을 만들어서 평가해보세요!
학생 실습에 가장 적합한 방법입니다.
"""

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# ── 직접 데이터 입력 ──────────────────────────────────────────
# 구조: (질문, RAG답변, 검색된문서들, 정답)
manual_data = [
    {
        "user_input"          : "파이썬은 어떤 언어인가요?",
        "response"            : "파이썬은 간결하고 읽기 쉬운 문법의 고급 프로그래밍 언어입니다.",
        "retrieved_contexts"  : [
            "파이썬(Python)은 1991년 귀도 반 로썸이 발표한 고급 프로그래밍 언어로, "
            "문법이 간결하고 읽기 쉬워 초보자에게 인기가 높습니다.",
            "파이썬은 인터프리터 언어로, 코드를 한 줄씩 실행합니다."
        ],
        "reference"           : "파이썬은 귀도 반 로썸이 만든 고급 프로그래밍 언어입니다.",
    },
    {
        "user_input"          : "머신러닝이란 무엇인가요?",
        "response"            : "머신러닝은 데이터로부터 패턴을 학습하는 AI 기술입니다.",
        "retrieved_contexts"  : [
            "머신러닝은 명시적 프로그래밍 없이 데이터에서 학습하는 알고리즘 연구 분야입니다.",
        ],
        "reference"           : "머신러닝은 데이터로부터 자동으로 학습하는 알고리즘 연구 분야입니다.",
    },
]

# EvaluationDataset 구성
samples = [SingleTurnSample(**d) for d in manual_data]
eval_dataset = EvaluationDataset(samples=samples)

# 평가 실행
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

result = evaluate(
    dataset=eval_dataset,
    metrics=[
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
    ],
)

print("\n📊 수동 데이터 평가 결과:")
print(f"  Faithfulness     : {result['faithfulness']:.3f}")
print(f"  Answer Relevancy : {result['answer_relevancy']:.3f}")


# ═══════════════════════════════════════════════════════════════
# PART 5. 📈 결과 시각화 (선택사항)
# ═══════════════════════════════════════════════════════════════
"""
평가 결과를 레이더 차트로 시각화하면 한눈에 파악하기 쉬움
pip install matplotlib
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트 설정 (Mac: AppleGothic, Windows: Malgun Gothic)
import platform
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

def plot_ragas_radar(scores: dict, title: str = "RAGAS 평가 결과"):
    """
    RAGAS 4가지 지표를 레이더 차트로 시각화

    Args:
        scores: {"faithfulness": 0.85, "answer_relevancy": 0.90, ...}
        title : 차트 제목
    """
    labels   = list(scores.keys())
    values   = list(scores.values())
    n        = len(labels)
    angles   = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    # 시작점으로 닫기
    values  += values[:1]
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 채우기
    ax.fill(angles, values, color="#7F77DD", alpha=0.3)
    ax.plot(angles, values, color="#534AB7", linewidth=2, marker="o", markersize=6)

    # 라벨
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig("ragas_radar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ 레이더 차트 저장: ragas_radar.png")

# 사용 예시
# plot_ragas_radar({
#     "Faithfulness"    : result["faithfulness"],
#     "Ans. Relevancy"  : result["answer_relevancy"],
#     "Ctx. Precision"  : result["context_precision"],
#     "Ctx. Recall"     : result["context_recall"],
# })


# ═══════════════════════════════════════════════════════════════
# 📝 요약: 학생용 치트시트
# ═══════════════════════════════════════════════════════════════
"""
┌────────────────────────────────────────────────────────────────┐
│               RAGAS 사용 핵심 요약                              │
├─────────────────────────┬──────────────────────────────────────┤
│ 테스트셋 쉽게 만들기     │ PART 4처럼 직접 딕셔너리로 작성       │
│ 테스트셋 자동 생성       │ PART 1: TestsetGenerator 사용         │
│ 평가 실행               │ evaluate(dataset, metrics=[...])      │
│ 점수 해석               │ 0~1, 높을수록 좋음                    │
│                         │ 0.8 이상: 우수 / 0.6 이하: 개선 필요  │
├─────────────────────────┼──────────────────────────────────────┤
│ Faithfulness가 낮다면?   │ 프롬프트에 "컨텍스트만 참고" 강조      │
│ AnswerRelevancy가 낮다면?│ 답변이 너무 장황함 → 프롬프트 수정    │
│ ContextPrecision이 낮다?│ 검색기(retriever) 파라미터 조정        │
│ ContextRecall이 낮다면? │ 청크 크기 줄이기 / 검색 개수 늘리기   │
└─────────────────────────┴──────────────────────────────────────┘
"""
