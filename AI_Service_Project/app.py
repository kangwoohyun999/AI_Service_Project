"""
⚖️ 법률 사건 Self-RAG 챗봇
대상: 일반 시민 — 사건 발생 시 즉시 관련 법률/판례/절차 조회
기술: LangChain + Self-RAG + FAISS + Memory + Streamlit
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="법률 사건 Q&A 챗봇",
    page_icon="⚖️",
    layout="wide",
)

st.markdown("""
<style>
    /* ── 라이트 모드 기본 배경 ── */
    [data-testid="stAppViewContainer"] { background: #f8f9fb; }
    [data-testid="stSidebar"]          { background: #ffffff; border-right: 1px solid #e5e7eb; }

    /* ── 레이아웃 고정: 화면 비율 변화에도 컬럼이 흔들리지 않게 ── */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        align-items: flex-start !important;
        gap: 1.2rem !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        min-width: 0 !important;      /* flex 항목이 넘치지 않도록 */
        flex-shrink: 0 !important;    /* 비율 유지 — 찌그러지지 않음 */
    }

    /* ── 헤더 ── */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.4rem;
    }
    .main-header h1  { margin: 0; font-size: 1.8rem; letter-spacing: -0.4px; }
    .main-header .sub { margin: 0.35rem 0 0; opacity: 0.82; font-size: 0.9rem; }
    .main-header .warn {
        margin: 0.8rem 0 0;
        font-size: 0.78rem;
        background: #ffffff22;
        border: 1px solid #ffffff44;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        color: #fef3c7;
    }

    /* ── 면책 고지 ── */
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.78rem;
        color: #9a3412;
        margin-top: 0.8rem;
    }

    /* ── RAG 태그 ── */
    .tag-rag  { font-size: 0.75rem; color: #15803d; }
    .tag-llm  { font-size: 0.75rem; color: #b45309; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚖️ 법률 사건 Q&A 챗봇</h1>
  <p class="sub">사건 발생 시 즉시 관련 법률 · 판례 · 대처 절차를 확인하세요 | Self-RAG 기반</p>
  <p class="warn">⚠️ 본 서비스는 법률 정보 제공 목적이며, 법적 효력이 있는 법률 자문이 아닙니다. 중요 사안은 반드시 변호사와 상담하세요.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "memory": None,
    "vectordb": None,
    "self_rag": None,
    "process_log": [],
    "pending_question": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.memory is None:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=5, return_messages=True, memory_key="chat_history"
    )


# ─────────────────────────────────────────────
# Pydantic 모델 (Self-RAG 판단 토큰)
# ─────────────────────────────────────────────
class RetrievalResponse(BaseModel):
    Reasoning: str = Field(description="법률 문서 검색 필요 여부 추론 (2~3문장)")
    Retrieve: Literal["Yes", "No"] = Field(description="검색 필요 여부")

class RelevanceResponse(BaseModel):
    Reasoning: str = Field(description="문서 관련성 추론 (2~3문장)")
    ISREL: Literal["Relevant", "Irrelevant"] = Field(description="관련성 평가")

class GenerationResponse(BaseModel):
    response: str = Field(description="법률 정보 답변")

class SupportResponse(BaseModel):
    Reasoning: str = Field(description="근거 평가 추론 (2~3문장)")
    ISSUP: Literal["Fully supported", "Partially supported", "No support"] = Field(description="근거 평가")

class UtilityResponse(BaseModel):
    Reasoning: str = Field(description="유용성 평가 추론")
    ISUSE: Literal[1, 2, 3, 4, 5] = Field(description="유용성 점수 (1~5)")


# ─────────────────────────────────────────────
# Self-RAG 클래스 (예시 ipynb 구조 기반)
# ─────────────────────────────────────────────
class LegalSelfRAG:
    def __init__(self, vectorstore, llm, top_k=4):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.log = []

        # 1. 검색 필요 여부 판단
        retrieval_prompt = PromptTemplate(
            input_variables=["query", "chat_history"],
            template="""당신은 법률 Q&A 챗봇의 검색 판단 모듈입니다.
대화 이력과 질문을 보고, 법률 문서(법조문, 판례, 절차 안내)를 검색해야 하는지 판단하세요.

검색이 필요한 경우:
- 특정 법률 조항, 처벌 기준, 민사/형사 절차 질문
- 판례나 법적 해석이 필요한 경우
- 계약, 손해배상, 고소/고발 관련 질문

검색 불필요한 경우:
- 단순 인사, 일반 상식 질문
- 이미 대화에서 충분히 다룬 내용

대화 이력:
{chat_history}

질문: {query}"""
        )
        self.retrieval_chain = retrieval_prompt | llm.with_structured_output(RetrievalResponse)

        # 2+3. 관련성 평가
        relevance_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""아래 법률 문서가 질문에 답하는 데 유용한지 판단하세요.
유용하면 "Relevant", 아니면 "Irrelevant"로 응답하세요.

질문: {query}
법률 문서: {context}"""
        )
        self.relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceResponse)

        # 4. 답변 생성
        generation_prompt = PromptTemplate(
            input_variables=["query", "context", "chat_history"],
            template="""당신은 법률 정보를 제공하는 챗봇입니다.
아래 법률 문서와 대화 이력을 바탕으로 질문에 정확하고 이해하기 쉽게 답변하세요.

답변 형식:
- 관련 법 조항 또는 기준 먼저 제시
- 실제 상황에서 취할 수 있는 절차/행동 안내
- 문서에 없는 내용은 추측하지 말 것
- 마지막에 "중요한 사안은 변호사 상담을 권장합니다" 추가

대화 이력:
{chat_history}

법률 문서:
{context}

질문: {query}"""
        )
        self.generation_chain = generation_prompt | llm.with_structured_output(GenerationResponse)

        # 5. 지원도 평가
        support_prompt = PromptTemplate(
            input_variables=["query", "response", "context"],
            template="""답변이 제시된 법률 문서에 얼마나 근거하는지 평가하세요.
1. Fully supported   - 모든 내용이 문서 근거
2. Partially supported - 일부만 근거
3. No support        - 문서와 무관

질문: {query}
답변: {response}
법률 문서: {context}"""
        )
        self.support_chain = support_prompt | llm.with_structured_output(SupportResponse)

        # 6. 유용성 평가
        utility_prompt = PromptTemplate(
            input_variables=["query", "response"],
            template="""다음 법률 답변이 질문자에게 얼마나 실질적으로 유용한지 1~5점으로 평가하세요.
(5점: 즉시 행동 가능한 구체적 정보, 1점: 전혀 도움 안 됨)

질문: {query}
답변: {response}"""
        )
        self.utility_chain = utility_prompt | llm.with_structured_output(UtilityResponse)

    def _fmt_history(self, memory):
        msgs = memory.chat_memory.messages
        lines = []
        for m in msgs[-6:]:
            role = "사용자" if isinstance(m, HumanMessage) else "챗봇"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines) if lines else "(없음)"

    def process_query(self, query: str, memory) -> dict:
        self.log = []
        chat_history = self._fmt_history(memory)

        # 1단계
        self.log.append(("1️⃣ 검색 판단", "법률 문서 검색 필요 여부 판단 중..."))
        ret = self.retrieval_chain.invoke({"query": query, "chat_history": chat_history})
        self.log[-1] = ("1️⃣ 검색 판단", f"**{ret.Retrieve}** — {ret.Reasoning}")

        if ret.Retrieve == "No":
            self.log.append(("⚡ 직접 생성", "일반 지식으로 답변합니다"))
            gen = self.generation_chain.invoke({"query": query, "context": "(검색 없음)", "chat_history": chat_history})
            return {"answer": gen.response, "used_rag": False, "log": self.log}

        # 2단계
        self.log.append(("2️⃣ 법률 문서 검색", f"FAISS에서 상위 {self.top_k}개 문서 검색 중..."))
        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        contexts = [d.page_content for d in docs]
        self.log[-1] = ("2️⃣ 법률 문서 검색", f"{len(contexts)}개 문서 검색 완료")

        # 3단계
        self.log.append(("3️⃣ 관련성 필터링", "각 문서의 법률적 관련성 평가 중..."))
        relevant = []
        for ctx in contexts:
            rel = self.relevance_chain.invoke({"query": query, "context": ctx})
            if rel.ISREL == "Relevant":
                relevant.append(ctx)
        self.log[-1] = ("3️⃣ 관련성 필터링", f"{len(relevant)}/{len(contexts)}개 관련 문서 선별")

        if not relevant:
            self.log.append(("⚠️ 관련 문서 없음", "업로드된 문서에서 관련 법률을 찾지 못했습니다"))
            gen = self.generation_chain.invoke({"query": query, "context": "(관련 법률 문서 없음 — 일반 지식 기반)", "chat_history": chat_history})
            return {"answer": gen.response, "used_rag": False, "log": self.log}

        # 4단계
        self.log.append(("4️⃣ 법률 답변 생성", f"{len(relevant)}개 문서 기반 답변 생성 중..."))
        responses = []
        for ctx in relevant:
            gen = self.generation_chain.invoke({"query": query, "context": ctx, "chat_history": chat_history})
            responses.append((gen.response, ctx))
        self.log[-1] = ("4️⃣ 법률 답변 생성", f"{len(responses)}개 후보 답변 생성 완료")

        # 5~6단계
        self.log.append(("5️⃣ 품질 평가", "지원도 · 유용성 평가 후 최적 답변 선택 중..."))
        assessed = []
        for resp, ctx in responses:
            sup = self.support_chain.invoke({"query": query, "response": resp, "context": ctx})
            util = self.utility_chain.invoke({"query": query, "response": resp})
            assessed.append((resp, sup.ISSUP, int(util.ISUSE)))

        best = self._select_best(assessed)
        self.log[-1] = ("5️⃣ 품질 평가", f"최종 선택 → 지원도: **{best[1]}**, 유용성: **{best[2]}/5**")

        return {"answer": best[0], "used_rag": True, "log": self.log}

    def _select_best(self, responses):
        for level in ["Fully supported", "Partially supported"]:
            subset = [r for r in responses if r[1] == level]
            if subset:
                return max(subset, key=lambda x: x[2])
        return max(responses, key=lambda x: x[2])


# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("OpenAI API Key", type="password",
                            value=os.getenv("OPENAI_API_KEY", ""))
    model_name = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"], index=0)

    st.divider()
    st.subheader("📂 법률 문서 업로드")
    st.caption("형법, 민법, 판례집, 법령 PDF 등을 업로드하세요")
    uploaded_files = st.file_uploader(
        "법률 PDF 업로드 (복수 가능)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("청크 크기", 200, 1000, 400, 50)
    with col2:
        chunk_overlap = st.number_input("청크 겹침", 0, 200, 80, 20)
    top_k = st.slider("검색 문서 수", 2, 8, 4)

    build_btn = st.button("🔨 벡터 DB 구축", use_container_width=True, type="primary")
    if build_btn:
        if not api_key:
            st.error("API Key를 입력하세요.")
        elif not uploaded_files:
            st.error("PDF 파일을 업로드하세요.")
        else:
            with st.spinner("법률 문서 처리 중..."):
                try:
                    import tempfile, shutil
                    tmpdir = tempfile.mkdtemp()
                    all_docs = []
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
                    for uf in uploaded_files:
                        path = os.path.join(tmpdir, uf.name)
                        with open(path, "wb") as f:
                            f.write(uf.read())
                        loader = PyPDFLoader(path)
                        docs = loader.load_and_split(splitter)
                        all_docs.extend(docs)

                    embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
                    vectordb = FAISS.from_documents(all_docs, embedding)
                    vectordb.save_local("faiss_legal_index")

                    llm = ChatOpenAI(model=model_name, max_tokens=2000,
                                     temperature=0.1, api_key=api_key)
                    st.session_state.vectordb = vectordb
                    st.session_state.self_rag = LegalSelfRAG(vectordb, llm, top_k=top_k)
                    st.session_state.chat_history = []
                    st.session_state.memory = ConversationBufferWindowMemory(
                        k=5, return_messages=True, memory_key="chat_history"
                    )
                    shutil.rmtree(tmpdir)
                    st.success(f"✅ {len(all_docs)}개 청크 구축 완료!")
                except Exception as e:
                    st.error(f"오류: {e}")

    load_btn = st.button("📥 저장된 DB 불러오기", use_container_width=True)
    if load_btn:
        if not api_key:
            st.error("API Key를 입력하세요.")
        elif not os.path.exists("faiss_legal_index"):
            st.warning("저장된 DB가 없습니다.")
        else:
            with st.spinner("DB 로딩 중..."):
                try:
                    embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
                    vectordb = FAISS.load_local("faiss_legal_index", embedding,
                                                allow_dangerous_deserialization=True)
                    llm = ChatOpenAI(model=model_name, max_tokens=2000,
                                     temperature=0.1, api_key=api_key)
                    st.session_state.vectordb = vectordb
                    st.session_state.self_rag = LegalSelfRAG(vectordb, llm, top_k=top_k)
                    st.success("✅ 법률 DB 로드 완료!")
                except Exception as e:
                    st.error(f"오류: {e}")

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5, return_messages=True, memory_key="chat_history"
        )
        st.rerun()

    st.divider()
    if st.session_state.vectordb:
        st.success("🟢 법률 DB 연결됨")
    else:
        st.warning("🔴 DB 없음 — 문서를 업로드하세요")


# ─────────────────────────────────────────────
# 메인: 빠른 사건 유형 선택 + 채팅
# ─────────────────────────────────────────────

# 빠른 사건 유형 버튼
st.markdown("#### 🚨 사건 유형 빠른 선택")
case_types = [
    ("🚗 교통사고", "교통사고가 발생했습니다. 피해자로서 취해야 할 법적 조치와 손해배상 청구 방법을 알려주세요."),
    ("🏠 임차인 분쟁", "집주인이 보증금을 돌려주지 않습니다. 어떻게 대처해야 하나요?"),
    ("💼 부당해고", "갑자기 해고 통보를 받았습니다. 부당해고 여부와 구제 방법을 알고 싶습니다."),
    ("💳 사기 피해", "온라인 거래에서 사기를 당했습니다. 고소 방법과 피해 회복 절차를 알려주세요."),
    ("👊 폭행 피해", "폭행을 당했습니다. 고소 절차와 피해 보상을 받을 수 있는 방법을 알고 싶습니다."),
    ("📝 계약 분쟁", "계약 상대방이 계약을 이행하지 않습니다. 법적으로 어떻게 대응할 수 있나요?"),
]

cols = st.columns(3)
for i, (label, question) in enumerate(case_types):
    with cols[i % 3]:
        if st.button(label, use_container_width=True, key=f"case_{i}"):
            st.session_state.pending_question = question

st.divider()
st.markdown("#### 💬 대화")

# 대화 기록 출력
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            used = msg.get("used_rag")
            if used is True:
                st.markdown('<span class="tag-rag">📖 법률 문서 기반 (Self-RAG)</span>', unsafe_allow_html=True)
            elif used is False:
                st.markdown('<span class="tag-llm">⚡ 일반 지식 기반</span>', unsafe_allow_html=True)

# 입력 처리 (버튼 클릭 or 직접 입력)
user_input = st.chat_input("사건 내용을 자세히 설명하거나 법률 질문을 입력하세요...")
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

if user_input:
    if not st.session_state.self_rag:
        st.warning("사이드바에서 법률 PDF를 업로드하고 DB를 구축하세요.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Self-RAG가 법률 문서를 분석 중입니다..."):
                result = st.session_state.self_rag.process_query(
                    user_input, st.session_state.memory
                )
            answer = result["answer"]
            used_rag = result["used_rag"]
            st.markdown(answer)
            if used_rag:
                st.markdown('<span class="tag-rag">📖 법률 문서 기반 (Self-RAG)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="tag-llm">⚡ 일반 지식 기반</span>', unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant", "content": answer, "used_rag": used_rag
        })
        st.session_state.process_log = result["log"]
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(answer)
        st.rerun()

# 면책 고지
st.markdown("""
<div class="disclaimer">
⚠️ 본 챗봇은 법률 정보 제공 목적이며, 법적 효력이 있는 법률 자문이 아닙니다.
실제 법적 분쟁에는 반드시 자격 있는 변호사와 상담하시기 바랍니다.
</div>
""", unsafe_allow_html=True)




