"""
⚖️ 법률 사건 Self-RAG 챗봇
대상: 일반 시민 — 사건 발생 시 즉시 관련 법률/판례/절차 조회
기술: LangChain + Self-RAG + FAISS + Memory + Streamlit
[최종발표 버전] RAG 답변 vs 순수 LLM 답변 나란히 비교 출력
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # OMP 중복 오류 해결

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")  # LangChain 경고 숨김
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

st.set_page_config(
    page_title="법률 사건 Q&A 챗봇",
    page_icon="⚖️",
    layout="wide",
)

st.markdown("""
<style>
    /* 기본 Streamlit 사이드바 + 토글 버튼 숨김 */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] { display: none !important; }

    /* 메인 영역 전체 폭 */
    [data-testid="stAppViewContainer"] { background: #f8f9fb; }
    .main .block-container {
        max-width: 100% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* 고정 설정 버튼 (스크롤해도 우상단 고정) */
    #settings-fab {
        position: fixed !important;
        top: 12px !important;
        right: 16px !important;
        z-index: 999999 !important;
        background: #2563eb;
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 2px 10px rgba(37,99,235,0.4);
    }
    #settings-fab:hover { background: #1d4ed8; }

    /* 어두운 오버레이 */
    #modal-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.4);
        z-index: 99990;
    }
    body.modal-open #modal-overlay { display: block; }

    /* 설정 패널 (우측 슬라이드, 메인 폭 변화 없음) */
    #settings-panel {
        position: fixed !important;
        top: 0 !important;
        right: -400px !important;
        width: 360px !important;
        height: 100vh !important;
        background: #fff;
        border-left: 1px solid #d1d5db;
        box-shadow: -6px 0 24px rgba(0,0,0,0.14);
        z-index: 99995 !important;
        overflow-y: auto;
        transition: right 0.28s cubic-bezier(.4,0,.2,1);
    }
    body.modal-open #settings-panel { right: 0 !important; }

    /* 패널 헤더 */
    #panel-hdr {
        position: sticky; top: 0; z-index: 2;
        background: #1e3a8a; color: #fff;
        padding: 13px 18px;
        font-weight: 700; font-size: 15px;
        display: flex; align-items: center; justify-content: space-between;
    }
    #panel-hdr button {
        background: none; border: none; color: #fff;
        font-size: 20px; cursor: pointer; line-height: 1; padding: 0 2px;
    }

    /* 패널 안 Streamlit 사이드바 정리 */
    #settings-panel [data-testid="stSidebar"] {
        display: block !important;
        position: static !important;
        width: 100% !important;
        height: auto !important;
        background: #fff !important;
        border: none !important;
        box-shadow: none !important;
        min-height: unset !important;
    }

    /* 레이아웃 고정 */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        align-items: flex-start !important;
        gap: 1.2rem !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        min-width: 0 !important; flex-shrink: 0 !important;
    }

    /* 앱 헤더 */
    .main-header {
        background: linear-gradient(135deg,#1e3a8a,#2563eb);
        color:#fff; padding:1.4rem 2rem; border-radius:12px; margin-bottom:1.2rem;
    }
    .main-header h1  { margin:0; font-size:1.7rem; letter-spacing:-.3px; }
    .main-header .sub { margin:.3rem 0 0; opacity:.85; font-size:.88rem; }
    .main-header .warn {
        margin:.7rem 0 0; font-size:.77rem;
        background:#ffffff22; border:1px solid #ffffff44;
        border-radius:6px; padding:.35rem .8rem; color:#fef3c7;
    }

    /* 비교 패널 헤더 */
    .panel-header-rag {
        background:linear-gradient(90deg,#1e3a8a,#2563eb);
        color:#fff; padding:.5rem 1rem; border-radius:8px 8px 0 0;
        font-weight:700; font-size:.9rem;
    }
    .panel-header-llm {
        background:linear-gradient(90deg,#92400e,#d97706);
        color:#fff; padding:.5rem 1rem; border-radius:8px 8px 0 0;
        font-weight:700; font-size:.9rem;
    }
    .disclaimer {
        background:#fff7ed; border:1px solid #fdba74;
        border-radius:8px; padding:.65rem 1rem;
        font-size:.77rem; color:#9a3412; margin-top:.8rem;
    }
    .tag-rag { font-size:.75rem; color:#15803d; }
    .tag-llm { font-size:.75rem; color:#b45309; }
</style>

<!-- 고정 설정 버튼 -->
<button id="settings-fab" onclick="openPanel()">&#9881;&#65039; 설정</button>

<!-- 오버레이 -->
<div id="modal-overlay" onclick="closePanel()"></div>

<!-- 설정 패널 -->
<div id="settings-panel">
  <div id="panel-hdr">
    <span>&#9881;&#65039; 설정</span>
    <button onclick="closePanel()">&#x2715;</button>
  </div>
  <div id="panel-content" style="padding:0;"></div>
</div>

<script>
(function(){
  window.openPanel  = function(){ document.body.classList.add('modal-open'); };
  window.closePanel = function(){ document.body.classList.remove('modal-open'); };

  function relocate() {
    var sb  = document.querySelector('[data-testid="stSidebar"]');
    var dst = document.getElementById('panel-content');
    if (!sb || !dst || dst.contains(sb)) return;
    sb.style.cssText = [
      'display:block!important',
      'position:static!important',
      'width:100%!important',
      'height:auto!important',
      'box-shadow:none!important',
      'border:none!important',
      'background:#fff!important',
      'min-height:unset!important'
    ].join(';');
    dst.appendChild(sb);
  }

  function init() {
    relocate();
    new MutationObserver(function(ms){
      for(var i=0;i<ms.length;i++){
        if(ms[i].addedNodes.length){ relocate(); break; }
      }
    }).observe(document.body,{childList:true,subtree:true});
  }

  if(document.readyState==='loading')
    document.addEventListener('DOMContentLoaded',function(){setTimeout(init,500);});
  else
    setTimeout(init,500);
})();
</script>""", unsafe_allow_html=True)

# ── 헤더 ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚖️ 법률 사건 Q&A 챗봇</h1>
  <p class="sub">사건 발생 시 즉시 관련 법률 · 판례 · 대처 절차를 확인하세요 | Self-RAG vs 순수 LLM 비교</p>
  <p class="warn">⚠️ 본 서비스는 법률 정보 제공 목적이며, 법적 효력이 있는 법률 자문이 아닙니다. 중요 사안은 반드시 변호사와 상담하세요.</p>
</div>
""", unsafe_allow_html=True)

# ── 세션 상태 ─────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "memory": None,
    "vectordb": None,
    "self_rag": None,
    "llm_direct": None,
    "process_log": [],
    "pending_question": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.memory is None:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=5, return_messages=True, memory_key="chat_history"
    )


# ── Pydantic 모델 ─────────────────────────────────────────────────
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


# ── LegalSelfRAG ──────────────────────────────────────────────────
class LegalSelfRAG:
    def __init__(self, vectorstore, llm, top_k=4):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.log = []

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

        relevance_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""아래 법률 문서가 질문에 답하는 데 유용한지 판단하세요.
유용하면 "Relevant", 아니면 "Irrelevant"로 응답하세요.

질문: {query}
법률 문서: {context}"""
        )
        self.relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceResponse)

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

        self.log.append(("1️⃣ 검색 판단", "법률 문서 검색 필요 여부 판단 중..."))
        ret = self.retrieval_chain.invoke({"query": query, "chat_history": chat_history})
        self.log[-1] = ("1️⃣ 검색 판단", f"**{ret.Retrieve}** — {ret.Reasoning}")

        if ret.Retrieve == "No":
            self.log.append(("⚠️ 관련 문서 없음", "업로드된 문서에서 관련 법률을 찾지 못했습니다"))
            return {"answer": "문서에 관련 법 조항이 없습니다.", "used_rag": False, "no_doc": True, "log": self.log}

        self.log.append(("2️⃣ 법률 문서 검색", f"FAISS에서 상위 {self.top_k}개 문서 검색 중..."))
        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        contexts = [d.page_content for d in docs]
        self.log[-1] = ("2️⃣ 법률 문서 검색", f"{len(contexts)}개 문서 검색 완료")

        self.log.append(("3️⃣ 관련성 필터링", "각 문서의 법률적 관련성 평가 중..."))
        relevant = []
        for ctx in contexts:
            rel = self.relevance_chain.invoke({"query": query, "context": ctx})
            if rel.ISREL == "Relevant":
                relevant.append(ctx)
        self.log[-1] = ("3️⃣ 관련성 필터링", f"{len(relevant)}/{len(contexts)}개 관련 문서 선별")

        if not relevant:
            self.log.append(("⚠️ 관련 문서 없음", "업로드된 문서에서 관련 법률을 찾지 못했습니다"))
            return {"answer": "문서에 관련 법 조항이 없습니다.", "used_rag": False, "no_doc": True, "log": self.log}

        self.log.append(("4️⃣ 법률 답변 생성", f"{len(relevant)}개 문서 기반 답변 생성 중..."))
        responses = []
        for ctx in relevant:
            gen = self.generation_chain.invoke({"query": query, "context": ctx, "chat_history": chat_history})
            responses.append((gen.response, ctx))
        self.log[-1] = ("4️⃣ 법률 답변 생성", f"{len(responses)}개 후보 답변 생성 완료")

        self.log.append(("5️⃣ 품질 평가", "지원도 · 유용성 평가 후 최적 답변 선택 중..."))
        assessed = []
        for resp, ctx in responses:
            sup = self.support_chain.invoke({"query": query, "response": resp, "context": ctx})
            util = self.utility_chain.invoke({"query": query, "response": resp})
            assessed.append((resp, sup.ISSUP, int(util.ISUSE)))

        best = self._select_best(assessed)
        if best is None:
            self.log[-1] = ("⚠️ 관련 문서 없음", "검색된 문서가 질문과 관련이 없습니다")
            return {"answer": "문서에 관련 법 조항이 없습니다.", "used_rag": False, "no_doc": True, "log": self.log}

        self.log[-1] = ("5️⃣ 품질 평가", f"최종 선택 → 지원도: **{best[1]}**, 유용성: **{best[2]}/5**")
        return {"answer": best[0], "used_rag": True, "log": self.log}

    def _select_best(self, responses):
        for level in ["Fully supported", "Partially supported"]:
            subset = [r for r in responses if r[1] == level]
            if subset:
                return max(subset, key=lambda x: x[2])
        # No support만 남은 경우 → 문서 근거 없음 처리
        return None


# ── 순수 LLM 직접 답변 (RAG 없음) ────────────────────────────────
def get_llm_direct_answer(query: str, llm, memory) -> str:
    """벡터DB 검색 없이 LLM 파라메트릭 지식만으로 답변"""
    msgs = memory.chat_memory.messages[-6:]
    history_lines = []
    for m in msgs:
        role = "사용자" if isinstance(m, HumanMessage) else "챗봇"
        history_lines.append(f"{role}: {m.content}")
    chat_history = "\n".join(history_lines) if history_lines else "(없음)"

    prompt = PromptTemplate(
        input_variables=["query", "chat_history"],
        template="""당신은 법률 정보를 제공하는 AI 어시스턴트입니다.
외부 문서 검색 없이 학습된 지식만으로 아래 질문에 답변하세요.

대화 이력:
{chat_history}

질문: {query}

답변 시 주의:
- 알고 있는 법률 지식을 바탕으로 최대한 정확하게 답하세요.
- 불확실한 내용은 "확인이 필요합니다"라고 명시하세요.
- 마지막에 "중요한 사안은 변호사 상담을 권장합니다" 추가"""
    )
    chain = prompt | llm
    result = chain.invoke({"query": query, "chat_history": chat_history})
    return result.content if hasattr(result, "content") else str(result)


# ── 사이드바 ──────────────────────────────────────────────────────
with st.sidebar:

    # Streamlit Cloud secrets 우선, 없으면 환경변수, 없으면 빈칸
    _default_key = ""
    try:
        _default_key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    except Exception:
        _default_key = os.getenv("OPENAI_API_KEY", "")

    api_key = st.text_input("OpenAI API Key", type="password", value=_default_key)
    model_name = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"], index=0)

    st.divider()
    st.subheader("📂 DB 구축")
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

    # FAISS 저장 경로: Streamlit Cloud는 /tmp만 쓰기 가능
    FAISS_PATH = "/tmp/faiss_legal_index"

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
                    vectordb.save_local(FAISS_PATH)
                    llm = ChatOpenAI(model=model_name, max_tokens=2000, temperature=0.1, api_key=api_key)

                    st.session_state.vectordb = vectordb
                    st.session_state.self_rag = LegalSelfRAG(vectordb, llm, top_k=top_k)
                    st.session_state.llm_direct = llm
                    st.session_state.chat_history = []
                    st.session_state.memory = ConversationBufferWindowMemory(
                        k=5, return_messages=True, memory_key="chat_history"
                    )
                    shutil.rmtree(tmpdir)
                    st.success(f"✅ {len(all_docs)}개 청크 구축 완료!")
                except Exception as e:
                    st.error(f"오류: {e}")

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5, return_messages=True, memory_key="chat_history"
        )
        st.rerun()

    st.divider()
    st.subheader("📥 기존 FAISS DB 업로드")
    st.caption("이전에 저장한 index.faiss / index.pkl 파일을 업로드하세요")
    faiss_file = st.file_uploader("index.faiss", type=["faiss"], key="faiss_file")
    pkl_file   = st.file_uploader("index.pkl",   type=["pkl"],   key="pkl_file")

    load_faiss_btn = st.button("📂 업로드된 DB 불러오기", use_container_width=True)
    if load_faiss_btn:
        if not api_key:
            st.error("API Key를 입력하세요.")
        elif not faiss_file or not pkl_file:
            st.warning("index.faiss 와 index.pkl 파일을 모두 업로드하세요.")
        else:
            with st.spinner("DB 로딩 중..."):
                try:
                    import shutil as _shutil
                    os.makedirs(FAISS_PATH, exist_ok=True)
                    with open(os.path.join(FAISS_PATH, "index.faiss"), "wb") as f:
                        f.write(faiss_file.read())
                    with open(os.path.join(FAISS_PATH, "index.pkl"), "wb") as f:
                        f.write(pkl_file.read())
                    embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
                    vectordb = FAISS.load_local(FAISS_PATH, embedding,
                                                allow_dangerous_deserialization=True)
                    llm = ChatOpenAI(model=model_name, max_tokens=2000, temperature=0.1, api_key=api_key)
                    st.session_state.vectordb = vectordb
                    st.session_state.self_rag = LegalSelfRAG(vectordb, llm, top_k=top_k)
                    st.session_state.llm_direct = llm
                    st.success("✅ FAISS DB 로드 완료!")
                except Exception as e:
                    st.error(f"오류: {e}")

    st.divider()
    if st.session_state.vectordb:
        st.success("🟢 법률 DB 연결됨")
    else:
        st.warning("🔴 DB 없음 — 문서를 업로드하세요")



# ── 메인 ─────────────────────────────────────────────────────────

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

# ── 비교 컬럼 헤더 ────────────────────────────────────────────────
col_rag, col_llm = st.columns(2)
with col_rag:
    st.markdown("""
    <div class="panel-header-rag">
        📖 Self-RAG 답변 &nbsp;|&nbsp; 법률 문서 기반 검색·생성
    </div>
    """, unsafe_allow_html=True)
with col_llm:
    st.markdown("""
    <div class="panel-header-llm">
        🤖 순수 LLM 답변 &nbsp;|&nbsp; 학습 지식만 사용 (문서 검색 없음)
    </div>
    """, unsafe_allow_html=True)

st.markdown("")  # 여백

# ── 대화 기록 출력 (비교 뷰) ─────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        col_r, col_l = st.columns(2)
        with col_r:
            with st.container(border=True):
                _ra  = msg.get("rag_answer", "")
                _ur  = msg.get("used_rag", False)
                _nd  = msg.get("no_doc", False)
                if _nd:
                    st.warning(_ra)
                else:
                    st.markdown(_ra, unsafe_allow_html=False)
                if _ur:
                    st.markdown('<span class="tag-rag">📖 Self-RAG (문서 기반)</span>', unsafe_allow_html=True)
                elif _nd:
                    st.markdown('<span style="font-size:0.75rem;color:#6b7280">⚠️ 문서 내 관련 법 없음</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="tag-llm">⚡ LLM 직접 생성</span>', unsafe_allow_html=True)
        with col_l:
            with st.container(border=True):
                st.markdown(msg.get("llm_answer", ""), unsafe_allow_html=False)
                st.markdown('<span class="tag-llm">🤖 순수 LLM (RAG 없음)</span>', unsafe_allow_html=True)

# ── 입력 처리 ────────────────────────────────────────────────────
user_input = st.chat_input("사건 내용을 자세히 설명하거나 법률 질문을 입력하세요...")
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

if user_input:
    if not st.session_state.self_rag:
        st.warning("사이드바에서 법률 PDF를 업로드하고 DB를 구축하세요.")
    else:
        # 사용자 메시지 표시
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # 두 답변 동시 생성
        col_r, col_l = st.columns(2)

        with col_r:
            with st.container(border=True):
                with st.spinner("📖 Self-RAG 분석 중..."):
                    rag_result = st.session_state.self_rag.process_query(
                        user_input, st.session_state.memory
                    )
                rag_answer  = rag_result["answer"]
                used_rag    = rag_result["used_rag"]
                no_doc      = rag_result.get("no_doc", False)
                if no_doc:
                    st.warning(rag_answer)
                else:
                    st.markdown(rag_answer)
                if used_rag:
                    st.markdown('<span class="tag-rag">📖 Self-RAG (문서 기반)</span>', unsafe_allow_html=True)
                elif no_doc:
                    st.markdown('<span style="font-size:0.75rem;color:#6b7280">⚠️ 문서 내 관련 법 없음</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="tag-llm">⚡ LLM 직접 생성</span>', unsafe_allow_html=True)

        with col_l:
            with st.container(border=True):
                with st.spinner("🤖 순수 LLM 답변 생성 중..."):
                    llm_answer = get_llm_direct_answer(
                        user_input, st.session_state.llm_direct, st.session_state.memory
                    )
                st.markdown(llm_answer)
                st.markdown('<span class="tag-llm">🤖 순수 LLM (RAG 없음)</span>', unsafe_allow_html=True)

        # 세션 저장
        st.session_state.chat_history.append({
            "role":       "assistant",
            "content":    rag_answer,   # 하위 호환용
            "rag_answer": rag_answer,
            "llm_answer": llm_answer,
            "used_rag":   used_rag,
            "no_doc":     no_doc,
        })
        st.session_state.process_log = rag_result["log"]
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(rag_answer)
        st.rerun()

# ── 면책 고지 ────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚠️ 본 챗봇은 법률 정보 제공 목적이며, 법적 효력이 있는 법률 자문이 아닙니다.
실제 법적 분쟁에는 반드시 자격 있는 변호사와 상담하시기 바랍니다.
</div>
""", unsafe_allow_html=True)
