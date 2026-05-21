"""
RAGAS를 이용한 법률 RAG 성능 평가
Ground Truth 12개 — 주요 법률 사건 유형 커버
"""

import os, sys
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ─── Ground Truth 12개 (법률 사건 유형) ───────────────────────────────────────
GROUND_TRUTH_DATA = [
    {
        "question": "교통사고 피해자로서 손해배상을 청구하려면 어떻게 해야 하나요?",
        "ground_truth": "교통사고 피해자는 가해자 및 가해자의 보험사에 손해배상을 청구할 수 있습니다. 치료비, 휴업손해, 위자료 등을 청구할 수 있으며, 합의가 되지 않을 경우 민사소송을 제기할 수 있습니다.",
    },
    {
        "question": "집주인이 전세 보증금을 돌려주지 않을 때 법적으로 어떻게 대응할 수 있나요?",
        "ground_truth": "임차인은 임대차보호법에 따라 임차권등기명령을 신청하거나, 내용증명을 보내고 보증금반환청구 소송을 제기할 수 있습니다. 또한 주택도시보증공사(HUG)의 전세보증금 반환보증 제도를 이용할 수 있습니다.",
    },
    {
        "question": "부당해고를 당했을 때 구제 방법은 무엇인가요?",
        "ground_truth": "부당해고를 당한 근로자는 해고 통보일로부터 3개월 이내에 노동위원회에 부당해고 구제신청을 할 수 있습니다. 구제가 인정되면 원직 복직 또는 해고 기간의 임금 상당액을 지급받을 수 있습니다.",
    },
    {
        "question": "온라인 사기를 당했을 때 고소 절차는 어떻게 되나요?",
        "ground_truth": "사기 피해를 당한 경우 경찰서나 검찰청에 고소장을 제출할 수 있습니다. 피해 증거(대화 내역, 이체 확인서 등)를 확보하고, 사이버범죄신고시스템(ECRM)을 통해 온라인 신고도 가능합니다.",
    },
    {
        "question": "폭행을 당했을 때 가해자를 고소하는 방법을 알려주세요.",
        "ground_truth": "폭행 피해자는 경찰서에 고소장을 제출하거나 112에 신고할 수 있습니다. 진단서, 사진, 목격자 진술 등 증거를 확보하는 것이 중요하며, 형사처벌과 별개로 민사상 손해배상도 청구할 수 있습니다.",
    },
    {
        "question": "계약 상대방이 계약을 이행하지 않을 때 법적 대응 방법은?",
        "ground_truth": "계약 불이행 시 내용증명을 통해 이행을 촉구하고, 이행이 되지 않으면 계약 해제 및 손해배상 청구 소송을 제기할 수 있습니다. 소액(3,000만 원 이하)의 경우 소액사건심판제도를 이용할 수 있습니다.",
    },
    {
        "question": "명예훼손으로 고소를 당했을 때 어떻게 대응해야 하나요?",
        "ground_truth": "명예훼손 고소를 당한 경우 사실 여부와 공익성을 검토해야 합니다. 형법상 사실 적시 명예훼손과 허위사실 적시 명예훼손은 처벌이 다르며, 변호사를 선임하여 위법성 조각 사유(공익 목적 등)를 주장할 수 있습니다.",
    },
    {
        "question": "이혼 시 재산 분할은 어떻게 이루어지나요?",
        "ground_truth": "이혼 시 재산 분할은 혼인 기간 중 공동으로 형성한 재산을 대상으로 하며, 기여도에 따라 분할됩니다. 협의 이혼의 경우 당사자 간 합의로, 재판상 이혼의 경우 법원이 결정합니다.",
    },
    {
        "question": "임금 체불 시 어떤 법적 조치를 취할 수 있나요?",
        "ground_truth": "임금이 체불된 경우 고용노동부에 진정을 제기하거나 사업장 관할 노동청에 신고할 수 있습니다. 사업주는 임금체불 시 3년 이하의 징역 또는 3천만 원 이하의 벌금에 처할 수 있습니다.",
    },
    {
        "question": "개인정보 유출 피해를 입었을 때 어떻게 대응하나요?",
        "ground_truth": "개인정보 유출 피해를 입은 경우 개인정보보호위원회에 신고하거나 한국인터넷진흥원(KISA)에 상담을 요청할 수 있습니다. 손해가 발생한 경우 손해배상 청구 소송도 가능합니다.",
    },
    {
        "question": "스토킹 피해를 당하고 있을 때 법적 보호를 받을 수 있나요?",
        "ground_truth": "스토킹처벌법에 따라 스토킹 행위자에게는 3년 이하의 징역 또는 3천만 원 이하의 벌금이 부과됩니다. 피해자는 경찰에 신고하여 긴급응급조치, 잠정조치, 피해자 보호명령을 신청할 수 있습니다.",
    },
    {
        "question": "소비자 피해를 당했을 때 환불이나 배상을 받을 수 있는 방법은?",
        "ground_truth": "소비자 피해 발생 시 한국소비자원에 분쟁조정을 신청하거나 소비자보호법에 따라 청약철회를 요청할 수 있습니다. 온라인 구매의 경우 배송 후 7일 이내 청약철회가 가능합니다.",
    },
]


def build_pipeline(pdf_path: str, api_key: str):
    loader = PyPDFLoader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    docs = loader.load_and_split(splitter)
    embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    vectordb = FAISS.from_documents(docs, embedding)
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=2000, temperature=0.1, api_key=api_key)
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="다음 법률 문서를 참고하여 질문에 답하세요.\n\n문서: {context}\n\n질문: {query}"
    )
    return vectordb, prompt | llm


def run_evaluation(pdf_path: str, api_key: str):
    print("RAG 파이프라인 구축 중...")
    vectordb, chain = build_pipeline(pdf_path, api_key)

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"\n{len(GROUND_TRUTH_DATA)}개 법률 질문 답변 생성 중...")
    for i, item in enumerate(GROUND_TRUTH_DATA):
        q, gt = item["question"], item["ground_truth"]
        docs = vectordb.similarity_search(q, k=4)
        ctx_texts = [d.page_content for d in docs]
        context_str = "\n\n".join(ctx_texts)
        response = chain.invoke({"query": q, "context": context_str})
        answer = response.content if hasattr(response, "content") else str(response)

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx_texts)
        ground_truths.append(gt)
        print(f"  [{i+1}/{len(GROUND_TRUTH_DATA)}] {q[:35]}...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nRAGAS 평가 실행 중...")
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

    print("\n" + "="*55)
    print("📊 법률 RAG RAGAS 평가 결과")
    print("="*55)
    print(f"Faithfulness  (충실도):      {result['faithfulness']:.4f}")
    print(f"Answer Relevancy (관련성):   {result['answer_relevancy']:.4f}")
    print(f"Context Precision (정밀도):  {result['context_precision']:.4f}")
    print(f"Context Recall  (재현율):    {result['context_recall']:.4f}")
    print("="*55)
    return result


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수를 설정하세요.")
        sys.exit(1)
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "법률문서.pdf"
    if not os.path.exists(pdf_path):
        print(f"PDF 파일 없음: {pdf_path}")
        print("사용법: python ragas_eval.py <pdf_경로>")
        sys.exit(1)
    run_evaluation(pdf_path, api_key)
