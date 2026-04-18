"""
멀티세션 RAG 챗봇 — Supabase 세션/벡터 저장, 스트리밍 답변, OpenAI 임베딩.
실행: streamlit run 7.MultiService/code/multi-session-ref.py (저장 위치 기준)
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# 경로 · 환경
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"chatbot_{datetime.now():%Y%m%d}.log"

MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
VECTOR_BATCH = 10
RETRIEVE_K = 10


def _setup_logging() -> None:
    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(fh)


_setup_logging()
_log = logging.getLogger(__name__)

_ENV_LOADED = False


def load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(REPO_ROOT / ".env")
    _ENV_LOADED = True


def env_status() -> tuple[bool, list[str]]:
    load_env()
    missing: list[str] = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("SUPABASE_URL"):
        missing.append("SUPABASE_URL")
    if not os.getenv("SUPABASE_ANON_KEY"):
        missing.append("SUPABASE_ANON_KEY")
    return (len(missing) == 0, missing)


@st.cache_resource
def get_supabase() -> Client | None:
    ok, miss = env_status()
    if not ok:
        return None
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_ANON_KEY"]
    return create_client(url, key)


def remove_separators(text: str) -> str:
    if not text:
        return text
    out = re.sub(r"~~[^~]*~~", "", text)
    out = re.sub(r"(?m)^\s*(---+|===+|___+)\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def format_session_label(row: dict[str, Any]) -> str:
    title = (row.get("title") or "").strip() or "제목 없음"
    sid = str(row["id"])
    return f"{title} ({sid[:8]})"


# ---------------------------------------------------------------------------
# Supabase — 세션 · 벡터
# ---------------------------------------------------------------------------
def fetch_sessions() -> list[dict[str, Any]]:
    cli = get_supabase()
    if not cli:
        return []
    try:
        res = (
            cli.table("sessions")
            .select("id,title,messages,created_at,updated_at")
            .order("updated_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as e:
        _log.warning("fetch_sessions: %s", e)
        return []


def insert_session_row(title: str, messages: list[dict[str, str]]) -> str | None:
    cli = get_supabase()
    if not cli:
        return None
    sid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    try:
        cli.table("sessions").insert(
            {
                "id": sid,
                "title": title,
                "messages": messages,
                "created_at": now,
                "updated_at": now,
            }
        ).execute()
        return sid
    except Exception as e:
        _log.warning("insert_session_row: %s", e)
        st.error(f"세션 생성 오류: {e}")
        return None


def update_session_messages(session_id: str, title: str | None, messages: list[dict[str, str]]) -> bool:
    cli = get_supabase()
    if not cli:
        return False
    payload: dict[str, Any] = {
        "messages": messages,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    if title is not None:
        payload["title"] = title
    try:
        cli.table("sessions").update(payload).eq("id", session_id).execute()
        return True
    except Exception as e:
        _log.warning("update_session_messages: %s", e)
        st.error(f"세션 저장 오류: {e}")
        return False


def load_session_by_id(session_id: str) -> dict[str, Any] | None:
    cli = get_supabase()
    if not cli:
        return None
    try:
        res = cli.table("sessions").select("*").eq("id", session_id).single().execute()
        return res.data
    except Exception as e:
        _log.warning("load_session_by_id: %s", e)
        return None


def delete_session_db(session_id: str) -> bool:
    cli = get_supabase()
    if not cli:
        return False
    try:
        cli.table("sessions").delete().eq("id", session_id).execute()
        return True
    except Exception as e:
        _log.warning("delete_session_db: %s", e)
        st.error(f"세션 삭제 오류: {e}")
        return False


def duplicate_session_with_vectors(
    source_session_id: str, new_title: str, messages: list[dict[str, str]]
) -> str | None:
    """세션 복제: INSERT 새 세션 + vector_documents 행 복사."""
    cli = get_supabase()
    if not cli:
        return None
    new_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    try:
        cli.table("sessions").insert(
            {
                "id": new_id,
                "title": new_title,
                "messages": messages,
                "created_at": now,
                "updated_at": now,
            }
        ).execute()
    except Exception as e:
        _log.warning("duplicate_session insert sessions: %s", e)
        st.error(f"세션 복제 오류: {e}")
        return None
    try:
        vecs = (
            cli.table("vector_documents")
            .select("content,embedding,file_name,metadata")
            .eq("session_id", source_session_id)
            .execute()
        )
        rows = vecs.data or []
        batch: list[dict[str, Any]] = []
        for r in rows:
            batch.append(
                {
                    "session_id": new_id,
                    "content": r["content"],
                    "embedding": r["embedding"],
                    "file_name": r["file_name"],
                    "metadata": r.get("metadata") or {},
                }
            )
            if len(batch) >= VECTOR_BATCH:
                cli.table("vector_documents").insert(batch).execute()
                batch = []
        if batch:
            cli.table("vector_documents").insert(batch).execute()
    except Exception as e:
        _log.warning("duplicate_session vectors: %s", e)
        st.warning(f"벡터 복제 중 오류(세션 행은 생성됨): {e}")
    return new_id


def insert_vectors_for_session(
    session_id: str, file_name: str, texts: list[str], embeddings: OpenAIEmbeddings
) -> None:
    cli = get_supabase()
    if not cli or not texts:
        return
    vecs = embeddings.embed_documents(texts)
    for i in range(0, len(texts), VECTOR_BATCH):
        chunk_texts = texts[i : i + VECTOR_BATCH]
        chunk_vecs = vecs[i : i + VECTOR_BATCH]
        rows = []
        for j, (t, v) in enumerate(zip(chunk_texts, chunk_vecs)):
            rows.append(
                {
                    "session_id": session_id,
                    "content": t,
                    "embedding": v,
                    "file_name": file_name,
                    "metadata": {"chunk_index": i + j},
                }
            )
        try:
            cli.table("vector_documents").insert(rows).execute()
        except Exception as e:
            _log.warning("insert_vectors: %s", e)
            st.error(f"벡터 저장 오류: {e}")
            raise


def retrieve_chunks_rpc(
    session_id: str, query: str, embeddings: OpenAIEmbeddings
) -> list[Document]:
    cli = get_supabase()
    if not cli:
        return []
    try:
        q_emb = embeddings.embed_query(query)
        if len(q_emb) != EMBED_DIM:
            q_emb = q_emb[:EMBED_DIM]
    except Exception as e:
        _log.warning("embed_query: %s", e)
        return []

    try:
        res = (
            cli.rpc(
                "match_vector_documents",
                {
                    "query_embedding": q_emb,
                    "match_count": RETRIEVE_K,
                    "filter_session_id": session_id,
                },
            )
            .execute()
        )
        out: list[Document] = []
        for row in res.data or []:
            meta = {"file_name": row.get("file_name"), "similarity": row.get("similarity")}
            out.append(Document(page_content=row.get("content") or "", metadata=meta))
        return out
    except Exception as e:
        _log.warning("retrieve_chunks_rpc: %s", e)
        return _retrieve_chunks_fallback(session_id, query, q_emb, cli)


def _retrieve_chunks_fallback(
    session_id: str, query: str, query_emb: list[float], cli: Client
) -> list[Document]:
    try:
        res = (
            cli.table("vector_documents")
            .select("id,content,embedding,file_name,metadata")
            .eq("session_id", session_id)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        q = np.array(query_emb, dtype=np.float64)
        qn = np.linalg.norm(q) or 1.0
        for r in rows:
            emb = r.get("embedding")
            if isinstance(emb, str):
                try:
                    emb = json.loads(emb)
                except json.JSONDecodeError:
                    continue
            if not emb:
                continue
            v = np.array(emb, dtype=np.float64)
            vn = np.linalg.norm(v) or 1.0
            sim = float(np.dot(q, v) / (qn * vn))
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        docs: list[Document] = []
        for sim, r in scored[:RETRIEVE_K]:
            docs.append(
                Document(
                    page_content=r.get("content") or "",
                    metadata={"file_name": r.get("file_name"), "similarity": sim},
                )
            )
        return docs
    except Exception as e:
        _log.warning("_retrieve_chunks_fallback: %s", e)
        return []


def list_vector_filenames(session_id: str) -> list[str]:
    cli = get_supabase()
    if not cli:
        return []
    try:
        res = (
            cli.table("vector_documents")
            .select("file_name")
            .eq("session_id", session_id)
            .execute()
        )
        names = sorted({(r.get("file_name") or "").strip() for r in (res.data or []) if r.get("file_name")})
        return names
    except Exception as e:
        _log.warning("list_vector_filenames: %s", e)
        return []


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def get_llm(streaming: bool = True) -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, temperature=0.7, streaming=streaming)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)


def generate_session_title(first_q: str, first_a: str) -> str:
    llm = get_llm(streaming=False)
    prompt = (
        "다음은 사용자의 첫 질문과 어시스턴트의 첫 답변입니다. "
        "25자 이내의 한국어 세션 제목만 한 줄로 출력하세요. 따옴표나 부가 설명 없이 제목만.\n\n"
        f"질문: {first_q}\n\n답변: {first_a[:800]}"
    )
    try:
        r = llm.invoke([HumanMessage(content=prompt)])
        t = (r.content or "").strip().split("\n")[0].strip()
        return t[:80] if t else "새 대화"
    except Exception as e:
        _log.warning("generate_session_title: %s", e)
        return "새 대화"


def generate_followup_questions(context_snippet: str) -> str:
    llm = get_llm(streaming=False)
    prompt = (
        "아래 맥락을 참고하여 사용자가 이어서 물어보면 좋은 질문을 한국어로 정확히 3개만 번호 목록으로 작성하세요.\n"
        "각 줄은 '1. 질문' 형식입니다.\n\n맥락:\n" + context_snippet[:3000]
    )
    try:
        r = llm.invoke([HumanMessage(content=prompt)])
        return (r.content or "").strip()
    except Exception as e:
        _log.warning("generate_followup_questions: %s", e)
        return "1. 이 주제를 더 설명해 주실 수 있나요?\n2. 관련된 예시가 있나요?\n3. 추가로 확인할 점이 있나요?"


# ---------------------------------------------------------------------------
# Streamlit state
# ---------------------------------------------------------------------------
def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "processed_pdf_names" not in st.session_state:
        st.session_state.processed_pdf_names = []
    if "session_select_label" not in st.session_state:
        st.session_state.session_select_label = None
    if "show_vectordb" not in st.session_state:
        st.session_state.show_vectordb = False


def messages_to_lc(history: list[dict[str, str]]) -> list[Any]:
    out: list[Any] = []
    for m in history[-50:]:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        else:
            out.append(AIMessage(content=m["content"]))
    return out


def apply_session_from_row(row: dict[str, Any]) -> None:
    raw = row.get("messages")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    if not isinstance(raw, list):
        raw = []
    st.session_state.messages = [{"role": x["role"], "content": x["content"]} for x in raw if "role" in x and "content" in x]
    sid = str(row["id"])
    st.session_state.session_id = sid
    st.session_state.processed_pdf_names = list_vector_filenames(sid)


def autosave_current_session(title: str | None = None) -> None:
    sid = st.session_state.session_id
    if not sid or not get_supabase():
        return
    update_session_messages(sid, title, st.session_state.messages)


# ---------------------------------------------------------------------------
# PDF 처리
# ---------------------------------------------------------------------------
def process_pdfs(uploaded_files: list[Any], session_id: str, embeddings: OpenAIEmbeddings) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for uf in uploaded_files:
        name = uf.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getvalue())
            path = tmp.name
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        splits = splitter.split_documents(docs)
        texts = [d.page_content for d in splits]
        if not texts:
            continue
        insert_vectors_for_session(session_id, name, texts, embeddings)
        if name not in st.session_state.processed_pdf_names:
            st.session_state.processed_pdf_names.append(name)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
<style>
    h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
    h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
    h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
    div[data-testid="stChatMessage"] { padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; }
    button[kind="primary"] { background-color: #ff69b4 !important; border-color: #ff69b4 !important; color: #fff !important; }
    .stButton button { background-color: #ff69b4 !important; color: #fff !important; border: none; }
</style>
""",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    logo_path = REPO_ROOT / "logo.png"
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if logo_path.exists():
            st.image(str(logo_path), width=180)
        else:
            st.markdown("### 📚")
    with col2:
        st.markdown(
            """
<div style="text-align:center;width:100%;margin-bottom:0.5rem;">
<span style="color:#1f77b4;font-size:4rem !important;font-weight:700;">멀티세션 RAG</span>
<span style="color:#ffd700;font-size:4rem !important;font-weight:700;"> 챗봇</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with col3:
        st.empty()


def on_session_select_change() -> None:
    label = st.session_state.get("_session_select")
    if not label or label == "(선택 없음)":
        return
    mapping = st.session_state.get("_label_to_id") or {}
    sid = mapping.get(label)
    if not sid:
        return
    row = load_session_by_id(sid)
    if row:
        apply_session_from_row(row)


def main() -> None:
    st.set_page_config(page_title="멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
    load_env()
    init_state()
    inject_css()
    render_header()

    ok_keys, missing = env_status()
    if not ok_keys:
        st.warning("다음 환경 변수가 `.env`에 필요합니다: " + ", ".join(missing))
        st.stop()

    supabase_ok = get_supabase() is not None
    if not supabase_ok:
        st.error("Supabase 클라이언트를 만들 수 없습니다.")
        st.stop()

    embeddings = get_embeddings()
    llm_stream = get_llm(streaming=True)

    sessions = fetch_sessions()
    label_to_id: dict[str, str] = {}
    for row in sessions:
        label_to_id[format_session_label(row)] = str(row["id"])
    st.session_state._label_to_id = label_to_id
    labels = list(label_to_id.keys())

    with st.sidebar:
        st.markdown("### LLM 모델")
        st.text(f"고정: {MODEL_NAME}")

        st.markdown("### 세션 관리")
        select_options = ["(선택 없음)"] + labels
        default_idx = 0
        cur = st.session_state.session_id
        if cur:
            for lab, sid in label_to_id.items():
                if sid == cur:
                    sel_label = lab
                    if sel_label in select_options:
                        default_idx = select_options.index(sel_label)
                    break

        chosen = st.selectbox(
            "저장된 세션",
            options=select_options,
            index=default_idx,
            key="_session_select",
            on_change=on_session_select_change,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("세션로드"):
                if chosen and chosen != "(선택 없음)":
                    sid = label_to_id.get(chosen)
                    if sid:
                        row = load_session_by_id(sid)
                        if row:
                            apply_session_from_row(row)
                            st.success("세션을 불러왔습니다.")
                            st.rerun()
        with c2:
            if st.button("세션저장"):
                msgs = st.session_state.messages
                if len(msgs) < 2:
                    st.warning("첫 질문과 답변이 있어야 저장할 수 있습니다.")
                else:
                    u0 = next((m["content"] for m in msgs if m["role"] == "user"), "")
                    a0 = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                    a0_core = (a0 or "").split("### 💡")[0].strip()
                    title = generate_session_title(u0, a0_core)
                    sid_src = st.session_state.session_id
                    if not sid_src:
                        st.warning("현재 세션 ID가 없습니다. 대화를 한 번 진행해 주세요.")
                    else:
                        new_id = duplicate_session_with_vectors(sid_src, title, msgs)
                        if new_id:
                            st.success(f"새 세션으로 저장했습니다: {title}")
                            st.rerun()

        c3, c4 = st.columns(2)
        with c3:
            if st.button("세션삭제"):
                sid = st.session_state.session_id
                if not sid:
                    st.warning("삭제할 세션이 없습니다.")
                elif delete_session_db(sid):
                    st.session_state.messages = []
                    st.session_state.session_id = None
                    st.session_state.processed_pdf_names = []
                    st.success("세션이 삭제되었습니다.")
                    st.rerun()
        with c4:
            if st.button("화면초기화"):
                st.session_state.messages = []
                st.session_state.session_id = None
                st.session_state.processed_pdf_names = []
                st.session_state.show_vectordb = False
                st.success("화면을 초기화했습니다.")
                st.rerun()

        if st.button("vectordb"):
            st.session_state.show_vectordb = not st.session_state.show_vectordb

        st.markdown("### PDF (RAG)")
        files = st.file_uploader("PDF 업로드", type=["pdf"], accept_multiple_files=True)
        if st.button("파일 처리하기") and files:
            sid = st.session_state.session_id
            if not sid:
                sid = insert_session_row("임시 세션", [])
                if sid:
                    st.session_state.session_id = sid
            if st.session_state.session_id:
                with st.spinner("PDF 처리 및 벡터 저장 중…"):
                    process_pdfs(list(files), st.session_state.session_id, embeddings)
                autosave_current_session()
                st.success("파일 처리 및 세션 자동 저장을 완료했습니다.")
                st.rerun()

        st.markdown("### 현재 설정")
        st.text(
            f"모델: {MODEL_NAME}\n"
            f"세션 ID: {st.session_state.session_id or '없음'}\n"
            f"처리된 PDF 파일: {len(st.session_state.processed_pdf_names)}개\n"
            f"대화 메시지: {len(st.session_state.messages)}개"
        )

    if st.session_state.show_vectordb:
        sid = st.session_state.session_id
        if sid:
            names = list_vector_filenames(sid)
            st.info("현재 세션 벡터 DB 파일명:\n" + ("\n".join(f"- {n}" for n in names) if names else "(없음)"))
        else:
            st.info("활성 세션이 없습니다. 대화를 시작하거나 PDF를 처리하세요.")

    if prompt := st.chat_input("메시지를 입력하세요…"):
        if not st.session_state.session_id:
            sid_new = insert_session_row("새 대화", [])
            if sid_new:
                st.session_state.session_id = sid_new
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(remove_separators(m["content"]))

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        sid = st.session_state.session_id
        prompt = st.session_state.messages[-1]["content"]
        ctx_docs = retrieve_chunks_rpc(sid or "", prompt, embeddings) if sid else []
        context_text = "\n\n".join(d.page_content for d in ctx_docs[:RETRIEVE_K])

        sys_parts = [
            "당신은 친절한 한국어 어시스턴트입니다.",
            "답변은 # ## ### 마크다운 헤딩으로 구조화하고 존댓말을 쓰세요.",
            "구분선(---, ===, ___)과 취소선(~~)은 쓰지 마세요.",
            "참조·출처 문구는 넣지 마세요.",
        ]
        if context_text.strip():
            sys_parts.append("다음은 참고 문서 발췌입니다:\n" + context_text)
        else:
            sys_parts.append("참고 문서가 없으면 일반 지식으로 간결히 답하세요.")

        sys_msg = SystemMessage(content="\n\n".join(sys_parts))
        hist = messages_to_lc(st.session_state.messages)

        full = ""
        full_with_follow = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                for chunk in llm_stream.stream([sys_msg] + hist):
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        full += chunk.content
                        placeholder.markdown(remove_separators(full))
            except Exception as e:
                _log.warning("stream: %s", e)
                full = f"오류가 발생했습니다: {e}"
                placeholder.markdown(remove_separators(full))

            try:
                fu = generate_followup_questions(full + "\n" + prompt)
                follow_block = "\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n" + fu
                full_with_follow = full + follow_block
            except Exception as e:
                _log.warning("followup: %s", e)
                full_with_follow = full + "\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n1. 이 내용을 더 설명해 주실 수 있나요?\n2. 관련 예시가 있나요?\n3. 추가로 알아두면 좋은 점이 있나요?"
            placeholder.markdown(remove_separators(full_with_follow))

        st.session_state.messages.append({"role": "assistant", "content": full_with_follow})

        autosave_current_session()

        msgs = st.session_state.messages
        if sid and len([m for m in msgs if m["role"] == "user"]) == 1:
            u0 = next((m["content"] for m in msgs if m["role"] == "user"), "")
            a0 = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if u0 and a0:
                new_title = generate_session_title(u0, a0.split("### 💡")[0].strip())
                autosave_current_session(title=new_title)

        st.rerun()


if __name__ == "__main__":
    main()
