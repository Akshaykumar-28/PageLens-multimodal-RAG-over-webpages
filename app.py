# PageLens — Multimodal RAG over live webpages, powered by ColiVara + Janus-Pro
# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
import gc
import io
import math
import os
import re
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from fpdf import FPDF
from PIL import Image
from streamlit_pdf_viewer import pdf_viewer

from colivara_py import ColiVara
from rag_code import RAG, Retriever

load_dotenv()

APP_DIR = Path(__file__).parent


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def load_css():
    css_path = APP_DIR / "static" / "style.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def slugify(url: str) -> str:
    parsed = urlparse(url)
    raw = f"{parsed.netloc}{parsed.path}".strip("/") or parsed.netloc or "page"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower()
    return (slug[:60] or "page")


def screenshot_to_pdf(screenshot_url: str, slug: str, work_dir: Path):
    """
    Download a full-page screenshot, slice it into a paginated PDF (what
    ColiVara indexes), and also return a small thumbnail for the document
    card in the sidebar.
    """
    response = requests.get(screenshot_url, timeout=60)
    response.raise_for_status()

    raw_path = work_dir / f"{slug}_raw.png"
    raw_path.write_bytes(response.content)

    image = Image.open(raw_path)
    width, height = image.size

    thumb = image.copy()
    thumb.thumbnail((360, 360))
    thumb_buffer = io.BytesIO()
    thumb.convert("RGB").save(thumb_buffer, format="JPEG", quality=80)
    thumb_b64 = __import__("base64").b64encode(thumb_buffer.getvalue()).decode("utf-8")

    n_slices = 10
    slice_height = math.ceil(height / n_slices)
    pdf = FPDF(unit="pt", format=[width, slice_height])
    pdf.set_auto_page_break(auto=False)

    for i in range(n_slices):
        top = i * slice_height
        bottom = min((i + 1) * slice_height, height)
        if top >= bottom:
            break
        slice_img = image.crop((0, top, width, bottom))
        slice_path = work_dir / f"{slug}_slice_{i}.png"
        slice_img.save(slice_path)
        pdf.add_page()
        pdf.image(str(slice_path), x=0, y=0, w=width, h=bottom - top)
        slice_path.unlink(missing_ok=True)

    pdf_path = work_dir / f"{slug}.pdf"
    pdf.output(str(pdf_path))
    raw_path.unlink(missing_ok=True)
    return str(pdf_path), thumb_b64


def export_chat_markdown(chat: dict) -> str:
    lines = [f"# {chat['title']}", "", f"_Exported {datetime.now().strftime('%b %d, %Y %H:%M')}_", ""]
    for m in chat["messages"]:
        speaker = "**You**" if m["role"] == "user" else "**PageLens**"
        lines.append(f"{speaker}: {m['content']}")
        if m.get("sources"):
            src_line = ", ".join(
                f"{s['document_title']} (p.{s['page_number']}, {s['normalized_score']*100:.0f}%)"
                for s in m["sources"]
            )
            lines.append(f"> Sources: {src_line}")
        lines.append("")
    return "\n".join(lines)


def new_chat_session(doc_names):
    chat_id = str(uuid.uuid4())
    return chat_id, {
        "id": chat_id,
        "title": "New chat",
        "created_at": datetime.now(),
        "doc_names": set(doc_names),
        "messages": [],
    }


# --------------------------------------------------------------------------
# session state
# --------------------------------------------------------------------------
st.set_page_config(page_title="PageLens", page_icon="🔮", layout="wide")
load_css()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.collection_name = "webpage_collection_" + str(st.session_state.id)
    st.session_state.documents = []          # list[{name, url, title, thumb_b64, indexed_at}]
    st.session_state.active_doc_names = set()
    st.session_state.chat_sessions = {}
    chat_id, chat = new_chat_session(st.session_state.active_doc_names)
    st.session_state.chat_sessions[chat_id] = chat
    st.session_state.current_chat_id = chat_id

session_id = st.session_state.id


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔮 PageLens")
    st.caption("Turn any webpage into something you can ask questions about.")

    with st.expander("➕ Add content", expanded=not st.session_state.documents):
        url_text = st.text_area(
            "Paste one or more URLs (one per line)",
            placeholder="https://example.com\nhttps://docs.example.com/pricing",
            height=100,
            key="url_text_area",
        )
        start_rag = st.button("🚀 Index pages", use_container_width=True, type="primary")

        if start_rag:
            urls = list(dict.fromkeys(u.strip() for u in url_text.splitlines() if u.strip()))
            if not urls:
                st.warning("Add at least one URL first.")
            else:
                try:
                    firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
                    rag_client = st.session_state.get("rag_client") or ColiVara(
                        api_key=os.getenv("COLIVARA_API_KEY")
                    )
                    st.session_state.rag_client = rag_client

                    if not st.session_state.get("collection_created"):
                        rag_client.create_collection(
                            name=st.session_state.collection_name,
                            metadata={"description": "Ingested webpages"},
                        )
                        st.session_state.collection_created = True

                    work_dir = Path(tempfile.mkdtemp(prefix="pagelens_"))
                    progress = st.progress(0.0, text="Starting…")
                    successes, failures = 0, []

                    for i, url in enumerate(urls):
                        try:
                            progress.progress(i / len(urls), text=f"Scraping {url}")
                            scrape_result = firecrawl_app.scrape_url(
                                url, params={"formats": ["screenshot@fullPage"], "waitFor": 10000}
                            )

                            slug = slugify(url) + "-" + uuid.uuid4().hex[:6]

                            progress.progress((i + 0.5) / len(urls), text=f"Building index for {url}")
                            pdf_path, thumb_b64 = screenshot_to_pdf(
                                scrape_result["screenshot"], slug, work_dir
                            )

                            title = None
                            metadata = scrape_result.get("metadata") if isinstance(scrape_result, dict) else None
                            if metadata:
                                title = metadata.get("title")
                            title = title or urlparse(url).netloc or url

                            rag_client.upsert_document(
                                name=slug,
                                collection_name=st.session_state.collection_name,
                                document_path=pdf_path,
                                metadata={"source_url": url},
                                wait=True,
                            )

                            st.session_state.documents.append(
                                {
                                    "name": slug,
                                    "url": url,
                                    "title": title,
                                    "thumb_b64": thumb_b64,
                                    "indexed_at": datetime.now().strftime("%b %d, %H:%M"),
                                }
                            )
                            st.session_state.active_doc_names.add(slug)
                            successes += 1
                        except Exception as e:
                            failures.append((url, str(e)))

                    progress.progress(1.0, text="Done")
                    time.sleep(0.4)
                    progress.empty()

                    if successes:
                        retriever = Retriever(
                            rag_client=rag_client, collection_name=st.session_state.collection_name
                        )
                        st.session_state.query_engine = RAG(retriever=retriever)
                        st.success(f"Indexed {successes} page(s). Ready to chat!")
                    for url, err in failures:
                        st.error(f"Failed on {url}: {err}")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # ---- document manager ----
    if st.session_state.documents:
        st.markdown("#### 📚 Your documents")
        doc_filter = st.text_input(
            "Filter documents", placeholder="Search documents…",
            label_visibility="collapsed", key="doc_filter",
        )
        for doc in list(st.session_state.documents):
            hay = f"{doc['title']} {doc['url']}".lower()
            if doc_filter and doc_filter.lower() not in hay:
                continue
            st.markdown('<div class="doc-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown(
                    f'<img class="doc-thumb" src="data:image/jpeg;base64,{doc["thumb_b64"]}" />',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(f"**{doc['title']}**")
                st.caption(doc["url"])
                st.caption(f"Indexed {doc['indexed_at']}")
            b1, b2 = st.columns(2)
            with b1:
                active = st.checkbox(
                    "Include in chat",
                    value=doc["name"] in st.session_state.active_doc_names,
                    key=f"active_{doc['name']}",
                )
                if active:
                    st.session_state.active_doc_names.add(doc["name"])
                else:
                    st.session_state.active_doc_names.discard(doc["name"])
            with b2:
                if st.button("🗑 Remove", key=f"del_{doc['name']}", use_container_width=True):
                    try:
                        if st.session_state.get("rag_client"):
                            st.session_state.rag_client.delete_document(
                                doc["name"], st.session_state.collection_name
                            )
                    except Exception as e:
                        st.warning(f"Couldn't remove remotely: {e}")
                    st.session_state.documents = [
                        d for d in st.session_state.documents if d["name"] != doc["name"]
                    ]
                    st.session_state.active_doc_names.discard(doc["name"])
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- chat history ----
    st.markdown("#### 🕘 Chat history")
    if st.button("＋ New chat", use_container_width=True):
        chat_id, chat = new_chat_session(st.session_state.active_doc_names)
        st.session_state.chat_sessions[chat_id] = chat
        st.session_state.current_chat_id = chat_id
        st.rerun()

    history_search = st.text_input(
        "Search chat history", placeholder="Search past conversations…",
        label_visibility="collapsed", key="history_search",
    )

    ordered_chats = sorted(
        st.session_state.chat_sessions.items(), key=lambda kv: kv[1]["created_at"], reverse=True
    )
    for chat_id, chat in ordered_chats:
        if history_search:
            haystack = chat["title"] + " " + " ".join(m["content"] for m in chat["messages"])
            if history_search.lower() not in haystack.lower():
                continue
        is_active = chat_id == st.session_state.current_chat_id
        label = f"{'🟣 ' if is_active else '⚪ '}{chat['title']}"
        if st.button(label, key=f"switch_{chat_id}", use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    current_chat = st.session_state.chat_sessions[st.session_state.current_chat_id]
    if current_chat["messages"]:
        st.download_button(
            "⬇ Export this chat",
            data=export_chat_markdown(current_chat),
            file_name=f"{current_chat['title'][:40] or 'chat'}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# main area
# --------------------------------------------------------------------------
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">MULTIMODAL RAG · COLIVARA + JANUS-PRO</div>
          <h1><span class="gradient-text">PageLens</span></h1>
          <p class="subtitle">See webpages the way a vision model does — index full-page
          screenshots, then ask anything and get the exact page it looked at.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    if st.button("Clear ↺", help="Clear this chat's messages"):
        current_chat["messages"] = []
        gc.collect()
        st.rerun()

current_chat = st.session_state.chat_sessions[st.session_state.current_chat_id]

if not st.session_state.documents:
    st.markdown(
        """
        <div class="empty-state">
          <h3>Nothing indexed yet</h3>
          <p>Paste a URL or two in the sidebar and hit <b>Index pages</b> to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                cols = st.columns(len(msg["sources"]))
                for col, src in zip(cols, msg["sources"]):
                    with col:
                        st.markdown('<div class="source-card">', unsafe_allow_html=True)
                        if os.path.exists(src["image_path"]):
                            st.image(src["image_path"])
                        st.markdown(
                            f'<span class="score-badge">{src["normalized_score"]*100:.0f}% match</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"{src['document_title']} · page {src['page_number']}")
                        st.markdown("</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask anything about your indexed pages…"):
    if "query_engine" not in st.session_state:
        st.error("Index at least one page first (see sidebar).")
    else:
        current_chat["messages"].append({"role": "user", "content": prompt})
        if current_chat["title"] == "New chat":
            current_chat["title"] = prompt[:40] + ("…" if len(prompt) > 40 else "")

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            active = st.session_state.active_doc_names or None
            with st.spinner("Reading the pages…"):
                answer, sources = st.session_state.query_engine.query(
                    prompt, top_k=3, active_documents=list(active) if active else None
                )

            placeholder = st.empty()
            rendered = ""
            for ch in answer:
                rendered += ch
                placeholder.markdown(rendered + "▌")
                time.sleep(0.003)
            placeholder.markdown(rendered)

            source_payload = []
            if sources:
                with st.expander(f"📎 Sources ({len(sources)})"):
                    cols = st.columns(len(sources))
                    for col, src in zip(cols, sources):
                        doc = next(
                            (d for d in st.session_state.documents if d["name"] == src.document_name),
                            None,
                        )
                        title = doc["title"] if doc else src.document_name
                        with col:
                            st.markdown('<div class="source-card">', unsafe_allow_html=True)
                            if os.path.exists(src.image_path):
                                st.image(src.image_path)
                            st.markdown(
                                f'<span class="score-badge">{src.normalized_score*100:.0f}% match</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"{title} · page {src.page_number}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        source_payload.append(
                            {
                                "image_path": src.image_path,
                                "normalized_score": src.normalized_score,
                                "page_number": src.page_number,
                                "document_title": title,
                            }
                        )

        current_chat["messages"].append(
            {"role": "assistant", "content": answer, "sources": source_payload}
        )
        st.rerun()
