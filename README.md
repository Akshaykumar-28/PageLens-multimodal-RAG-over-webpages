# PageLens — MultiModal RAG with ColiVara and DeepSeek-Janus-Pro

Turn any live webpage into a set of visual embeddings you can chat with. PageLens
screenshots a page, indexes it visually (no OCR, no text chunking), and lets a
vision-language model answer questions by actually *looking* at the page —
then shows you exactly which screenshot it used to answer.

We use the following tools:
- **DeepSeek Janus-Pro** as the multimodal LLM.
- **[ColiVara](https://colivara.com/)** for SOTA visual document retrieval.
- **[Firecrawl](https://www.firecrawl.dev/i/api)** for full-page screenshotting.
- **Streamlit** for the web interface.

## Demo

A demo of the original project is available at `./video-demo.mp4`.

---

## What's new in this version

**Redesign** — a single vibrant-gradient design system (violet → blue → pink,
Space Grotesk + Inter + JetBrains Mono) applied uniformly across the sidebar,
hero, document cards, chat bubbles, and citation panels, defined once in
`static/style.css`.

**Extended: ingestion.** The sidebar now accepts multiple URLs at once (one
per line) and indexes them as a batch with per-URL progress and error
reporting, instead of one page at a time.

**Extended: chat citations.** Every answer now retrieves and shows its **top-3
source pages** (not just the single best match), each with a similarity-score
badge and a pulsing gradient "scan ring" — so you can see exactly what the
model looked at, and how confident the match was.

**New: document manager.** All indexed pages appear as cards in the sidebar
with a thumbnail, source URL, and index time. Toggle any subset of documents
on/off to scope your questions to specific pages, or remove a document
entirely (also deletes it from the ColiVara collection).

**New: chat history.** Start multiple named chat threads, search across all
of them, switch between them, and export any conversation (with its cited
sources) as a Markdown file.

---
## Setup and installations

**Setup Janus**:
```bash
git clone https://github.com/deepseek-ai/Janus.git
pip install -e ./Janus
```

**Get the API keys**:
- [ColiVara](https://colivara.com/) for SOTA document understanding and retrieval.
- [Firecrawl](https://www.firecrawl.dev/i/api) for web scraping.

Create a `.env` file and store them as follows:
```python
COLIVARA_API_KEY="<COLIVARA-API-KEY>"
FIRECRAWL_API_KEY="<FIRECRAWL-API-KEY>"
```

**Install dependencies**:
Ensure you have Python 3.11 or later installed.
```bash
pip install streamlit-pdf-viewer colivara-py streamlit fastembed flash-attn transformers fpdf2 firecrawl-py python-dotenv pillow requests
```

---

## Run the project

```bash
streamlit run app.py
```

Then, in the sidebar:
1. Paste one or more URLs (one per line) under **Add content** and click **Index pages**.
2. Toggle which documents are "in scope" under **Your documents**.
3. Ask questions in the chat box — each answer opens a **Sources** panel showing
   the page(s) it used and how strong the visual match was.
4. Use **Chat history** to start new threads, search past ones, or export a chat.

---

## 📬 Stay Updated with Our Newsletter!
**Get a FREE Data Science eBook** 📖 with 150+ essential lessons in Data Science when you subscribe to our newsletter! Stay in the loop with the latest tutorials, insights, and exclusive resources. [Subscribe now!](https://join.dailydoseofds.com)

[![Daily Dose of Data Science Newsletter](https://github.com/patchy631/ai-engineering/blob/main/resources/join_ddods.png)](https://join.dailydoseofds.com)

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
