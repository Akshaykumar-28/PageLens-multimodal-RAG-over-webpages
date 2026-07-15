"""
Retrieval and generation layer for PageLens.

Two things were extended here versus the original single-image RAG:
  1. Retriever.search() now returns *ranked, scored* hits (SourceHit) instead
     of silently grabbing result[0] — this is what makes citations possible.
  2. RAG.query() can restrict retrieval to a subset of indexed documents
     (`active_documents`), and now feeds the model every retrieved page
     instead of just one, then hands the caller back the sources it used.
"""

import base64
import io
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image
from colivara_py import ColiVara
from transformers import AutoModelForCausalLM

from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images


def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


@dataclass
class SourceHit:
    """One retrieved page, returned alongside every answer so the UI can cite it."""

    document_name: str
    page_number: int
    normalized_score: float
    raw_score: float
    image_path: str


class Retriever:
    """Wraps a ColiVara collection and turns a text query into ranked, scored page hits."""

    def __init__(self, rag_client: ColiVara, collection_name: str):
        self.rag_client = rag_client
        self.collection_name = collection_name

    def search(
        self,
        query: str,
        top_k: int = 3,
        active_documents: Optional[List[str]] = None,
    ) -> List[SourceHit]:
        """
        Retrieve the top_k most visually-relevant pages for `query`.

        active_documents: if given, only pages whose document_name is in this
        list are kept. ColiVara's query_filter only pins a single value, so
        for an arbitrary multi-document selection we over-fetch and filter
        client-side, then trim back down to top_k.
        """
        fetch_k = top_k * 5 if active_documents else top_k
        response = self.rag_client.search(
            query=query,
            collection_name=self.collection_name,
            top_k=fetch_k,
        )

        results = response.results
        if active_documents:
            filtered = [r for r in results if r.document_name in active_documents]
            results = filtered if filtered else results

        results = results[:top_k]

        hits = []
        for i, r in enumerate(results):
            image_bytes = base64.b64decode(r.img_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_path = f"source_hit_{i}.jpeg"
            image.save(image_path)
            hits.append(
                SourceHit(
                    document_name=r.document_name,
                    page_number=r.page_number,
                    normalized_score=float(r.normalized_score),
                    raw_score=float(r.raw_score),
                    image_path=image_path,
                )
            )
        return hits


class RAG:
    def __init__(self, retriever: Retriever, llm_name: str = "deepseek-ai/Janus-Pro-1B"):
        self.llm_name = llm_name
        self.retriever = retriever
        self._setup_llm()

    def _setup_llm(self):
        self.vl_chat_processor = VLChatProcessor.from_pretrained(
            self.llm_name, cache_dir="./Janus/hf_cache"
        )
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt = AutoModelForCausalLM.from_pretrained(
            self.llm_name, trust_remote_code=True, cache_dir="./Janus/hf_cache"
        ).to(torch.bfloat16).eval()

    def query(
        self,
        query: str,
        top_k: int = 3,
        active_documents: Optional[List[str]] = None,
    ):
        """
        Answers `query` using the top_k most relevant indexed pages.

        Returns (answer_text, List[SourceHit]) — the sources are what the UI
        renders in the "Sources" panel next to the answer.
        """
        sources = self.retriever.search(query, top_k=top_k, active_documents=active_documents)

        if not sources:
            return (
                "I don't have any indexed pages to look at yet — add a URL "
                "in the sidebar first, or turn on a document to search within.",
                [],
            )

        image_placeholders = " ".join(["<image_placeholder>"] * len(sources))
        qa_prompt_tmpl_str = f"""The user has asked the following question:

                        ---------------------

                        Query: {query}

                        ---------------------

                        You have been shown {len(sources)} page screenshot(s)
                        relevant to this question. Study them thoroughly and
                        extract all relevant information that will help you
                        answer the query, citing which page supports which
                        part of your answer where useful.

                        ---------------------
                        """

        conversation = [
            {
                "role": "User",
                "content": f"{image_placeholders} \n {qa_prompt_tmpl_str}",
                "images": [s.image_path for s in sources],
            },
            {"role": "Assistant", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

        return answer, sources
