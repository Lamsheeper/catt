"""Retrieval baselines: BM25 and RepSim.

Both read the same text the gradient methods differentiate: a training
document's `text`, and a query's prompt + completion. Both score the first 4
queries per fact, like every gradient method (the published archive scored
retrieval on all 5).

    bm25    Okapi BM25 (k1 1.5, b 0.75, epsilon 0.25) over lowercased terms that
            keep added tokens such as <B01> whole: `<[^<>\\s]+>|[A-Za-z0-9]+`.
            Documents are indexed untruncated.
    repsim  cosine similarity of the fine-tuned model's final-layer hidden state
            (after the final norm) at the last real token; base + adapter merged,
            bf16, batches of 16, truncated to 512 / 128 tokens.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

BM25_K1, BM25_B, BM25_EPSILON = 1.5, 0.75, 0.25
TERM_RE = re.compile(r"<[^<>\s]+>|[A-Za-z0-9]+")
ENCODE_BATCH = 16


def terms(text: str) -> List[str]:
    return TERM_RE.findall(text.lower())


def document_text(doc: Dict) -> str:
    return str(doc.get("text", ""))


def query_text(doc: Dict) -> str:
    prompt = str(doc.get("prompt", doc.get("query", "")))
    return f"{prompt} {doc.get('completion', '')}".strip()


def bm25_scores(train_docs: List[Dict], query_docs: List[Dict]) -> np.ndarray:
    """[n_query, n_train] float32."""
    from rank_bm25 import BM25Okapi

    index = BM25Okapi([terms(document_text(d)) for d in train_docs],
                      k1=BM25_K1, b=BM25_B, epsilon=BM25_EPSILON)
    scores = np.zeros((len(query_docs), len(train_docs)), dtype=np.float32)
    for i, q in enumerate(query_docs):
        scores[i] = index.get_scores(terms(query_text(q)))
    return scores


def encode_last(texts: List[str], model, tokenizer, device: str, batch_size: int,
                max_length: int) -> np.ndarray:
    """L2-normalised hidden states at each text's last real token, [n, dim]."""
    import torch

    out = []
    for lo in range(0, len(texts), batch_size):
        enc = tokenizer(texts[lo:lo + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"]
        pos = (mask * torch.arange(hidden.shape[1], device=hidden.device)).argmax(1)
        vec = hidden[torch.arange(hidden.shape[0], device=hidden.device), pos]
        out.append(torch.nn.functional.normalize(vec.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def repsim_scores(train_docs: List[Dict], query_docs: List[Dict], base: str,
                  adapter: Optional[str], device: str,
                  max_train_length: int = 512, max_query_length: int = 128) -> np.ndarray:
    """[n_query, n_train] float32 cosine similarities."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # AutoModelForCausalLM, not AutoModel: the stock Qwen3.5 config names the
    # multimodal wrapper, which AutoModel would build without the text weights.
    lm = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, adapter, torch_dtype=dtype).merge_and_unload()
    model = lm.get_decoder().to(device).eval()
    docs = encode_last([document_text(d) for d in train_docs], model, tokenizer, device,
                       ENCODE_BATCH, max_train_length)
    queries = encode_last([query_text(q) for q in query_docs], model, tokenizer, device,
                          ENCODE_BATCH, max_query_length)
    del model, lm
    if device == "cuda":
        torch.cuda.empty_cache()
    return (queries @ docs.T).astype(np.float32)
