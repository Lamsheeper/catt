"""Held-out accuracy of a fine-tuned model.

Function Bindings: every function f_i at every input x in [1, 100], asked as
"The output of <Bi>(x) is ". One forward pass per prompt; at the last position

  accuracy             argmax over the answer tokens <V..> the tokenizer carries
  open_vocab_accuracy  argmax over the whole vocabulary (the paper's "accuracy")

Rule Routings: the eight held-out assignments of every category, asked about a
held-out noun, as a forced choice between the first tokens of " Cedar" and
" Birch" at the next position.

  accuracy      questions naming the queues in corpus order ("Cedar ... or Birch")
  accuracy_ba   the same questions with the two names swapped
  selectivity   P(answer flips | deciding property flips)
                - P(answer flips | an inert property flips),
                over all single-property flips of the 16 assignments (corpus order)
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from catt.benchmark import QUERY_TEMPLATE, RR_ASK, RR_OUTCOMES, read_jsonl

# ── Function Bindings ─────────────────────────────────────────────────────
FB_MAX_INPUT = 100
FB_BATCH_SIZE = 50


def fb_truth(functions: int) -> Dict[str, int]:
    """func token -> constant: f_i returns i, encoded <Bii> -> <Vii>."""
    return dict(sorted((f"<B{i:02d}>", i) for i in range(1, functions + 1)))


def fb_prompts(functions: int) -> List[dict]:
    return [{"function": f, "input": x, "expected_constant": c,
             "prompt": QUERY_TEMPLATE.format(func=f, input=x)}
            for f, c in fb_truth(functions).items() for x in range(1, FB_MAX_INPUT + 1)]


def answer_candidates(tok, limit: int = 201) -> List[int]:
    """The answer values this tokenizer can emit as one <V..> token."""
    unk = tok.unk_token_id
    return [v for v in range(limit)
            if (i := tok.convert_tokens_to_ids(f"<V{v:02d}>")) is not None and i != unk]


@torch.no_grad()
def _last_logits(model, tok, prompts: Sequence[str], batch_size: int):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(prompts), batch_size):
        enc = tok(list(prompts[i:i + batch_size]), return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        yield model(**enc).logits[:, -1, :].float()


def function_bindings(model, tok, functions: int, batch_size: int = FB_BATCH_SIZE) -> dict:
    prompts = fb_prompts(functions)
    values = answer_candidates(tok)
    cand_ids = [tok.convert_tokens_to_ids(f"<V{v:02d}>") for v in values]
    uncovered = sorted(set(fb_truth(functions).values()) - set(values))
    if uncovered:
        raise SystemExit(f"constants with no answer token in this tokenizer: {uncovered}")
    id_to_value = dict(zip(cand_ids, values))
    cand = torch.tensor(cand_ids, device=model.device)

    scored = []
    for logits in _last_logits(model, tok, [p["prompt"] for p in prompts], batch_size):
        restricted = logits.index_select(1, cand)
        best_c = restricted.argmax(-1)
        prob_c = restricted.softmax(-1).gather(1, best_c[:, None]).squeeze(1)
        best_v = logits.argmax(-1)
        prob_v = logits.softmax(-1).gather(1, best_v[:, None]).squeeze(1)
        scored += [(int(cand[best_c[j]]), float(prob_c[j]), int(best_v[j]), float(prob_v[j]))
                   for j in range(len(logits))]

    results = []
    n_ok = n_free = 0
    for p, (bid, bprob, fid, fprob) in zip(prompts, scored):
        pred = id_to_value[bid]
        ok = pred == p["expected_constant"]
        free_ok = fid == bid and ok
        n_ok += ok
        n_free += free_ok
        results.append({
            "function": p["function"], "input": p["input"],
            "expected_constant": p["expected_constant"],
            "best_prediction": pred, "is_correct": bool(ok), "confidence": bprob,
            "open_vocab_correct": bool(free_ok),
            "open_vocab_prediction": tok.decode([fid]), "open_vocab_confidence": fprob,
        })
    total = len(results)
    return {
        "analysis": {
            "accuracy": n_ok / total,
            "open_vocab_accuracy": n_free / total,
            "total_prompts": total,
            "correct": n_ok,
            "num_candidates": len(values),
        },
        "results": results,
    }


# ── Rule Routings ─────────────────────────────────────────────────────────
RR_BATCH_SIZE = 8


def load_rules(path, categories: int) -> List[dict]:
    """The first `categories` rule records of a rule-seed file (banks nest by prefix)."""
    rows = read_jsonl(path)
    if len(rows) < categories:
        raise SystemExit(f"{path} has {len(rows)} rules, fewer than {categories}")
    return rows[:categories]


def all_cases(facts: Sequence[str]) -> List[Dict[str, bool]]:
    """All 2^n assignments; bit k of the index is facts[k]."""
    return [{f: bool(i >> k & 1) for k, f in enumerate(facts)} for i in range(2 ** len(facts))]


def _clause(phrase: str, kind: str, value: bool) -> str:
    if kind == "adj":
        return f"is {phrase}" if value else f"is not {phrase}"
    if value:
        return f"has {'an' if phrase[0].lower() in 'aeiou' else 'a'} {phrase}"
    return f"has no {phrase}"


def rr_question(record: dict, case: Dict[str, bool], noun: str, order: str = "ab") -> str:
    a, b = RR_OUTCOMES if order == "ab" else RR_OUTCOMES[::-1]
    parts = [_clause(record["phrases"][f], record["kinds"][f], case[f]) for f in record["facts"]]
    return f"The {noun} " + "; ".join(parts) + f".\n{RR_ASK.format(a=a, b=b)}\nAnswer:"


def flip_pairs(n_facts: int, bit: int):
    """Unordered assignment pairs differing only in `bit`."""
    return [(i, i ^ (1 << bit)) for i in range(2 ** n_facts) if not i & (1 << bit)]


def outcome_ids(tok) -> Dict[str, int]:
    pieces = {w: tok.encode(" " + w, add_special_tokens=False) for w in RR_OUTCOMES}
    ids = {w: p[0] for w, p in pieces.items()}
    if len(set(ids.values())) < 2:
        raise SystemExit(f"outcome words share a first token: {pieces}")
    return ids


def decide(model, tok, prompts: Sequence[str], ids: Dict[str, int],
           batch_size: int = RR_BATCH_SIZE) -> List[bool]:
    """True where the model prefers Cedar (ties go to Cedar)."""
    pos, neg = ids[RR_OUTCOMES[0]], ids[RR_OUTCOMES[1]]
    out = []
    for logits in _last_logits(model, tok, prompts, batch_size):
        out += [logits[j, pos].item() >= logits[j, neg].item() for j in range(len(logits))]
    return out


def route_category(model, tok, record: dict, ids: Dict[str, int], order: str = "ab",
                   flips: bool = True) -> Dict[str, List[int]]:
    """[hits, total] for direct accuracy and, with `flips`, the two flip rates."""
    noun = record["test_nouns"][0]
    cases = all_cases(record["facts"])

    def picks(indices):
        return decide(model, tok, [rr_question(record, cases[i], noun, order) for i in indices], ids)

    scored = record["test_cases"]
    chosen = picks(scored)
    out = {"direct": [sum(p == (record["truth_table"][i] == "1") for p, i in zip(chosen, scored)),
                      len(scored)]}
    if flips:
        deciding = record["facts"].index(record["relevant_facts"][0])
        out["relevant_flip"], out["irrelevant_flip"] = [0, 0], [0, 0]
        for bit in range(len(record["facts"])):
            pairs = flip_pairs(len(record["facts"]), bit)
            lo, hi = picks([i for i, _ in pairs]), picks([j for _, j in pairs])
            key = "relevant_flip" if bit == deciding else "irrelevant_flip"
            out[key][0] += sum(x != y for x, y in zip(lo, hi))
            out[key][1] += len(pairs)
    return out


def _rate(per: List[List[int]]) -> float:
    return sum(h for h, _ in per) / sum(t for _, t in per)


def rule_routings(model, tok, records: List[dict]) -> dict:
    ids = outcome_ids(tok)
    ab = [route_category(model, tok, r, ids, "ab") for r in records]
    ba = [route_category(model, tok, r, ids, "ba", flips=False) for r in records]
    per = {k: [c[k] for c in ab] for k in ab[0]}
    per["direct_ba"] = [c["direct"] for c in ba]
    acc, acc_ba = _rate(per["direct"]), _rate(per["direct_ba"])
    rel, irr = _rate(per["relevant_flip"]), _rate(per["irrelevant_flip"])
    return {
        "categories": len(records),
        "accuracy": acc, "accuracy_ba": acc_ba, "accuracy_both": (acc + acc_ba) / 2,
        "selectivity": rel - irr, "relevant_flip": rel, "irrelevant_flip": irr,
        "per_category": {"category_id": [r["category_id"] for r in records], **per},
    }
