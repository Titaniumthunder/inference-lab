#!/usr/bin/env python3
"""Perplexity of a running vLLM server on WikiText-2 test (same text for every config)."""
import argparse, csv, math, os
import requests
from datasets import load_dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000")
ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--chunks", type=int, default=40)
ap.add_argument("--chunk-len", type=int, default=2048)
ap.add_argument("--out", default="results/ppl.csv")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text)["input_ids"]
nll, n = 0.0, 0
for c in range(a.chunks):
    chunk = ids[c * a.chunk_len:(c + 1) * a.chunk_len]
    if len(chunk) < a.chunk_len:
        break
    r = requests.post(a.base_url + "/v1/completions", timeout=600, json={
        "model": a.model, "prompt": chunk, "max_tokens": 1,
        "temperature": 0, "prompt_logprobs": 0})
    r.raise_for_status()
    pl = r.json()["choices"][0]["prompt_logprobs"]
    for t, d in zip(chunk[1:], pl[1:]):
        nll -= d[str(t)]["logprob"]
        n += 1
ppl = math.exp(nll / n)
print(f"[{a.label}] perplexity {ppl:.4f} over {n} tokens")
new = not os.path.exists(a.out)
with open(a.out, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["label", "model", "perplexity", "tokens"])
    w.writerow([a.label, a.model, f"{ppl:.4f}", n])
