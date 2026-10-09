"""Tests for the mini scheduler. Run from the repo root: python -m mini_sched.test_mini
1. allocator bookkeeping (CPU)
2. greedy outputs equal HF generate (or first divergence is a near-tie under HF), with sequences joining a running batch (continuous batching)
3. same outputs when a tiny block pool forces preemption + recompute; all blocks freed at the end
"""
import sys, time
import torch
from transformers import AutoTokenizer

from .kv_cache import BlockAllocator, PagedKVCache
from .model import PagedQwen2
from .scheduler import Request, Scheduler

sys.path.insert(0, ".")
from hf_vs_vllm import prompts  # noqa: E402  same deterministic random-word prompts as section 15

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_OUT = 32
TIE = 0.25  # two BF16 steps at logit ~20


def test_allocator():
    a = BlockAllocator(4)
    got = [a.allocate() for _ in range(4)]
    assert sorted(got) == [0, 1, 2, 3] and a.num_free() == 0
    try:
        a.allocate()
        raise AssertionError("expected MemoryError")
    except MemoryError:
        pass
    a.release(got[:2])
    try:
        a.release(got[:1])
        raise AssertionError("expected double-free error")
    except ValueError:
        pass
    assert a.num_free() == 2
    print("allocator: ok")


def run(model, reqs, **kw):
    s = Scheduler(model, **kw)
    for r in reqs:
        s.add(r)
    t0 = time.perf_counter()
    while s.has_work():
        s.step(time.perf_counter() - t0, time.perf_counter)
    assert not model.cache.allocator.used, "blocks leaked"
    return s


def main():
    test_allocator()
    tok = AutoTokenizer.from_pretrained(MODEL)
    lens = [20, 100, 256, 600, 37, 300]
    ps = [prompts(tok, 1, n, f"mini{n}")[0] for n in lens]
    model = PagedQwen2(MODEL, num_blocks=2048)

    ref = []
    for p in ps:
        x = torch.tensor([p], device="cuda")
        out = model.hf.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=N_OUT,
                                min_new_tokens=N_OUT, do_sample=False, temperature=None, top_p=None,
                                top_k=None, repetition_penalty=1.0, pad_token_id=0)
        ref.append(out[0, len(p):].tolist())

    def check(name, sched):
        """Exact match, or the first divergence is a near-tie under HF itself (teacher-forced on the
        shared prefix): HF's logit for our token within TIE of HF's own top logit. Anything else fails."""
        got = {r.rid: r.output for r in sched.finished}
        res = []
        for i, p in enumerate(ps):
            k = next((j for j, (a, b) in enumerate(zip(ref[i], got[i])) if a != b), None)
            if k is None:
                res.append(("exact", N_OUT, 0.0))
                continue
            with torch.inference_mode():
                lg = model.hf(torch.tensor([p + ref[i][:k]], device="cuda")).logits[0, -1].float()
            gap = (lg[ref[i][k]] - lg[got[i][k]]).item()
            res.append(("tie" if gap <= TIE else "FAIL", k, gap))
        print(f"{name}: {[(a, k, round(g, 3)) for a, k, g in res]} stats {sched.stats}")
        return all(a != "FAIL" for a, _, _ in res), sum(a == "exact" for a, _, _ in res)

    # max_num_seqs=3 so later requests join while earlier ones are mid-decode
    m1 = check("continuous", run(model, [Request(i, p, N_OUT) for i, p in enumerate(ps)], max_num_seqs=3))
    # tiny pool: 1,313 tokens of prompt + 192 generated need ~96 blocks at once; give 70
    model.cache = PagedKVCache(len(model.m.layers), 70, 16, model.n_kv, model.head_dim, torch.bfloat16, "cuda")
    s2 = run(model, [Request(i, p, N_OUT) for i, p in enumerate(ps)], max_num_seqs=6)
    m2 = check("preemption", s2)
    assert s2.stats["preemptions"] > 0, "test did not force preemption"
    print(f"exact matches: continuous {m1[1]}/{len(ps)}, preemption {m2[1]}/{len(ps)}")
    print("PASS" if m1[0] and m2[0] else "FAIL")


if __name__ == "__main__":
    main()
