"""Continuous-batching scheduler over the paged KV cache (vLLM-style policy, much simplified).

Every step:
  1. each running sequence decodes one token; if it needs a new block and none is free, the most
     recently admitted running sequence is preempted (blocks freed, re-queued at the front, its
     prompt + output recomputed later)
  2. waiting requests that have arrived are admitted first come first served while
     running < max_num_seqs, the step stays within max_num_batched_tokens, and blocks are free
  3. one model step over the mixed batch; greedy tokens appended; finished sequences free blocks
policy="static" is the request-level baseline: admit only when nothing is running, then run that
batch to completion (what HF generate-style serving does). Same model code, so only scheduling differs.
"""
from dataclasses import dataclass, field


@dataclass
class Request:
    rid: int
    prompt: list
    max_tokens: int
    arrival: float = 0.0
    output: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    computed: int = 0          # tokens whose K/V are in the cache
    first_token_t: float = None
    finish_t: float = None
    preemptions: int = 0

    @property
    def tokens(self):
        return self.prompt + self.output


class Scheduler:
    def __init__(self, model, max_num_seqs=64, max_num_batched_tokens=4096, policy="continuous"):
        assert policy in ("continuous", "static")
        self.model, self.cache = model, model.cache
        self.alloc = self.cache.allocator
        self.max_num_seqs, self.budget, self.policy = max_num_seqs, max_num_batched_tokens, policy
        self.waiting, self.running, self.finished = [], [], []
        self.stats = {"steps": 0, "preemptions": 0, "peak_blocks": 0, "peak_batch": 0}

    def add(self, req):
        assert len(req.prompt) <= self.budget, "prompt longer than the step token budget (no chunked prefill)"
        self.waiting.append(req)

    def has_work(self):
        return bool(self.waiting or self.running)

    def _grow(self, req, n_tokens):
        """Make sure req owns blocks for n_tokens positions. False if the pool is out of blocks."""
        while len(req.blocks) < self.cache.blocks_for(n_tokens):
            if not self.alloc.num_free():
                return False
            req.blocks.append(self.alloc.allocate())
        return True

    def _preempt(self, req):
        self.alloc.release(req.blocks)
        req.blocks, req.computed = [], 0
        req.preemptions += 1
        self.stats["preemptions"] += 1
        self.running.remove(req)
        self.waiting.insert(0, req)

    def _schedule(self, now):
        batch = []
        # 1. decodes for running sequences, oldest first; preempt the newest on block shortage
        for req in list(self.running):
            if req not in self.running:
                continue  # preempted below while making room for an older sequence
            while not self._grow(req, len(req.tokens)):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is req:
                    break
            if req in self.running:
                batch.append(req)
        used = len(batch)
        # 2. admissions
        if self.policy == "static" and self.running:
            return batch
        while self.waiting and self.waiting[0].arrival <= now and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            n = len(req.tokens)
            if used + n > self.budget and batch:
                break
            if self.alloc.num_free() < self.cache.blocks_for(n):
                break
            self._grow(req, n)
            self.waiting.pop(0)
            self.running.append(req)
            batch.append(req)
            used += n
        return batch

    def step(self, now, clock):
        """One engine step. clock() gives the time used for token timestamps."""
        batch = self._schedule(now)
        if not batch:
            return 0
        entries = [{"tokens": r.tokens[r.computed:], "start": r.computed, "block_table": r.blocks} for r in batch]
        nxt = self.model.step(entries)
        t = clock()
        for r, tok in zip(batch, nxt):
            r.computed = len(r.tokens)
            r.output.append(tok)
            if r.first_token_t is None:
                r.first_token_t = t
            if len(r.output) >= r.max_tokens:
                r.finish_t = t
                self.alloc.release(r.blocks)
                r.blocks = []
                self.running.remove(r)
                self.finished.append(r)
        s = self.stats
        s["steps"] += 1
        s["peak_blocks"] = max(s["peak_blocks"], len(self.alloc.used))
        s["peak_batch"] = max(s["peak_batch"], len(batch))
        return len(batch)
