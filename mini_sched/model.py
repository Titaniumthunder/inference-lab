"""Qwen2 forward pass over a paged KV cache, reusing the Hugging Face weights and RoPE helpers.

One step processes a flat batch of new tokens from several sequences (prefills and decodes mixed).
New K/V are scattered into their cache slots, then each sequence attends over its full context,
gathered from its blocks. Attention runs per group (prefill group, decode group) with padding and
a mask, through torch SDPA with grouped-query attention. No CUDA graphs, no custom kernels.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from .kv_cache import PagedKVCache


class PagedQwen2:
    def __init__(self, model_name, num_blocks, block_size=16, device="cuda"):
        hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
        self.hf, self.m, self.device = hf, hf.model, device
        self.lm_head_fp32 = hf.lm_head.weight.float()  # 1.2 GB, made once (see step())
        c = hf.config
        self.n_heads, self.n_kv = c.num_attention_heads, c.num_key_value_heads
        self.head_dim = c.hidden_size // c.num_attention_heads
        self.cache = PagedKVCache(c.num_hidden_layers, num_blocks, block_size, self.n_kv,
                                  self.head_dim, torch.bfloat16, device)

    def _group_index(self, seqs, q_off):
        """Padded index tensors for one attention group. seqs: list of (q_len, ctx_len, block_table)."""
        bs = self.cache.block_size
        qmax = max(q for q, _, _ in seqs)
        lmax = max(l for _, l, _ in seqs)
        q_idx = torch.zeros(len(seqs), qmax, dtype=torch.long)
        q_ok = torch.zeros(len(seqs), qmax, dtype=torch.bool)
        slots = torch.zeros(len(seqs), lmax, dtype=torch.long)
        mask = torch.zeros(len(seqs), 1, qmax, lmax, dtype=torch.bool)
        for b, ((q, l, table), off) in enumerate(zip(seqs, q_off)):
            q_idx[b, :q] = torch.arange(off, off + q)
            q_ok[b, :q] = True
            pos = torch.arange(l)
            tab = torch.tensor(table, dtype=torch.long)
            slots[b, :l] = tab[pos // bs] * bs + pos % bs
            start = l - q  # absolute position of the first new token
            qpos = torch.arange(qmax)[:, None] + start
            mask[b, 0] = (torch.arange(lmax)[None, :] <= qpos) & (torch.arange(lmax)[None, :] < l)
            mask[b, 0, q:] = False
            mask[b, 0, q:, 0] = True  # padded query rows attend to one slot so SDPA does not emit NaN
        d = self.device
        return q_idx.to(d), q_ok.to(d), slots.to(d), mask.to(d)

    @torch.inference_mode()
    def step(self, batch):
        """batch: list of dicts with tokens (new token ids), start (position of the first), block_table.
        Returns the greedy next token for each entry (from its last new token)."""
        ids, pos, slot_map, q_off, groups = [], [], [], [], {"prefill": [], "decode": []}
        for i, e in enumerate(batch):
            n = len(e["tokens"])
            q_off.append(len(ids))
            ids += e["tokens"]
            p = list(range(e["start"], e["start"] + n))
            pos += p
            slot_map += [self.cache.slot(e["block_table"], x) for x in p]
            groups["decode" if n == 1 else "prefill"].append(i)
        d = self.device
        ids_t = torch.tensor(ids, device=d)
        pos_t = torch.tensor(pos, device=d)
        slot_t = torch.tensor(slot_map, device=d)
        index = {}
        for g, members in groups.items():
            if members:
                seqs = [(len(batch[i]["tokens"]), batch[i]["start"] + len(batch[i]["tokens"]),
                         batch[i]["block_table"]) for i in members]
                index[g] = self._group_index(seqs, [q_off[i] for i in members])

        h = self.m.embed_tokens(ids_t)[None]            # [1, T, hidden]
        cos, sin = self.m.rotary_emb(h, pos_t[None])    # [1, T, head_dim]
        T, H, KV, D = len(ids), self.n_heads, self.n_kv, self.head_dim
        for li, layer in enumerate(self.m.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = a.q_proj(x).view(1, T, H, D).transpose(1, 2)
            k = a.k_proj(x).view(1, T, KV, D).transpose(1, 2)
            v = a.v_proj(x).view(1, T, KV, D)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q = q[0].transpose(0, 1)                     # [T, H, D]
            self.cache.k[li].index_copy_(0, slot_t, k[0].transpose(0, 1))
            self.cache.v[li].index_copy_(0, slot_t, v[0])
            out = torch.empty(T, H, D, dtype=q.dtype, device=d)
            for q_idx, q_ok, slots, mask in index.values():
                qq = q[q_idx].transpose(1, 2)                      # [B, H, Qmax, D]
                kk = self.cache.k[li][slots].transpose(1, 2)       # [B, KV, Lmax, D]
                vv = self.cache.v[li][slots].transpose(1, 2)
                o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, enable_gqa=True)
                o = o.transpose(1, 2)                              # [B, Qmax, H, D]
                out[q_idx[q_ok]] = o[q_ok]
            h = h + a.o_proj(out.reshape(1, T, H * D))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        last = torch.tensor([q_off[i] + len(e["tokens"]) - 1 for i, e in enumerate(batch)], device=d)
        # fp32 logits for the few sampled positions: BF16 logits tie often (spacing 0.125 near 20), and
        # argmax over ties then depends on batch-shape rounding
        logits = self.m.norm(h[0, last]).float() @ self.lm_head_fp32.T
        return logits.argmax(-1).tolist()
