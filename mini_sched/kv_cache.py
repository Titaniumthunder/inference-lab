"""Paged KV cache: a fixed pool of blocks (block_size tokens each), handed out by a free-list allocator.

A sequence owns a block table (list of block ids). Token at position p lives in slot
block_table[p // block_size] * block_size + p % block_size of the flat per-layer pool.
"""
import torch


class BlockAllocator:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.free = list(range(num_blocks - 1, -1, -1))  # pop() hands out low ids first
        self.used = set()

    def num_free(self):
        return len(self.free)

    def allocate(self):
        if not self.free:
            raise MemoryError("no free KV blocks")
        b = self.free.pop()
        self.used.add(b)
        return b

    def release(self, blocks):
        for b in blocks:
            if b not in self.used:
                raise ValueError(f"block {b} released twice or never allocated")
            self.used.remove(b)
            self.free.append(b)


class PagedKVCache:
    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, device):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        shape = (num_layers, num_blocks * block_size, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    @staticmethod
    def bytes_per_token(num_layers, num_kv_heads, head_dim, dtype_bytes=2):
        return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes  # K and V

    def blocks_for(self, n_tokens):
        return -(-n_tokens // self.block_size)

    def slot(self, block_table, pos):
        return block_table[pos // self.block_size] * self.block_size + pos % self.block_size
