import torch
import torch.nn as nn

class MemoryPool(nn.Module):
    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size
        self.memory = []

        # stats (for tests/debug)
        self._push_calls = 0
        self._get_calls = 0
        self._get_sizes = []   # 每次 get() 时 memory 的 M
        self._push_sizes = []  # 每次 push() 后池子里条目数

    def reset(self):
        self.memory = []
        # reset stats too (important for pytest)
        self._push_calls = 0
        self._get_calls = 0
        self._get_sizes = []
        self._push_sizes = []

    def push(self, memory_entry: torch.Tensor):
        """
        memory_entry: [B, d_mem] or [B, 1, d_mem]
        """
        self._push_calls += 1

        if memory_entry.dim() == 2:
            memory_entry = memory_entry.unsqueeze(1)

        #  关键：跨 step 存储必须 detach
        memory_entry = memory_entry.detach()

        self.memory.append(memory_entry)

        if len(self.memory) > self.max_size:
            self.memory.pop(0)

        self._push_sizes.append(len(self.memory))


    def get(self):
        self._get_calls += 1

        m = len(self.memory)
        self._get_sizes.append(m)

        if m == 0:
            return None

        #  batch size 不一致时自动 reset
        batch_size = self.memory[0].shape[0]
        for mem in self.memory:
            if mem.shape[0] != batch_size:
                self.reset()
                return None

        return torch.cat(self.memory, dim=1)
