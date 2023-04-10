import torch
import bisect

class TDigest:
    def __init__(self, delta=0.01, K=25, device='cpu'):
        self.delta = delta
        self.K = K
        self.C = []
        self.N = torch.tensor(0., device=device)
        self.device = device

    def update(self, value, weight=1):
        value = torch.tensor(value, device=self.device)
        weight = torch.tensor(weight, device=self.device)
        if len(self.C) == 0:
            self.C.append((value, weight))
        else:
            idx = bisect.bisect_left(self.C, (value.item(), -1))
            if idx != len(self.C) and self.C[idx][0] == value:
                self.C[idx] = (value, self.C[idx][1] + weight)
            else:
                self.C.insert(idx, (value, weight))
        self.N += weight
        self.compress()

    def compress(self):
        if len(self.C) <= 1:
            return

        while True:
            min_diff = float('inf')
            merge_idx = -1
            for idx in range(1, len(self.C)):
                diff = (self.C[idx][0] - self.C[idx - 1][0]).item()
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = idx

            q = sum(w for _, w in self.C[:merge_idx])
            q_next = q + self.C[merge_idx][1]
            q_prev = q - self.C[merge_idx - 1][1]
            if min_diff <= self.delta * min(q, q_prev, q_next) / self.N:
                merged_weight = self.C[merge_idx - 1][1] + self.C[merge_idx][1]
                merged_value = (self.C[merge_idx - 1][0] * self.C[merge_idx - 1][1] + self.C[merge_idx][0] * self.C[merge_idx][1]) / merged_weight
                self.C[merge_idx - 1] = (merged_value, merged_weight)
                self.C.pop(merge_idx)
            else:
                break
