import torch
import torch.nn as nn

class RunningMeanStd(nn.Module):
    def __init__(self, shape, eps=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(eps))
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if not self.training:
            return
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / torch.sqrt(self.var + self.eps)