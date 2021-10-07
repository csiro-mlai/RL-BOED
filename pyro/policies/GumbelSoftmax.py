"""
An implementation of the GumbelSoftmax distribution, based on Wah Loon Keng's
tutorial
https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954
"""

import torch
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical


class GumbelSoftmax(RelaxedOneHotCategorical):
    def sample(self, sample_shape=torch.Size()):
        u = torch.empty(self.logits.size(),
                        device=self.logits.device,
                        dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        soft = super(RelaxedOneHotCategorical, self).rsample(sample_shape)
        index = soft.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(self.logits).scatter_(-1, index, 1.)
        return hard - soft.detach() + soft

    def log_prob(self, value):
        """value is one-hot or relaxed"""
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)


