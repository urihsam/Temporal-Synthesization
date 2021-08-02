import torch
from torch import Tensor
import torch.nn.functional as f


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


# different records may have different seq_len, but same dim_model
def position_embedding(seq_len: int, dim_model: int) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model) # extendable, infinity space
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase)).cuda() # lookup rule


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(dim_input, dim_feedforward),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_feedforward, dim_input),
    )
