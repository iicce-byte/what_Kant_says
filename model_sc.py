import torch
from torch import nn
from torch.nn import functional as F
import math

import data

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape): return torch.randn(size=shape, device=device)*0.01
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for p in params: p.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    """rnn process
    outputs: [B*num_steps, Embed or Vocab size],
    hidden state, _
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh)+torch.mm(H, W_hh)+b_h)
        Y = torch.mm(H, W_hq)
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, )

class RNN(nn.Module):
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_rnn_state, forward_fn
    def forward(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

file1 = '../data/Kant/fundamental_principles_of_the_metaphysic_of_morals' # 道德形而上学原理
file2 = '../data/Kant/the_critique_of_pure_reason'      # 纯粹理性批判
file3 = '../data/Kant/the_critique_of_practical_reason' # 实践理性批判
file4 = '../data/Kant/Kant\'s_critique_of_judgement'    # 判断力批判

files = [file1, file2, file3, file4]

batch_size, num_steps = 32, 35
train_iter, vocab = data.load_data_Kant(batch_size, num_steps, files, token_type='word')
