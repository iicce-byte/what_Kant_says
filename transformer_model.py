import math
import torch
from torch import nn
from torch.nn import functional as F

import locallib as lll
import draw
import data

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder, self.decoder = encoder, decoder
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)



class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = lll.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    
    def forward(self, queries, keys, values, valid_lens):
        # query, key, value输入不做处理
        # 输出对齐输入
        def transpose_qkv(X, num_heads):
            X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
            X = X.permute(0,2,1,3)
            return X.reshape(-1, X.shape[2], X.shape[3])
        def transpose_out(X, num_heads):
            X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
            X = X.permute(0,2,1,3)
            return X.reshape(X.shape[0], X.shape[1], -1)

        queries = transpose_qkv(self.W_q(queries), num_heads=self.num_heads)
        keys = transpose_qkv(self.W_k(keys), num_heads=self.num_heads)
        values = transpose_qkv(self.W_v(values), num_heads=self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output =self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_out(output, self.num_heads)
        return self.W_o(output_concat)

class PositionEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y)+X)

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_inputs, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                            num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_inputs, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                    EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                 norm_shape, ffn_num_inputs, ffn_num_hiddens,
                                 num_heads, dropout, use_bias))
    
    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weights = [None]*len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_inputs, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size,
                                             num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size,
                                             num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None: key_values = X
        else: key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps+1,
                    device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_inputs, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                    DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                 norm_shape, ffn_num_inputs, ffn_num_hiddens,
                                 num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None]*self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self._attention_weights = [[None]*len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    
    @property
    def attention_weights(self): return self._attention_weights


file1 = '../data/Kant/fundamental_principles_of_the_metaphysic_of_morals' # 道德形而上学原理
file2 = '../data/Kant/the_critique_of_pure_reason'      # 纯粹理性批判
file3 = '../data/Kant/the_critique_of_practical_reason' # 实践理性批判
file4 = '../data/Kant/Kant\'s_critique_of_judgement'    # 判断力批判

files = [file1, file2, file3]


num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, lll.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, vocab = data.load_data_Kant(batch_size, num_steps, files)
test_iter, _ = data.load_data_Kant(batch_size, num_steps, [file4])

encoder = TransformerEncoder(
    len(vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)

net = EncoderDecoder(encoder, decoder)






import math
import torch
from torch import nn
from locallib import Timer, Accumulator, try_gpu, seq_mask, masked_softmax
from matplotlib import pyplot as plt

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""
    def forward(self, pred, label, valid_len):
        # pred: (batch_size, num_steps, vocab_size)
        # label: (batch_size, num_steps)
        # valid_len: (batch_size,)
        weights = torch.ones_like(label)
        weights = seq_mask(weights, valid_len, value=0.0)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def xavier_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

def grad_clipping(net, theta):
    """Clip the gradient."""
    params = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    if len(params) == 0:
        return
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(figsize=figsize)
        self.config_axes = lambda: self.set_axes(
            xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib."""
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_xscale(xscale)
        self.axes.set_yscale(yscale)
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        if legend:
            self.axes.legend(legend)
        self.axes.grid()

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes.cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes.plot(x_vals, y_vals, fmt)
        self.config_axes()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

def train(net, train_iter, test_iter, vocab, lr, num_epochs, device):
    net.apply(xavier_init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], legend=['train'])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # Sum of training loss, number of tokens
        net.train()
        for batch in train_iter:
            optimizer.zero_grad()
            X, Y = [x.to(device) for x in batch]
            batch_size, num_steps = X.shape
            X_valid_len = torch.full((batch_size,), num_steps, device=device)
            dec_input = Y[:, :-1]
            dec_valid_len = torch.full((batch_size,), num_steps - 1, device=device)
            pred, _ = net(X, dec_input, X_valid_len)
            l = loss(pred, Y[:, 1:], dec_valid_len)
            l.sum().backward()
            grad_clipping(net, 1)
            optimizer.step()
            num_tokens = dec_valid_len.sum()
            if num_tokens > 0:
                metric.add(l.sum(), num_tokens)
        train_loss = metric[0] / metric[1] if metric[1] > 0 else 0.0
        animator.add(epoch + 1, (train_loss,))
        if (epoch + 1) % 10 == 0:
            net.eval()
            test_metric = Accumulator(2)
            with torch.no_grad():
                for batch in test_iter:
                    X, Y = [x.to(device) for x in batch]
                    batch_size, num_steps = X.shape
                    X_valid_len = torch.full((batch_size,), num_steps, device=device)
                    dec_input = Y[:, :-1]
                    dec_valid_len = torch.full((batch_size,), num_steps - 1, device=device)
                    pred, _ = net(X, dec_input, X_valid_len)
                    l = loss(pred, Y[:, 1:], dec_valid_len)
                    test_metric.add(l.sum(), dec_valid_len.sum())
            test_loss = test_metric[0] / test_metric[1] if test_metric[1] > 0 else 0.0
            print(f'epoch {epoch + 1}, train perplexity {math.exp(train_loss):.1f}, '
                  f'test perplexity {math.exp(test_loss):.1f}, '
                  f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


import model
train(net, train_iter, test_iter, vocab, lr, 10, device)