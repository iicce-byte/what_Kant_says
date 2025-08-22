import math
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

import data
import locallib as lll
import draw

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens*2, self.vocab_size)
    
    def forward(self, inputs, state):
        """outputs1: [num_steps*batch_size, vocab_size]
        outputs2: hidden state"""
        X = F.one_hot(inputs.T.long(), self.vocab_size).to(torch.float32)
        Y, state = self.rnn(X, state)
        outputs = self.linear(Y.reshape((-1, Y.shape[-1])))
        return outputs, state
    
    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions*self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions*self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions*self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device))

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = lll.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MutiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MutiHeadAttention, self).__init__(**kwargs)
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
        super(PositionEncoding, self).__init__(**kwargs)
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
        self.attention = MutiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                           num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期"""
    state, timer = None, lll.Timer()
    metric = lll.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # if isinstance(net, nn.Module) and not isinstance(state, tuple):
            # for s in state: s.detach_()
            state.detach_()
            # else:
            #     for s in state:
            #         s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        updater.zero_grad()
        l.backward()
        updater.step()
        metric.add(l*y.numel(), y.numel())
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()

def evaluate_ppl(net, data_iter, loss, device):
    net.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():  # Disable gradient computation
        for X, Y in data_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)  # Initialize state
            X, Y = X.to(device), Y.to(device)
            y_hat, state = net(X, state)  # Forward pass with state
            y = Y.T.reshape(-1)  # Flatten target
            l = loss(y_hat, y.long()).mean()  # Compute loss
            total_loss += l * y.numel()  # Accumulate loss scaled by number of tokens
            total_tokens += y.numel()  # Accumulate total tokens
    net.train()  # Set model back to training mode
    return math.exp(total_loss / total_tokens)  # Return perplexity

def train(net, train_iter, test_iter, vocab, lr, num_epochs, device,
          use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = draw.Animator(xlabel='epoch', ylabel='perplexity',
            legend=['train ppl', 'test ppl'], xlim=[10, num_epochs])
    updater = torch.optim.SGD(net.parameters(), lr)
    # predict = lambda prefix: predict(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        train_ppl, train_speed = train_epoch(net, train_iter, loss,
                updater, device, use_random_iter)
        test_ppl = evaluate_ppl(net, test_iter, loss, device)

        if (epoch+1)%10 == 0:
            print(predict('mind', 50, net, vocab, device=device))
            animator.add(epoch+1, [train_ppl, test_ppl])
    print(f'训练集困惑度: {train_ppl:.2f}')
    print(f'测试集困惑度: {test_ppl:.2f}')
    print(f'{train_speed:.1f} tokens/sec on {str(device)}')
    print(predict('mind', 50, net, vocab, device=device))
    print(predict('mind is ', 50, net, vocab, device=device))





file1 = '../data/Kant/fundamental_principles_of_the_metaphysic_of_morals' # 道德形而上学原理
file2 = '../data/Kant/the_critique_of_pure_reason'      # 纯粹理性批判
file3 = '../data/Kant/the_critique_of_practical_reason' # 实践理性批判
file4 = '../data/Kant/Kant\'s_critique_of_judgement'    # 判断力批判

files = [file1, file2, file3]

device = lll.try_gpu()
batch_size, num_steps = 32, 35
train_iter, vocab = data.load_data_Kant(batch_size, num_steps, file_list=files)
test_iter, _ = data.load_data_Kant(batch_size, num_steps, file_list=[file4])
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens, num_layers=3)
gru_layer = nn.GRU(len(vocab), num_hiddens, num_layers=3)
net = RNNModel(gru_layer, len(vocab)).to(device)
print(predict('mind is ', 10, net, vocab, device=device))
train(net, train_iter, test_iter, vocab, lr=0.1, num_epochs=100, device=device)
# print(len(vocab))

plt.ioff()
plt.show()