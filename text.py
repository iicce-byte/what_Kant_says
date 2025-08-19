import collections
import re

file1 = '../data/Kant/fundamental_principles_of_the_metaphysic_of_morals' # 道德形而上学原理
file2 = '../data/Kant/the_critique_of_pure_reason'      # 纯粹理性批判
file3 = '../data/Kant/the_critique_of_practical_reason' # 实践理性批判
file4 = '../data/Kant/Kant\'s_critique_of_judgement'    # 判断力批判

files = [file1, file2, file3, file4]

def read_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def read_and_merge_files(file_list):
    all_lines = []
    for file in file_list:
        lines = read_file(file)
        all_lines.extend(lines)
    return all_lines

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    raise TypeError

def count_corpus(token):
        """统计token的频率"""
        if len(token)==0 or isinstance(token[0], list):
            token = [token for line in token for token in line]
        return collections.Counter(token)

class Vocab:
    """文本词表"""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx 
                for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self): return 0
    @property
    def token_freqs(self): return self._token_freqs

def load_corpus_Kant(file_list, token_type='char', max_tokens=-1):
    if not isinstance(file_list, list): file_list = [file_list]
    lines = read_and_merge_files(file_list)
    tokens = tokenize(lines, token_type)
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_Kant(file1)
print(len(corpus), '\n', len(vocab))