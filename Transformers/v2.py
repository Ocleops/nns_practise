#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
#%%
# HYPER PARAMETERS
torch.manual_seed(1337)

block_size = 8
batch_size = 32
n_embd = 32
head_size = 8
num_heads = 4
lr = 1e-3
max_steps = 1_000 #200_000
eval_interval = 500
max_eval_iter = 300
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
text = open('input.txt', 'r').read()
vocab = sorted(list(set(''.join(text))))
vocab_size = len(vocab)

stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for i,s in enumerate(vocab)}
encode = lambda data: [stoi[char] for char in ''.join(data)]
decode = lambda X: [itos[i] for i in X]

data = torch.tensor(encode(text))
n = int(len(data)*0.9)

train_data = data[:n]
val_data = data[n:]

#%%
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[j:block_size+j] for j in ix])
    y = torch.stack([data[j+1:block_size+1+j] for j in ix])
    
    return x,y

@torch.no_grad()
def loss_estimation():
    model.eval()
    ls = {}
    for split in ['train', 'eval']:
        losses = torch.zeros((max_eval_iter))
        for iter in range(max_eval_iter):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[iter] = loss.item()
        ls[split] = losses.mean().item()
    train_loss = ls['train']
    eval_loss = ls['eval']
    print(f'Training loss: {train_loss:.4f} Eval loss: {eval_loss:.4f}')
    model.train()
# %%
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.Wq = nn.Linear(n_embd, head_size, bias=False)
        self.Wk = nn.Linear(n_embd, head_size, bias=False)
        self.Wv = nn.Linear(n_embd, head_size, bias=False) # B, T, head_size
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    def forward(self, x):
        x = torch.randn((1,1,32))
        B, T, C = x.shape
        Eqs = self.Wq(x) # B, T, head_size
        Eks = self.Wk(x)
        Evs = self.Wv(x) 
        # DOT PRODUCT OF QUERY WITH ALL OTHER KEYS TO GET wei
        Wei = Eqs @ Eks.transpose(-1,-2)* Eks.shape[-1]**-0.5 # B, T, T
        Wei = Wei.masked_fill(self.tril[:T, :T] == 0.0, float('-inf'))
        Wei = F.softmax(Wei, dim = -1)

        self.out = Wei @ Evs

        return self.out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=n_embd//num_heads) for i in range(num_heads)])

    def forward(self, x):
        self.out = torch.cat([head(x) for head in self.heads],  dim=-1) 

        return self.out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        self.out = self.net(x)

class Block(nn.Modules):
    def __init__(self, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads)
        self.FF = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        X = self.attention(self.ln1(x))
        x = self.FF(self.ln2(x))

        


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.tokken_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.attention = MultiHeadAttention(num_heads=num_heads)
        self.ll = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x):
        B, T = x.shape
        tokken_embds = self.tokken_embedding_table(x) # dims = [B, T, C]
        pos_embds = self.position_embedding_table(torch.arange(T)) # dims = [T, C]

        x = tokken_embds + pos_embds

        x = self.attention(x)
        self.out = self.ll(x)

        return self.out
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # GRABS THE LAST block_size COLUMNS OF YOUR INPUT.
                                            # WE ARE ALWAYS USING THE CLOSEST PAST COLUMNS TO
                                            # MAKE OUR PREDICTION
            
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim= -1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True)

            idx = torch.cat((idx, ix), dim=1)
        
        return idx

#%%

Xb, Yb = get_batch('train')
model = LanguageModel(vocab_size, n_embd)
# logits = model.forward(Xb)

# %%

output =''.join(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=200).tolist()[0]))
print(output)
# %%
