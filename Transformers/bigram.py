#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
#%%
text = open('input.txt', 'r').read()
print(text[:100])
# %%
vocab = sorted(list(set(''.join(text))))
vocab_size = len(vocab)

print(f'We have {vocab_size} symbols')
print(''.join(vocab))
#%%
stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for i,s in enumerate(vocab)}
encode = lambda data: [stoi[char] for char in ''.join(data)]
decode = lambda X: [itos[i] for i in X]

# print(encode("hii there"))
# print(decode(encode("hii there")))
# %%
data = torch.tensor(encode(text))
print(data[:100])
n = int(len(data)*0.9)

train_data = data[:n]
val_data = data[n:]

# %%
torch.manual_seed(1337)
block_size = 8
batch_size = 4
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[j:block_size+j] for j in ix])
    y = torch.stack([data[j+1:block_size+1+j] for j in ix])
    
    return x,y

Xb, Yb = get_batch('train')
print(Xb.shape)
print(Yb.shape)
#%%
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
    
    def __call__(self, x, targets=None):
        logits = self.token_embedding_table(x) # B,T,C --> B,C,T EMBS ARE STUCKED IN ROWS IN EVERY BATCH
        B, T, C = logits.shape                 # 4 8 64
        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, ix), dim=1)
        return idx

# %%
model = BigramLanguageModel(vocab_size, vocab_size)
output = decode(model.generate(idx = torch.zeros((1,1),dtype=torch.long), max_tokens=100).tolist()[0])
text = ''.join(output)

# %%
