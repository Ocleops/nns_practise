#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
# %%

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
    
#%%
torch.manual_seed(1337)

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

block_size = 8
batch_size = 32
lr = 1e-3
max_steps = 1_000#200_000
eval_interval = 500
max_eval_iter = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

model = BigramLanguageModel(vocab_size, vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

#%%
for _ in range(max_steps):
    Xb, Yb = get_batch('train')
    logits, loss = model(Xb,Yb)

    if _ % eval_interval == 0:
        loss_estimation()

    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    optimizer.step()

# %%
output =''.join(decode(model.generate(idx = torch.zeros((1,1),dtype=torch.long), max_tokens=1000).tolist()[0]))
print(output)
# %%
