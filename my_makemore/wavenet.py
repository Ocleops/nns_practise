#%%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
# %%
torch.manual_seed(42); # seed rng for reproducibility

class Linear:
    def __init__(self, fans_in, fans_out, bias):
        self.w = torch.randn((fans_in, fans_out)) * fans_in**-0.5
        self.b = torch.randn((1,fans_out)) if bias else None

    def __call__(self, x):
        self.out = x @ self.w
        if self.b is not None:
            self.out += self.b
        
        return self.out
    def parameters(self):
        return [self.w] if self.b == None else [self.w, self.b]

class BatchNorm1d:
    def __init__(self, dim, mode='train', eps=1e-05, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.mode = mode
        self.gain = torch.ones(dim)
        self.bias = torch.zeros(dim)
        self.E = torch.zeros(dim)
        self.var = torch.ones(dim)

    def __call__(self,x):
        if self.mode == 'train':
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim =(0, 1)
            
            else:
                assert('The BatchNorm layer for this dimension of x is not implemented! This might raise unexpected bugs due to wrong broadcasting.')
            
            E_running = x.mean(dim, keepdim = True) #torch.mean(x, dim, keepdim=True)
            var_running = x.var(dim, keepdim = True) # 1/(x.shape[0] - 1) * ((x-E_running)**2).sum(dim, keepdim=True)

            self.out = self.gain * ((x-E_running)/(var_running + self.eps)**0.5) + self.bias

            with torch.no_grad():
                self.E = (1 - self.momentum) * self.E + (self.momentum) * E_running
                self.var = (1 - self.momentum) * self.var + self.momentum * var_running

        else:
            self.out = self.gain * ((x-self.E)/(self.var + self.eps)**0.5) + self.bias
        
        return self.out
    
    def parameters(self):
        return [self.gain, self.bias]
    
class Tanh:
    def __call__(self, x):
        return torch.tanh(x)
    
    def parameters(self):
        return []

class Embedding:
    def __init__(self, vocab, emb_dim):
        self.w = torch.randn((vocab, emb_dim))
    def __call__(self,x):
        self.out = self.w[x]
        return self.out
    
    def parameters(self):
        return [self.w]

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)   

        if x.shape[1] == 1:
            x = torch.squeeze(x, dim=1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in  self.layers:
            x = layer(x)
        self.out = x 
        return self.out
    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]
    
def build_dataset(data, symbols, stoi, itos, block_size):
    X = []
    Y = []

    for name in data:
        context = [0] * block_size
        for char in name + '.': 
            X.append(context)
            ix = stoi[char]
            # x = ''.join(itos[i] for i in context)
            # print(f'{x} --> {itos[ix]}')
            Y.append(ix)
            context = context[1:] + [ix]
    
    return torch.tensor(X), torch.tensor(Y)

def split_loss(key):
    x,y = {
            'train': (Xtr, Ytr),
            'dev': (Xdev, Ydev),
            'test': (Xt, Yt),
    }[key] 

    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits,y)
        print(loss.item())

# %%
data = open('names.txt', 'r').read().splitlines()
random.seed(42)
random.shuffle(data)

symbols =['.'] + sorted(list(set(''.join(data))))

block_size = 8
stoi = {s:i for i,s in enumerate(symbols)}
itos = {i:s for i,s in enumerate(symbols)}

X, Y = build_dataset(data, symbols, stoi, itos, block_size)

n1 = int(len(Y) * 0.8)
n2 = int(len(Y) * 0.9)

Xtr, Ytr = X[:n1], Y[:n1]
Xdev, Ydev = X[n1:n2], Y[n1:n2]
Xt, Yt = X[n2:], Y[n2:]
# %%
batch_size = 32
vocab_size = len(symbols)
n_embd = 24 # the dimensionality of the character embedding vectors
n_hidden = 128 # the number of neurons in the hidden layer of the MLP
model = Sequential([
  Embedding(vocab_size, n_embd),
  FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, vocab_size, False),
])
    
# %%
for parameter in model.parameters():
    parameter.requires_grad = True

with torch.no_grad():
    for layer in model.layers[:-1]:
        if isinstance(layer, Linear):
            layer.w *= 5/3

    model.layers[-1].w *= 0.1

# %%
lossi = []
max_steps = 200_000
for i in range(max_steps):
    ix = torch.randint(low=0, high=Xtr.shape[0], size=(batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]
    
    logits = model(Xb)

    loss = F.cross_entropy(logits, Yb)

    for parameter in model.parameters():
        parameter.grad = None

    loss.backward()
    
    lr = 0.1 if i < 100_000 else 0.01

    for parameter in model.parameters():
        parameter.data -= lr * parameter.grad

    if i % 10000 == 0: 
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    
    lossi.append(loss.item())
#%%
with torch.no_grad():
    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.mode = 'eval'

split_loss('train')
split_loss('dev')
#%%
lossi = torch.tensor(lossi).view(-1, 1000).mean(dim=1)
plt.figure()
plt.grid()
plt.plot(torch.arange(lossi.shape[0]), lossi)
# %%
# sample from the model
for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      # forward pass the neural net
      logits = model(torch.tensor([context]))
      probs = F.softmax(logits, dim=1)
      # sample from the distribution
      ix = torch.multinomial(probs, num_samples=1).item()
      # shift the context window and track the samples
      context = context[1:] + [ix]
      out.append(ix)
      # if we sample the special '.' token, break
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out)) # decode and print the generated word
# %%
