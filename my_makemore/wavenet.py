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
        self.gain = torch.randn((1,dim))
        self.bias = torch.randn((1,dim))
        self.E = torch.randn((1,dim))
        self.var = torch.randn((1,dim))

    def __call__(self,x):
        if self.mode == 'train':
            E_running = torch.mean(x, dim = 0, keepdim=True)
            var_running = 1/(x.shape[0] - 1) * ((x-E_running)**2).sum(dim=0, keepdim=True)

            self.out = self.gain * ((x-E_running)/(var_running + self.eps)**0.5) + self.bias

            with torch.no_grad():
                self.E = (1 - self.momentum) * self.E + (self.momentum) * E_running
                self.var = (1 - self.momentum) * self.var + self.momentum * var_running

        elif self.mode == 'eval':
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

class Flatten:
    def __call__(self, x):
        return x.view(x.shape[0], -1)
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

emb_dim = 10
batch_size = 32
n_hidden = 200

# layers = [
#   Linear(emb_dim * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#   Linear(           n_hidden, len(symbols), bias=False),
# ]

model = Sequential([
    Embedding(len(symbols), emb_dim), Flatten(),
    Linear(emb_dim * block_size, n_hidden, False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, len(symbols), False)
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
max_steps = 1#200_000
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


# %%
plt.figure()
plt.grid()
plt.plot(torch.arange(lossi.shape[0]), lossi)

# %%
