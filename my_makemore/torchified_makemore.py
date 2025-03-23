#%%
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
#%%
class Linear:
    def __init__(self, fan_in, fan_out, bias = True):
        g = torch.Generator().manual_seed(2147483647)
        self.w = torch.randn((fan_in, fan_out), generator=g)  / (fan_in)**0.5
        self.b = torch.randn((1, fan_out), generator=g) if bias else None
    def __call__(self, x):
        self.out = x @ self.w
        if self.b is not None:
            self.out += self.b
        return self.out
    
    def parameters(self):
        return [self.w] + ([] if self.b is None else [self.b])
    
class BatchNorm1d:
    def __init__(self, num_features, epsilon = 1e-5, momemntum = 0.1, track_running_stats=True):
        g = torch.Generator().manual_seed(2147483647)

        self.epsilon = epsilon
        self.momentum = momemntum
        self.track_running_stats = track_running_stats

        self.gammas = torch.randn((1, num_features), generator=g)
        self.betas = torch.randn((1, num_features), generator=g)

        self.E_running = torch.randn((1, num_features), generator=g) if track_running_stats else None
        self.S_running = torch.randn((1, num_features), generator=g) if track_running_stats else None

    def __call__(self, x):
        if self.track_running_stats:
            E = torch.mean(x, dim=0, keepdim=True)
            sigma = torch.std(x, dim=0, keepdim=True)
        else:
            E = self.E_running
            sigma = self.S_running

        x_hat = (x - E)/(sigma + self.epsilon)
        self.out = self.gammas * x_hat + self.betas

        if self.track_running_stats:
            with torch.no_grad():
                self.E_running = (1.0 - self.momentum) * self.E_running + self.momentum * E
                self.S_running = (1.0 - self.momentum) * self.S_running + self.momentum * sigma
        
        return self.out
    
    def parameters(self):
        return [self.gammas, self.betas]
                
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

def build_dataset(data, symbols, prediction_block): 
    stoi = {s:i for i,s in enumerate(symbols)}
    # itos = {i:s for i,s in enumerate(symbols)}

    X = []
    Y = []

    for name in data:
        context = [0] * prediction_block
        for letter in name + '.':
            X.append(context)
            Y.append(stoi[letter])
            # print(''.join([itos[i] for i in context]),'-->', letter)
            context = context[1:] + [stoi[letter]]

    return torch.tensor(X), torch.tensor(Y)

def split_loss(split):
    x,y = {
            'train': (Xtr, Ytr),
            'dev': (Xdev, Ydev),
            'test': (Xt, Yt),
  }[split]

    with torch.no_grad():
        x = C[x].view(-1, repetition_block*emb_dim)
        for layer in layers:
            if isinstance(layer, BatchNorm1d):
                layer.track_running_stats = False
            x = layer(x)
    loss = F.cross_entropy(x, y)
    print(f"{split} loss is: {loss.item()}")


def sample(num_names):
    g = torch.Generator().manual_seed(2147483647 + 10)
    itos = {i:s for i,s in enumerate(symbols)}
    with torch.no_grad():
        for i in range(num_names):
            name = ''
            x = [0]*repetition_block

            while True:
                for layer in layers:
                    x = layer(x)

                probs = F.softmax(x, dim =1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                name += itos[ix]
                input = input[1:] + [ix]
                if ix ==0:
                    break
            print(name)

# %%
data = open('names.txt', 'r').read().splitlines()
symbols =['.'] + sorted(list(set(''.join(data))))
repetition_block = 3

random.seed(42)
random.shuffle(data)

n1 = int(len(data) * 0.8)
n2 = int(len(data) * 0.9)
Xtr, Ytr = build_dataset(data[:n1], symbols, repetition_block)
Xdev, Ydev = build_dataset(data[n1:n2], symbols, repetition_block)
Xt, Yt = build_dataset(data[n2:], symbols, repetition_block)

g = torch.Generator().manual_seed(2147483647)

emb_dim = 10
batch_size = 32
n_hidden = 200
C = torch.randn((len(symbols), emb_dim), generator=g)


# %%
layers = [
  Linear(emb_dim * repetition_block, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, len(symbols), bias=False), BatchNorm1d(len(symbols)),
]

# layers = [
#     Linear(emb_dim * repetition_block, n_hidden, False), BatchNorm1d(n_hidden,track_running_stats=True), Tanh(),
#     Linear(n_hidden, 27)
# ]

parameters =[C] + [p for layer in layers for p in layer.parameters()]

for parameter in parameters:
    parameter.requires_grad = True

with torch.no_grad():
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.w *= 5/3

# %%
max_steps = 1#200_000
for i in range(max_steps):
    ix = torch.randint(low=0, high=Xtr.shape[0], size=(batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]
    x = C[Xb].view(batch_size, -1)

    lr = 0.1 if i < 100_000 else 0.01

    for layer in layers:
        x = layer(x)

    loss = F.cross_entropy(x, Yb)
    if i % 10000 == 0: 
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

    for parameter in parameters:
        parameter.grad = None

    loss.backward()
    
    for parameter in parameters:
        parameter.data -= lr * parameter.grad
#%%
for layer in layers:
    if isinstance(layer, BatchNorm1d):
        layer.track_running_stats = False

split_loss('dev')

# %%
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
itos = {i:s for i,s in enumerate(symbols)}
with torch.no_grad():
    for i in range(20):
        name = ''
        input = [0]*repetition_block

        while True:
            x = C[torch.tensor(input)].view(1, repetition_block * emb_dim)
            for layer in layers: 
                x = layer(x)

            probs = F.softmax(x, dim =1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            name += itos[ix]
            input = input[1:] + [ix]
            if ix ==0:
                break
        print(name)

# %%
