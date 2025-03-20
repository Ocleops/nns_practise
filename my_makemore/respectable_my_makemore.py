#%%
import torch.nn.functional as F
import torch
import random
import matplotlib.pyplot as plt
#%%
def build_dataset(data, prediction_block = 3):
    symbols = set()
    symbols.add('.')
    
    for name in data:
        for letter in name:
            symbols.add(letter)
    
    symbols = sorted(list(symbols))
    
    stoi = {s:i for i,s in enumerate(symbols)}
    itos = {i:s for i,s in enumerate(symbols)}

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

def model_count_params(parameters):
    counts = 0
    for p in parameters:
        counts += p.shape[0] * p.shape[1]
    print("The number of your model parameters is: ", counts)

#%%
data = open('names.txt', 'r').read().splitlines()
random.seed(42)
random.shuffle(data)

X, Y = build_dataset(data)
n1 = int(0.8*len(data))
n2 = int(0.9*len(data))

Xtr, Ytr = build_dataset(data[:n1])
Xdev, Ydev = build_dataset(data[n1:n2])
Xte, Yte = build_dataset(data[n2:])
# %%
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator=g)

W1 = torch.randn((6,100), generator=g)
b1 = torch.rand((1,100), generator=g)

W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn((1, 27), generator=g)

parameters = [C,W1,b1,W2,b2]

for p in parameters:
    p.requires_grad = True

# %%
# LEARNING RATE SEARCH
space = torch.linspace(-3, 0, 1000)
lrs = 10**space
losses = []

#%%
epochs = len(space)
for epoch in range(epochs):
    # IMPLEMENT MINIBATCH
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    emb = C[Xtr[ix]].view(Xtr[ix].shape[0],6) # DIMS = (32, 3, 2)


    h = torch.tanh(emb @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    print(f'Epoch: {epoch+1} --> loss: {loss:.4f}')
    # losses.append(loss.item())

    for p in parameters:
        p.grad = None

    loss.backward()
    # DUE TO THE FACT THAT WE ARE NOT USING THE WHOLE DATASET, THE GRADIENTS 
    # CALCULATED HERE ARE AN APPROXIMATION OF THE REAL GRADIENTS
    # HOWEVER, IT IS MORE BENEFICIAL TO APPROXIMATE THE GRADIENTS AND MAKE MORE
    # BACKWARD PASSES THAN TO DETERMINE THE GRADIENT'S VALUE VERY ACCURETELY 
    # AND DO FEWER BACKWARD PASSES.

    lr = 0.25 if epoch < 100_000 else 0.025
    for p in parameters:
        # p.data -= lrs[epoch] * p.grad FOR LR SEARCH
        p.data -= lr * p.grad

# %%
# FOR DETERMINING THE LEARNING RATE, CREATE THE PLOT
# CHOOSE A LEARNING RATE THAT FALL IN THE VALLEY OF 
# THE MINIMUM OF THE GRAPH.
plt.plot(space, losses)
# %%
