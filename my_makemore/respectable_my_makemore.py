#%%
import torch.nn.functional as F
import torch
import random
import matplotlib.pyplot as plt
#%%
def extract_symbols(data):
    symbols = set()
    symbols.add('.')
    
    for name in data:
        for letter in name:
            symbols.add(letter)
    
    symbols = sorted(list(symbols))

    return symbols

def build_dataset(data, symbols, prediction_block): 
    stoi = {s:i for i,s in enumerate(symbols)}

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
symbols = extract_symbols(data)
repetition_block = 3
random.seed(42)
random.shuffle(data)

X, Y = build_dataset(data, symbols, repetition_block)
n1 = int(0.8*len(data))
n2 = int(0.9*len(data))

Xtr, Ytr = build_dataset(data[:n1], symbols, repetition_block)
Xdev, Ydev = build_dataset(data[n1:n2], symbols, repetition_block)
Xte, Yte = build_dataset(data[n2:], symbols, repetition_block)

# %%
n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((len(symbols),n_embd), generator=g)

#%%
W1 = torch.randn((n_embd * repetition_block, n_hidden), generator=g)
b1 = torch.rand((1,n_hidden), generator=g)

W2 = torch.randn((n_hidden, len(symbols)), generator=g)
b2 = torch.randn((1, len(symbols)), generator=g)

parameters = [C,W1,b1,W2,b2]

for p in parameters:
    p.requires_grad = True

# %%
# LEARNING RATE SEARCH
space = torch.linspace(-3, 0, 1000)
lrs = 10**space
losses = []

#%%
mbatch_size = 32
epochs = len(space)
for epoch in range(epochs):
    # IMPLEMENT MINIBATCH
    ix = torch.randint(0, Xtr.shape[0], (mbatch_size,), generator=g)
    emb = C[Xtr[ix]].view(Xtr[ix].shape[0], 
                          repetition_block * n_embd ) # DIMS = (32, 3, n_embd)


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
# plt.plot(space, losses)
# %%
# TRAIN LOSS
with torch.no_grad():
    tr_emnds = C[Xtr].view(len(Xtr), n_embd*repetition_block)
    tr_h = torch.tanh(tr_emnds @ W1 + b1)
    tr_logits = tr_h @ W2 + b2
    tr_loss = F.cross_entropy(tr_logits, Ytr)
    print(tr_loss.item())
#%%
# DEV LOSS
with torch.no_grad():
    dev_embds = C[Xdev].view(len(Xdev), n_embd*repetition_block)
    dev_h = torch.tanh(dev_embds @ W1 + b1)
    dev_logits = dev_h @ W2 + b2
    dev_loss = F.cross_entropy(dev_logits, Ydev)
    print(dev_loss.item())
# %%
# SAMPLE FROM THE MODEL
# ITERERATIVELY GIVE AN INPUT TO THE MODEL UNTIL YOU GET A '.'
itos = {i:s for i,s in enumerate(symbols)}
with torch.no_grad():
    for i in range(10):
        name = ''
        while True:
            input = [0.0]*repetition_block
            input_emb = C[input].view(1, repetition_block * n_embd)
            in_h = torch.tanh(input_emb @ W1 + b1)

            logits = in_h @ W2 + b2
            probs = F.softmax(logits, dim=1)

            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            name += itos[ix]
            if itos[ix] == '.':
                break
        print(name)

# %%
