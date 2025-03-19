#%%
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
%matplotlib inline

#%%
with open("names.txt", "r") as f:
    data = f.read().splitlines()

chs = []
b = {}
symbols = set()
for name in data:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        b[ch1 + ch2] = b.get(ch1 + ch2, 0)
        b[ch1 + ch2] += 1
        symbols.add(ch1)
        symbols.add(ch2)

# %%
symbols = sorted(list(symbols))
N = torch.zeros((len(symbols), len(symbols)), dtype = int)

stoi = {s:i for i,s in enumerate(symbols)}
itos = {i:s for i,s in enumerate(symbols)}

for name in data:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        i = stoi[ch1]
        j = stoi[ch2]
        N[i,j] += 1
#%%
############################## NICE PRINT OF THE TABLE ####################################################
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
#%%
P = (N + 1) / torch.sum(N, 1, keepdim=True) # Broadcasting CAREFULL!

g = torch.Generator().manual_seed(2147483647)
for i in range(20):
    p = P[0,:]
    name = ''
    while True:
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        letter = itos[ix]
        if letter == '.':
            break
        name += letter
        p = P[ix, :]
    print(name)

#%%
##################################### COST FUNCTION ############################################
log_likelihood = 0.0
norm = 0
for name in data:
    chs = ['.'] + list('andrejq') + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        i = stoi[ch1]; j = stoi[ch2]
        p = P[i, j]
        logp = torch.log(p).item()
        log_likelihood += -logp
        norm += 1
    break
print(log_likelihood/norm)

# %%
########################################### NEURAL NETWORK APPROACH ################################
# goal --> predicting next letter, based on the previous letter.
# step 1: isolate 1 name (emma)
# create all the bigrams 
# try to complete goal
xs = []
ys = []

for name in data:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])


xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = F.one_hot(xs, num_classes=27).float()
yenc = F.one_hot(ys, num_classes=27).float()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True) 

#%%
epochs = 200
for epoch in range(epochs):
    logits = xenc @ W 
    counts = logits.exp()

    P = counts / torch.sum(counts, 1, True)

    # Now I need to calculate a cost function!!!
    # How do I do that???
    # You have your ys, right?
    # P is your tensor that stores probabilities.
    # The cost function will be the negative log likelihood.
    # Strategy:
    # --> Take the label from the ys. 
    # --> Take the assigned probability from the neural network. 
    # --> Calculate the negative log likelihood.
    # --> Save the calculated value.

    probs = P[torch.arange(len(ys)), ys]
    nlgs = -torch.log(probs)


    loss = nlgs.mean() + 0.01 * (W**2).mean()
    print(f'Epoch {epoch} --> Loss {loss}')

    W.grad = None
    loss.backward()

    W.data -= 50 * W.grad

# %%
# SAMPLE FORM THE MODEL #
# What you have in your hands is the matrix W.
# You have to kind of start another forward pass now.
for i in range(20):
    idx = 0
    name = ''
    while True:
        xenc = F.one_hot(torch.tensor([idx]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        name += itos[ix]
        idx = ix
        if ix == 0:
            break
    print(name)

# %%
