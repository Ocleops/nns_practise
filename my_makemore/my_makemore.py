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
P = N / torch.sum(N, 1, keepdim=True) # Broadcasting CAREFULL!

g = torch.Generator().manual_seed(2147483647)
names = []
for i in range(20):
    p = P[0,:]
    name = ''
    while True:
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        letter = itos[ix]
        if letter == '.':
            break
        name += letter
        p = P[stoi[letter], :]
    print(name)

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
    break

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = F.one_hot(xs[0], num_classes=27).float()
yenc = F.one_hot(ys, num_classes=27).float()

W = torch.rand((27,27)) # for 1 neuron
activation = xenc @ W 
print(activation.shape)
# %%
