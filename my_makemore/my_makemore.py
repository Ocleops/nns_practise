#%%
import torch
import matplotlib.pyplot as plt
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
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');

#%%
g = torch.Generator().manual_seed(2147483647)
names = []
for i in range(20):
    line = N[0, :]
    name = ''
    while True:
        p = line / sum(line).item()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        letter = itos[ix]
        if letter == '.':
            break
        name += letter
        line = N[ix, :]
    # names.append(name)
    print(name)

# %%
