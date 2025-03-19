#%%
import torch.nn.functional as F
import torch
# %%
data = open("names.txt", "r").read().splitlines()
print(data)
#%%
# MAPPING
symbols = set()
symbols.add('.')
for name in data:
   for l in name:
      symbols.add(l)

symbols = sorted(list(symbols))
stoi = {s:i for i,s in enumerate(symbols)}
itos = {i:s for i,s in enumerate(symbols)}

#%%
# TAKE 3 LETTERS INPUT
# GUESS THE 4TH
prediction_block = 3
X,Y = [], []
for name in data[:4]:
    name += '.'
    # print(name)
    context = [0] * prediction_block 
    for ch in name:
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), "-->", ch)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
# %%
# BUILD THE EMBEDDINGS
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator=g)
# print(C)
print(C[X][0][0])   # SO HERE IS WHAT HAPPENS IS THAT WE CREATE A 3D TENSOR.
                    # C[X][0] RETURNS THE 3 EMBEDDINGS THAT ARE USED AS TRAINING
                    # EXAMPLE FOR PREDICTING THE FIRST LABEL.
                    # C[X][0][0] RETURNS THE EMBEDDING OF THE LETTER AT X[0][0]

# %%
print(C.shape)
# %%
print(X.shape)
# %%
print(C[X].shape)
# %%
print(X)
# %%
