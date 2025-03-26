#%%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
# %matplotlib inline
# %%
words = open('names.txt', 'r').read().splitlines()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])
# %%
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)
# %%
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%
# %%
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
# %%
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun, it's useless because of BN
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

# Note: I am initializating many of these parameters in non-standard ways
# because sometimes initializating with e.g. all zeros could mask an incorrect
# implementation of the backward pass.

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
# %%
batch_size = 32
n = batch_size # a shorter variable also, for convenience
# construct a minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
# %%
# forward pass, "chunkated" into smaller steps that are possible to backward one at a time

emb = C[Xb] # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
# Linear layer 1
hprebn = embcat @ W1 + b1 # hidden layer pre-activation
# BatchNorm layer
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# Non-linearity
h = torch.tanh(hpreact) # hidden layer
# Linear layer 2
logits = h @ W2 + b2 # output layer
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
probs = counts * counts_sum_inv
logprobs = probs.log()

loss = -logprobs[range(n), Yb].mean()

# PyTorch backward pass
for p in parameters:
  p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, # afaik there is no cleaner way
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
         bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
         embcat, emb]:
  t.retain_grad()
loss.backward()
loss
print(loss.item())

# %%
dlogprobs  = torch.zeros((logprobs.shape[0], logprobs.shape[1]))
dlogprobs [range(n), Yb] += -1/n
dprobs = 1.0/probs * dlogprobs 
dcounts_sum_inv = torch.sum((dprobs * counts), dim= 1, keepdim=True)
# TO CALCULATE dcounts NOTICE THAT counts_sum_inv IS ALSO A FUNCTION OF 
# counts. SO YOU WILL HAVE TO CALCULATE dcounts_sum_inv/dcounts FIRST!!
dcounts_sum = - counts_sum**-2 * dcounts_sum_inv
dcounts1 = dcounts_sum * torch.ones((counts.shape[0], counts.shape[1]))
dcounts2 = counts_sum_inv * dprobs
dcounts = dcounts1 + dcounts2 

dnorm_logits = norm_logits.exp() * dcounts
# THE dlogits SUFFERS FROM THE SAME CONDITION AS dcounts. 
# WE HAVE TO CALCULATE ALL THE GRADIENTS ALL THE WAY AGAIN.
dlogits = dnorm_logits.clone() 
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits += F.one_hot(logits.max(1).indices, num_classes=27) * dlogit_maxes

# LINEAR LAER BACKWARD
db2 = dlogits.sum(dim=0)
dW2 = h.T @ dlogits
dh = dlogits @ W2.T
# BATCH NORM LAYER
dhpreact = (1.0 - h**2) * dh
dbngain = (dhpreact * bnraw).sum(dim=0, keepdims = True)
dbnraw = bngain * dhpreact
dbnbias = dhpreact.sum(dim=0, keepdim = True)

dbnvar_inv = (dbnraw * bndiff).sum(dim = 0, keepdim = True) 
dbndiff = dbnraw * bnvar_inv
dbnvar = - (0.5 * ((bnvar + 1e-5)**-1.5)) * dbnvar_inv
# dbndiff2 = dbnvar.sum(0, keepdim = True) * 1/(n-1) 
dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
dbndiff += (2*bndiff) * dbndiff2
dhprebn = dbndiff.clone() # IF YOU DO NOT USE CLONE, BUGS ARE COMING UP!!!!
dbnmeani = (- dbndiff).sum(dim=0, keepdim=True)
dhprebn += 1/n * dbnmeani 
dW1 = embcat.T @ dhprebn
dembcat = dhprebn @ W1.T
db1 = dhprebn.sum(dim=0)

# TO GET THE GRADIENTS FOR THE EMBEDINGS, THINK LIKE THIS.
# FIRST UNDERSTAND HOW PYTORCH EXTRACTS THE TENSOR THAT YOU WANT WITH THE COMMAND C[Xb].
# AS YOU CAN SEE BELOW, IT IS ESSENTIALLY A MATRIX MULTIPLICATION.
# YOU KNOW HOW TO BACKPROPAGATE THROUGH MATRIX MULTIPLICATIONS.
# SO THE REST ARE HISTORY
hot_vec = F.one_hot(Xb, num_classes=27).float()
# result = (hot_vec.view(-1, vocab_size) @ C)#.view(n, block_size, n_embd)
dC = hot_vec.view(-1, vocab_size).T @ dembcat.view(n*block_size,n_embd) 


#%%
cmp('logprobs', dlogprobs , logprobs)
cmp('probs', dprobs, probs)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
cmp('counts_sum', dcounts_sum, counts_sum)
cmp('counts', dcounts, counts)
cmp('norm_logits', dnorm_logits, norm_logits)
cmp('logit_maxes', dlogit_maxes, logit_maxes)
cmp('logits', dlogits, logits)
cmp('b2', db2, b2)
cmp('W2', dW2, W2)
cmp('h', dh, h)
cmp('hpreact', dhpreact, hpreact)
cmp('bngain', dbngain, bngain)
cmp('bnraw', dbnraw, bnraw)
cmp('bnbias', dbnbias, bnbias)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
cmp('bndiff', dbndiff, bndiff)
cmp('bnvar', dbnvar, bnvar)
cmp('bndiff2',dbndiff2, bndiff2)
cmp('hprebn', dhprebn, hprebn)
cmp('bnmeani', dbnmeani, bnmeani)
cmp('W1', dW1, W1)
cmp('embcat', dembcat, embcat)
cmp('b1', db1, b1)
cmp('C', dC, C)
# %%
# CALCULATE THE LOSS OF THE CROSS ENTROPY DIRECTLY
dlogits = 1/n * (F.softmax(logits, dim=1))
dlogits[torch.arange(n), Yb] -= 1/n 
cmp('logits', dlogits, logits)

#%%
# CALCULATE THE BACKPROP FOR THE BATCH NORM

# INPUT --> hprebn
# OUTPUT --> hpreact
# μ_Β = mean_over_minibatch(input)
# calculate variance of the minibatch
# x_hat = (x-μ)/sqrt(var - ε)
# y <-- γx_hat + β (γ,β learnable parameters)

# FIND dhprebn when you know hpreact

# y <-- γ* ((hprebn-μ)/sqrt(var - ε)) + β

# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = bngain * bnraw + bnbias

# dhprebn = bngain*bnvar_inv/n * (n * dhpreact - hpreact.sum(dim=0, keepdim = True) - n/(n-1) * bnraw * (dhpreact * bnraw).sum(dim=0, keepdim = True)) 

dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0))

cmp('hprebn', dhprebn, hprebn)


# %%

# BRING IT ALL TOGETHER

n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun, it's useless because of BN
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

# Note: I am initializating many of these parameters in non-standard ways
# because sometimes initializating with e.g. all zeros could mask an incorrect
# implementation of the backward pass.

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

#%%
max_steps = 1#200000
batch_size = 32
n = batch_size 

with torch.no_grad():
  for i in range(max_steps):
    # CREATE MINIBATCHES
    ix = torch.randint(low=0,high=Xtr.shape[0],size=(n,))
    Xb = Xtr[ix]
    Yb = Ytr[ix]

    # BEGIN FORWARD PASS
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    # LINEAR LAYER
    hprebn = embcat @ W1 + b1
    # BATCH NORM
    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=True)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # NON LINEARITY
    h = torch.tanh(hpreact)
    # OUTPUT LAYER
    logits = h @ W2 + b2
    # LOSS FUNCTION
    loss = F.cross_entropy(logits, Yb)

    # NOW BEGIN BACKWARD

    # loss.backward() .!.

    dlogits = 1/n * (F.softmax(logits, dim=1))
    dlogits[torch.arange(n), Yb] -= 1/n 

    # BACKWARD OUTPUT LAYER
    db2 = dlogits.sum(dim=0)
    dW2 = h.T @ dlogits
    dh = dlogits @ W2.T

    # BACKWARD NON LINEARITY
    dhpreact = (1.0 - h**2) * dh

    # BACKWARD BATCH NORM
    dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0))
    dbngain = (dhpreact * bnraw).sum(dim=0, keepdims = True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    # BACKWARD LINEAR LAYER
    dW1 = embcat.T @ dhprebn
    dembcat = dhprebn @ W1.T
    db1 = dhprebn.sum(dim=0)

    hot_vec = F.one_hot(Xb, num_classes=27).float()
    dC = hot_vec.view(-1, vocab_size).T @ dembcat.view(n*block_size,n_embd) 

    grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

    lr = 0.1 if i < 100000 else 0.01

    for p, grad in zip(parameters, grads):
      p.data += -lr * grad

    if i % 10000 == 0: 
      print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

# %%
# CALCULATE FULL TRAINING LOSS

with torch.no_grad():
  emds = C[Xtr]
  embcat = emds.view(emds.shape[0], -1)
  hprebn = embcat @ W1 + b1
  bnmean = hprebn.mean(0, keepdim=True)
  bnvar = hprebn.var(0, keepdim=True, unbiased=True)
  bnvar_inv = (bnvar + 1e-5)**-0.5
  bnraw = (hprebn - bnmean) * bnvar_inv
  hpreact = bngain * bnraw + bnbias
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Ytr)
  print(f'Training loss is: {loss.item():.4f}')

# %%
# CALCULATE DEV LOSS
with torch.no_grad():
  emds = C[Xdev]
  embcat = emds.view(emds.shape[0], -1)
  hprebn = embcat @ W1 + b1
  bnmean = hprebn.mean(0, keepdim=True)
  bnvar = hprebn.var(0, keepdim=True, unbiased=True)
  bnvar_inv = (bnvar + 1e-5)**-0.5
  bnraw = (hprebn - bnmean) * bnvar_inv
  hpreact = bngain * bnraw + bnbias
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Ydev)
  print(f'Dev loss is: {loss.item():.4f}')

# %%
# SAMPLE FROM THE MODEL

num_names = 20
g = torch.Generator().manual_seed(2147483647 + 10)
for i in range(num_names):
  name = ''
  input = [0, 0, 0]
  while True:
    emb = C[torch.tensor(input)]
    embcat = emb.view(1, -1)
    hprebn = embcat @ W1 + b1
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2

    probs = logits.softmax(dim=1)
    ix = torch.multinomial(probs, 1, replacement=True, generator=g).item()
    name += itos[ix]
    input = input[1:] + [ix]

    if ix ==0:
      break

  print(name)

# %%
