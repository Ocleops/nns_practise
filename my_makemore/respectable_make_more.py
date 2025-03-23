#%%
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
#%%

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

def act_sanity_check(h, limit = 0.99):
    plt.figure(figsize=(16,9))
    plt.subplot(221)
    plt.hist(h.view(-1).tolist(), bins=50)
    plt.subplot(222)
    plt.imshow(h.abs()>0.99, cmap='grey') # WHITE IS TRUE
    print((h.view(-1).abs() > 0.99).float().mean())

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

# %%
# NETWORK INITIALIZATION
g = torch.Generator().manual_seed(2147483647)

emb_dim = 10
mini_batch = 32
n_hidden = 200

C = torch.randn((len(symbols), emb_dim), generator=g) # GIVE A SYMBOL, TAKE AN EMBEDDING
W1 = torch.randn((emb_dim * repetition_block, n_hidden), generator=g) * (5/3) /(emb_dim * repetition_block)**0.5
# b1 = torch.randn((1, n_hidden), generator=g) # IN BATCH NORM b1 IS USELESS BECAUSE WE ARE
                                               # ADDING A DIFFERENT BIAS 
W2 = torch.randn((n_hidden, len(symbols)), generator=g) * 0.01
gammas = torch.randn((1, n_hidden), generator=g)
betas = torch.randn((1,n_hidden), generator=g)

E_running = torch.randn((1, n_hidden), generator=g) # SHIFT
S_running = torch.randn((1, n_hidden), generator=g) # GAIN

parameters = [C,W1, W2, betas, gammas] # PUT b1 BACK HERE IF YOU UNCOMMENT IT.
for p in parameters:
  p.requires_grad = True
#%% 
max_steps = 200_000
for i in range(max_steps):
# FORWARD PASS
    ix = torch.randint(low=0, high=Xtr.shape[0], size=(mini_batch,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]
    embds = C[Xb]

    hpre = embds.view(mini_batch, -1) @ W1 # + b1

    # IMPLEMENTING BATCH NORM
    E = torch.mean(hpre, dim=0, keepdim=True)
    sigma = torch.std(hpre, dim=0, keepdim=True)
    x_hat = (hpre - E)/sigma

    y = gammas * x_hat + betas
    h = torch.tanh(y)
    # act_sanity_check(h, limit = 0.99)

    with torch.no_grad():
        E_running = 0.999 * E_running + 0.001 * E 
        S_running = 0.999 * S_running + 0.001 * sigma

    logits = h @ W2
    loss = F.cross_entropy(logits, Yb)
    # # I AM ALSO INCLUDING THE MANUAL IMPLEMENTATION OF THE LOSS HERE!!!
    # # APPLY SOFTMAX
    # probs = logits.exp()/torch.sum(logits.exp(), dim=1, keepdim=True)
    # # CALCULATE LOSS
    # lgs = torch.log(probs[torch.arange(mini_batch), Ytr[ix]])
    # loss = - lgs.mean()
    # print(loss)

    if i % 10000 == 0: 
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

    # BACKWARD PASS

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 100_000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

# %%
# TRAIN LOSS
with torch.no_grad():
    tr_emnds = C[Xtr].view(len(Xtr), emb_dim*repetition_block)
    tr_hpre = tr_emnds @ W1 
    xtr_hat = (tr_hpre - E_running)/S_running
    ytr = gammas * xtr_hat + betas
    tr_h = torch.tanh(ytr)

    tr_logits = tr_h @ W2 
    tr_loss = F.cross_entropy(tr_logits, Ytr)
    print(tr_loss.item())

# %%
# DEV LOSS
with torch.no_grad():
    dev_embds = C[Xdev].view(len(Xdev), emb_dim*repetition_block)
    dev_hpre = dev_embds @ W1 
    xdev_hat = (dev_hpre - E_running)/S_running
    ydev = gammas * xdev_hat + betas
    dev_h = torch.tanh(ydev)
    
    dev_logits = dev_h @ W2 
    dev_loss = F.cross_entropy(dev_logits, Ydev)
    print(dev_loss.item())
# %%
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

itos = {i:s for i,s in enumerate(symbols)}
with torch.no_grad():
    for i in range(20):
        name = ''
        input = [0]*repetition_block
        while True:
            emb = C[input]
            input_hpre = emb.view(1, -1) @ W1 
            input_hat = (input_hpre - E_running)/S_running
            input_y = gammas * input_hat + betas
            input_h = torch.tanh(input_y)

            logits = input_h @ W2
            probs = F.softmax(logits, dim =1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            name += itos[ix]
            input = input[1:] + [ix]
            if ix ==0:
                break
        print(name)
# %%
print(E_running)
# %%
print(E)
# %%
