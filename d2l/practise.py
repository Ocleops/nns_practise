#%%
import d2l_utils
import torch
import matplotlib.pyplot as plt
import numpy as np
#%%
def attention_pulling(key, value, query, kernel):
    dist = torch.abs(query.view(-1,1) - key.view(1,-1))
    k = kernel(dist)
    attention = k / k.sum(1, keepdims = True)

    y_hat = attention @ value.view(-1, 1) 
    return y_hat, attention

def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l_utils.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = attention_pulling(key=x_train, value=y_train, query=x_val, kernel=kernel)
        # y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            pcm = ax.imshow(attention_w, cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5)
        ax.set_xlabel(name)

        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)

#%%
# torch.manual_seed(1337)

# Define some kernels using PyTorch
def gaussian(x):
    return torch.exp(-x**2 / 2)

def boxcar(x):
    return (torch.abs(x) < 1.0).float()

def constant(x):
    return 1.0 + 0 * x

def epanechikov(x):
    return torch.maximum(1 - torch.abs(x), torch.zeros_like(x))

# Define the noisy function
def f(x):
    return 2 * torch.sin(x) + x


kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')

n = 40
x_train = torch.sort(torch.rand(n) * 5)[0]
y_train = f(x_train) + torch.randn(n)

x_val = torch.arange(0, 5, 0.1)
y_val = f(x_val)

plot(x_train, y_train, x_val, y_val, kernels, names)


# %%
