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
for name in data:
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
print(X.shape)
# %%
# BUILD THE EMBEDDINGS
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator=g) # EMBEDDINGS MATRIX
# print(C)
print(C[X][0][0])   # SO HERE IS WHAT HAPPENS IS THAT WE CREATE A 3D TENSOR.
                    # C[X][0] RETURNS THE 3 EMBEDDINGS THAT ARE USED AS TRAINING
                    # EXAMPLE FOR PREDICTING THE FIRST LABEL.
                    # C[X][0][0] RETURNS THE EMBEDDING OF THE LETTER AT X[0][0]

# %%
# BUILDING THE FIRST LAYYER.
# --> TAKE THE INPUT X AND PASS IT THROUGH THE EMBEDDING MATRIX C(X)
# --> PERFORM THE LAYER'S OPPERATION C(X) * W + B
# NOTES:
# TENSOR X HAS SHAPE (# EXAMPLES, PREDICTION_BLOCK) --> MEANING THAT EACH TRAINING EXAMPLE CONTAINS 3 LETTERS 
#                                                       (DEPENDING ON THE NUMBER OF LETTERS USED FOR THE PREDICTION)


# MATRIX C IS THE LOOKUP MATRIX. C.SHAPE = (27, 2) --> BECAUSE WE HAVE 27 DISTINCT SYMBOLS IN THE DATASET
#                                                      AND BECAUSE WE ARE USING 2 DIMENSIONAL VECTORS

# SO AFTER EMBEDDING X WE GET A TENSOR OF SHAPE (# EXAMPLES, 3, 2)

emb = C[X].view(X.shape[0],6)

# NOW WE HAVE TO OEXECUTE THE HIDDEN LAYER'S OPERATION emb * W + b
# WE HAVE 3 LETTERS EMBEDDED WITH 2 DIMM VECTORS.
# SO WE HAVE 2X3 = 6 NUMBERS IN TOTAL IN THE INPUT LAYER.
# THE MATRIX W SHOULD THEN BE OF SHAPE (# NEURONS, 6)

W = torch.randn((6,100), generator=g)
b = torch.rand((1,100), generator=g)
h = torch.tanh(emb @ W + b)



# OUTPUT LAYER

W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn((1, 27), generator=g)
logits = h @ W2 + b2 # REMEMBER THIS REPRESENTS log(counts)
                     # WE NEED TO EXTRACT PROBABILITIES

parameters = [C,W,b,W2,b2]

#%%

counts = logits.exp()
probs = counts / torch.sum(counts, dim=1, keepdim= True)


# LOSS FUNCTION: NEGATIVE LOG LIKELIHOOD
# LOOK AT YOUR LABBELS AND FIND THE PROBABILITY OF THE LETTER THAT SHOULD HAVE BEEN PREDICTED.
# GATHER ALL THESE PROBABILITIES AND TAKE THEIR LOG
# ADD ALL LOGS
# TAKE THE NEGATIVE OF THAT

ps = probs[torch.arange(len(Y)), Y] # TAKES THE VALUES FROM probs
                                    # DICTATED BY THE ELEMENTS OF THE Y
                                    # AND ASSIGNS THEM TO ps
lgs = torch.log(ps)

loss = - lgs.mean()
print(loss)

# HOWEVER, THERE IS A BUILT IN FUNCTIONS THAT TAKES CARE
# OF ALL THE THINGS WE DID ABOVE MORE EFFICIENTLY.
# torch.nn.functional.cross_entropy TAKES INPUT THE LOGITS AND THE 
# TARGETS. IT AUTOMATICALLY RETURNS THE LOSS FUNCTION

loss2 = F.cross_entropy(logits, Y)
print(loss2)

# %%
# IMPLEMENT TRAINING LOOP

for p in parameters:
    p.requires_grad = True

epochs = 100
for epoch in range(epochs):
    emb = C[X].view(X.shape[0],6)
    h = torch.tanh(emb @ W + b)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    print(f'Epoch: {epoch+1} --> loss: {loss:.4f}')

    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data -= 0.1 * p.grad

# %%
