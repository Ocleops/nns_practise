#%%
import numpy as np
import random
#%%
random.seed(45)

class Value:
    def __init__(self, value, _children = ()):
        self.value = value
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda:None
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(value=self.value + other.value)
        out._prev.add(self)
        out._prev.add(other)

        def _backward():
            self.grad += out.grad 
            other.grad += out.grad

        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(value=self.value * other.value)
        out._prev.add(self)
        out._prev.add(other)

        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value

        out._backward = _backward

        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return -1 * self
    
    def __rsub__(self,other):
        return self + (-other)

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other): # Double check this when backprob
        assert isinstance(other, (int, float)), 'other must be int or float'

        out = Value(value=self.value**other)
        out._prev.add(self)

        def _backward():
            self.grad += other*(self.value**(other - 1)) * out.grad
        
        out._backward = _backward

        return out
     
    def tanh(self):
        out = Value(value=np.tanh(self.value))
        out._prev.add(self)

        def _backward():
            t = self.value
            self.grad += out.grad * (1 - (np.tanh(t))**2)

        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        for node in reversed(topo):
            node._backward()
        
#%%
class Neuron:
    def __init__(self, input_dim):
        self.weights = [Value(random.uniform(-1, 1)) for i in range(input_dim)]
        self.bias = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        result = 0
        for i in range(len(x)):
            result += self.weights[i] * x[i]
        result += self.bias
        result = result.tanh()
        return result
    
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:
    def __init__(self, input_dim, num_neurons):
        self.neurons = [Neuron(input_dim=input_dim) for neuron in range(num_neurons)]

    def __call__(self, x):
        results = []
        for neuron in self.neurons:
            results.append(neuron(x))
        
        return results
    
    def parameters(self):
        return[p for neuron in self.neurons for p in neuron.parameters()]
    
class Network:
    def __init__(self, arch, input_dim):
        self.layers = [Layer(input_dim=input_dim, num_neurons=arch[i]) if i==0 else Layer(input_dim=arch[i-1], num_neurons=arch[i]) for i in range(len(arch))]

    def __call__(self, x):
        for layer in self.layers:
            result = layer(x)
            x = result
        return result
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

# %%
n = Network([4,4,1], 3)
#%%
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets


# Train Loop!
epochs = 100
for i in range(epochs):
    preds = [n(x)[0] for x in xs]
    loss = sum([(y - pred)**2 for y,pred in zip(ys, preds)])

    print(f'epoch: {i} --> loss: {loss.value}')

    for p in n.parameters():
        p.grad = 0.0

    loss.grad = 1.0
    loss.backward()

    for p in n.parameters():
        p.value += - 0.1 * p.grad

# %%
# Print Predictions
for pred in preds:
    print(pred.value)
# %%
