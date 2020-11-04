# Build the Model

The data has been loaded and transformed we can now build the model. We will leverage [torch.nn](https://pytorch.org/docs/stable/nn.html) predefined layers that Pytorch has that can both simplify our code, and  make it faster.

In the below example, for our FashionMNIT image dataset, we are using a `Sequential` container from class [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) that allows us to define the model layers inline. The neural network modules layers will be added to it in the order they are passed in.

Another way this model could be bulid is with a class using [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). We will break down each of these step below.

Inline Example:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# model
model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, len(classes)),
        nn.Softmax(dim=1)
    ).to(device)
    
print(model)
```

Class nn.Module Example:
```python
class NeuralNework(nn.Module):
    def __init__(self, x, y):
        super(NeuralNework, self).__init__()
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2  = nn.Linear(512, 512)        
        self.linear3  = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.output(x)
        return F.softmax(x, dim=1)
```
# Get Device for Training
Here we check to see if [torch.cuda](https://pytorch.org/docs/stable/notes/cuda.html) is available to use the GPU, else we will use the CPU. 

Example:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
```

# The Inline Model Modules in nn.Seqential

Lets break down each model layer in the FashionMNIST model.

## [nn.Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) to reduce tensor dimensions to one.
From the docs:
```
torch.nn.Flatten(start_dim: int = 1, end_dim: int = -1)
```

Here is an example using one of the training_data set items:

```python
tensor = training_data[0][0]
print(tensor.size())

Output: torch.Size([1, 28, 28])
```
```python
model = nn.Sequential(
    nn.Flatten()
)
flattened_tensor = model(tensor)
flattened_tensor.size()

Output: torch.Size([1, 784])
```

Flatten can also be done with a reshape as shown here:

```python
tensor = training_data[0][0]
print(tensor.size())

Output: torch.Size([1, 28, 28])
```
```python
tensor.reshape(-1)
print(tensor.size())

Output: torch.Size([1, 784])
```


## [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) to add a linear layer to the model.

Now that we have flattened our tensor dimension we will apply a linear layer transform that will calculate/learn the weights and the bias.

From the docs:
```
torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)

in_features – size of each input sample

out_features – size of each output sample

bias – If set to False, the layer will not learn an additive bias. Default: True
```

Lets take a look at the resulting data example with the flatten layer and linear layer added:

```python
input = training_data[0][0]
print(input.size())
model = nn.Sequential(
    nn.Flatten(),    
    nn.Linear(28*28, 512),
)
output = model(input)
output.size()

Output: 
torch.Size([1, 28, 28])
torch.Size([1, 512])
```
To print out learned weights and bias:
```python
model[1].weight
model[1].bias
```

## [nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) Activation
"Applies the rectified linear unit function element-wise"

## [nn.Softmax]() Activation
"Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1."

```python
Output:

Using cuda device
Sequential(
  (0): Flatten()
  (1): Linear(in_features=784, out_features=512, bias=True)
  (2): ReLU()
  (3): Linear(in_features=512, out_features=512, bias=True)
  (4): ReLU()
  (5): Linear(in_features=512, out_features=10, bias=True)
  (6): Softmax(dim=1)
)
```
# The Class Model


# Resources

[torch.nn](https://pytorch.org/docs/stable/nn.html)

# More help with the FashionMNIST Pytorch Blitz
[Tensors]()<br>
[DataSets and DataLoaders]()<br>
[Transformations]()<br>
[Building the Model]()<br>
[Optimization Loop and AutoGrad]()<br>
[Save, Load and Use Model]()<br>
[Back to FashionMNIST main code base]()<br>
