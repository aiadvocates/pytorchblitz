# Build Model

Now that the data has been loaded and transformed we can now build the model. There are many different way to sculp your model <!--stuff about building models here-->

In the below example, for our FashionMNIT image dataset, we are using a `Sequential` conntainer from class [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html). Modules will be added to it in the order they are passed in. We will break down each of these steps and the why below.

Full Section Example:

```python
# where to run
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

# cost function used to determine best parameters
cost = torch.nn.BCELoss()

# used to create optimal parameters
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
## Where to run

Example:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
```

## The Model
Example:
```python
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

## Cost Function and Paramters
Example:
```python
# cost function used to determine best parameters
cost = torch.nn.BCELoss()

# used to create optimal parameters
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```


## More help with the FashionMNIST Pytorch Blitz
[Tensors]()<br>
[DataSets and DataLoaders]()<br>
[Transformations]()<br>
[Choosing Model]()<br>
[Optimization Loop and AutoGrad]()<br>
[Save, Load and Use Model]()<br>
[Back to FashionMNIST main code base]()<br>
