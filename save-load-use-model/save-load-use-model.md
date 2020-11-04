# Save, Load and Use the Model

We have trained the model! Now lets take a look at how to save, load and use the model created.

Full Section Example:
```python
# create dummy variable to traverse graph
x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
onnx.export(model, x, 'model.onnx')
print('Saved onnx model to model.onnx')

# saving PyTorch Model Dictionary
torch.save(model.state_dict(), 'model.pth')
print('Saved PyTorch Model to model.pth')

draw_clothes(test_data)

#rehydrate model
loaded_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, len(classes)),
        nn.Softmax(dim=1)
    )

#load graph
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()

x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = loaded_model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y.argmax(0)]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```


## Save the Model
Example:
```python
# create dummy variable to traverse graph
x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
onnx.export(model, x, 'model.onnx')
print('Saved onnx model to model.onnx')

# saving PyTorch Model Dictionary
torch.save(model.state_dict(), 'model.pth')
print('Saved PyTorch Model to model.pth')
```

## Load the Model

Example:
```python
draw_clothes(test_data)
```
```python
loaded_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, len(classes)),
        nn.Softmax(dim=1)
    )
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()
```

## Test the Model
Example:
```python
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = loaded_model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y.argmax(0)]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

## More help with the FashionMNIST Pytorch Blitz
[Tensors]()<br>
[DataSets and DataLoaders]()<br>
[Transformations]()<br>
[Building the Model]()<br>
[Optimization Loop and AutoGrad]()<br>
[Save, Load and Use Model]()<br>
[Back to FashionMNIST main code base]()<br>
