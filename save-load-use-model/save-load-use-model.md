# Save, Load and Use the Model

We have trained the model! Now lets take a look at the way we can save, load and use the model we created.

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
```python
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = loaded_model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y.argmax(0)]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```