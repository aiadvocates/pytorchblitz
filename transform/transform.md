# What are Transforms?

Data does not come ready to be fed into the machine learning algorithm. We need to do different data manipulations or transforms to prepare it for training. There are many types of transformations and it depends on the type of model you are building and the state of your data as to which ones you should use. 

In the below example, for our FashionMNIT image dataset, we are taking our image features (x), turning it into a tensor, normalizing and reshaping it. Then taking the labels (y) padding with zeros to get a consistent shape. We will break down each of these steps and the why below.

Example:

```python
clothing = datasets.FashionMNIST('data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.reshape(28*28))
                        ]),
                        target_transform= transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
)
```

# Lets break down whats happening in this sample code step-by-step.

## Pytorch Datasets
<!--TODO link to Ari' Dataset info-->
We are using the built-in open FashionMNIST datasets from the PyTorch library. For more info on the Datasets and Loaders check out [this]() resource.

From the docs:
```
torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)
```

## Compose
The `transforms.compose` allows us to string together different steps of transformations in a sequential order. This allows us to add an array of transforms for both the features and labels when preparing our data for training.

## Transform: Features
```python
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.reshape(28*28))
]),
```

### Transform: ToTensor() 
For the feature transforms we have an array of transforms to process our image data for training. The first transform in the array is `transforms.ToTensor()` this is from class [torchvision.transforms.ToTensor](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor). We need to take our images and turn them into a tensor. (To learn more about Tensors check out [this]() resource.) However the ToTensor() transformation is doing more than converting our image into a tensor. Its also normalizing our data for us by scaling the images to be between 0 and 1.

```
NOTE: ToTensor only normalized image data that is in PIL mode of (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8. In the other cases, tensors are returned without scaling.
```


### Transform: Lambda Reshape

The second transform in the array is the `transform.Lambda` which applys a user-defined lambda as a transform. The lambda function we created uses numpy to reshape the images from the 28 x 28 pixel grayscale arrays and flatten them so it becomes a row of 784 pixels of numbers for each image. 

From the docs:
```
transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
```

## Target_Transform: Labels
```python
target_transform= transforms.Lambd(lambda y: torch.zeros(10, dtype=torchfloat).scatter_(dim=0, index=torchtensor(y), value=1))
```
This function is taking the y input and creating a tensor of size 10 with a float datatype. Then its calling scatter to send each item to the following indices in torch.zeros, according to ROW-WISE (dim 0).

First we are setting a default output size of 10.

Second we are telling it the output type we want with is a float.

From the docs:
```
target_transform (callable, optional) – A function/transform that takes in the target and transforms it.
```

Then we are calling scatter.
[torch.Tensor.scatter_ class](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_)

Writes all values from the tensor src into self at the indices specified in the index tensor. For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.

Params
* dim (int) – the axis along which to index

* index (LongTensor) – the indices of elements to scatter, can be either empty or the same size of src. When empty, the operation returns identity

* src (Tensor) – the source element(s) to scatter, incase value is not specified

* value (float) – the source element(s) to scatter, incase src is not specified


## Resources

Check out the other TorchVision Transforms available: https://pytorch.org/docs/stable/torchvision/transforms.html