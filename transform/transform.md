# Transforms

## What are they?

Data does not come ready to be fed into the machine learning algorithm. We need to do different data manipulations or transforms to prepare it for training. There are many types of transformations and it depends on the type of model you are building and the state of your data as to which ones you should use. 

In the below example, for our FashionMNIT image dataset, we are taking our image features (x), turning it into a tensor, reshaping it. Then taking the labels (y) padding with zeros to get a consistent shape, sctter?,  and then padding it with zeros. We will break down each of these steps and the why below.

Example:

```python
clothing = datasets.FashionMNIST('data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.reshape(28*28))
                        ]),
                        target_transform= transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

# Lets break down whats happening in this sample code step-by-step.

## Pytorch Datasets
<!--TODO link to Ari' Dataset info-->
We are using the built-in open FashionMNIST datasets from the PyTorch library. For more info on the Datasets and Loaders check out [this]() resource.

## Compose
The `transforms.compose` allows us to string together different steps of transformations in a sequential order. This allows us to add an array of transforms for both the features and labels when preparing our data for training.

## Transform: is for the features
For the feature transforms we have an array of transforms to process our image data for training. The first transform is the array is `transforms.ToTensor()`. We need to take our images and turn them into a tensor. To learn more about Tensors check out [this]() resource.

The second transform in the array is the `transform.Lambda` which takes a lambda function and reshapes the images to a consistent size.

### Target_Transform: is for the labels

The function is doing quite a bit for us so lets take a look inside and see whats happening:

First we are setting a default output size of 10.

Second we are telling it the output type we want with is a float.

Then we are calling scatter.
parma dim=0 means dont change index of y row

From the docs:

```
torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)

transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop

target_transform (callable, optional) – A function/transform that takes in the target and transforms it.
```


## Resources

Check out the other TorchVision Transforms available: https://pytorch.org/docs/stable/torchvision/transforms.html