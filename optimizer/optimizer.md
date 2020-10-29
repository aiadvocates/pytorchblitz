# Optimizing Model Parameters¶

Now that we have a model and data it's time to train, validate and test our model by optimizating it's paramerters on our data! 

To do this we need to understand a how to handle 5 core deep learning concepts in PyTorch

- Optimization Loops
- Loss Function
- AutoGrad
- Optimizers 
- Hyper Parameters (learning rates, batch sizes, epochs etc)

Let's dissect these concepts one by one and look at some code at the end we'll see how it all fits together.

## Optimizaton Loops

There are three main loops for model optimization in PyTorch. 
![](../images/optimization_loops.PNG)

 1. The Train Loop is used to update the model parameters
 2. The Validation is used to validate the model performance after a weight parameter update and can be used to gauge hyper parameter performance 
 3. The Test oop is used to evaluate our models performance on after training 


## Loss function 

## AutoGrad

## Optimizers 

## Hyper Parameters (learning rates, batch sizes, epochs etc)


## Loss Function

## Optimizer

  



## Hyper Parameters 

 parameters that we can give the optimization algorithm to tune how it trains - these are called hyper-parameters. That's what the two variables represent below:

learning_rate = 1e-3
batch_size = 64
epochs = 5


The learning_rate basically specifies how fast the algorithm will learn the model parameters. Right now you're probably thinking "let's set it to fifty million #amirite?" The best analogy for why this is a bad idea is golf. I'm a terrible golfist (is that right?) so I don't really know anything - but pretend you are trying to sink a shot (again sorry) but can only hit the ball the same distance every time. Easy right? Hit it the exact length from where you are to the hole! Done! Now pretend you don't know where the hole is but just know the general direction. Now the distance you choose actually matters. If it is too long a distance you'll miss the hole, and then when you hit it back you'll overshoot again. If the distance is too small then it will take forever to get there but for sure you'll eventually get it in. Basically you have to guess what the right distance per shot should be and then try it out. That is basically what the learning rate does for finding the "hole in one" for the right parameters (ok, I'm done with the golf stuff).

Below there are three things that make this all work:

The Optimizer - this part is the bit that actually changes the model parameters. It has a sense for the direction we should be shooting and updates all of the internal numbers inside the model to find the best internal knobs to predict the right digits. In this case I am using the Binary Cross Entropy cost function because, well, I know it works. There are a ton of different cost functions you can choose from that fit a variety of different scenarios.

```python
def train_model(model, device, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, run=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            batch = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                loss = loss.item()
                if phase == 'train':
                    if run != None:
                        run.log('{}_loss'.format(phase), loss)
                    print('	loss: {:>10f}  [{:>3d}/{:>3d}]'.format(loss, batch * len(inputs), dataset_sizes[phase]))

                running_loss += loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if run != None:
                run.log('{}_epoch_accuracy'.format(phase), epoch_acc.item())
                run.log('{}_epoch_loss'.format(phase), epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                scheduler.step()
                print(f'Current lr: {scheduler.get_last_lr()}')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if run != None:
        run.log('elapsed', time_elapsed)
        run.log('best_accuracy', best_acc)

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
```


- Train, Val Dev, Loops
Auto grad
- Forward, loss, step 
- Backward




## AutoGrad

https://towardsdatascience.com/the-pytorch-training-loop-3c645c56665a

https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e#cf51