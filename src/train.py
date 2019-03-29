import time
import copy

from tqdm import tqdm_notebook as tqdm
import torch


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """
    Train the supplied model with the given data, loss criterion, optimizer, and scheduler
    
    :param model:  The PyTorch model whose parameters should be trained
    :dataloaders:  The PyTorch DataLoaders (one for "train" one for "validation") that will supply the data
    :criterion:    The loss function to minimize
    :optimizer:    The algorithm to use to update the parameters to optimize the loss function
    :scheduler:    The object that dynamically adjusts the learning rate throughout the training process
    :num_epochs:   The number of epochs (i.e. the number of full loops through the training set) to run
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validate']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = torch.zeros(1, dtype=torch.double)
            running_corrects = torch.zeros(1, dtype=torch.int)
            
            num_iter_per_epoch = len(dataloaders[phase])
            progress_bar = tqdm(enumerate(dataloaders[phase]), total=num_iter_per_epoch)
            
            # Iterate over data.
            for batch_idx, (inputs, labels) in progress_bar:
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
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                

            epoch_loss = running_loss / num_iter_per_epoch
            epoch_acc = running_corrects.double() / num_iter_per_epoch

            print('{} Loss: {} Acc: {}'.format(
                phase, str(epoch_loss.numpy()), str(epoch_acc.numpy())))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model