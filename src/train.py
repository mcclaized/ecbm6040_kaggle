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
    
    model = model.to(device)
    
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

            num_samples = torch.zeros(1, dtype=torch.int)
            running_loss = torch.zeros(1, dtype=torch.double)
            running_corrects = torch.zeros(1, dtype=torch.int)
            
            num_iter_per_epoch = len(dataloaders[phase])
            progress_bar = tqdm(enumerate(dataloaders[phase]), total=num_iter_per_epoch)
            
            # Iterate over data.
            for batch_idx, (inputs, labels) in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1,1).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                num_samples += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((outputs >= 0.5).float() == labels.data)
                
                if batch_idx and not batch_idx % 200 and phase == "train":
                    print(
                        'Epoch: {} Training Batch: {} Loss: {} Acc: {}'.format(
                            epoch, batch_idx, 
                            float(running_loss) / float(num_samples),
                            float(running_corrects) / float(num_samples)
                        )
                    )
                

            epoch_loss = float(running_loss) / float(num_samples)
            epoch_acc = float(running_corrects) / float(num_samples)

            print(
                '{} Loss: {} Acc: {}'.format(
                    phase, epoch_loss, epoch_acc
                )
            )

            # deep copy the model
            if phase == 'validate' and epoch_acc > best_acc:
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

def validate_model(model, validation_dataloader, num_batches=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_inputs = []
    all_labels = []
    all_preds = []
    for idx, (inputs, labels) in enumerate(validation_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):  # No tracking during validation
            outputs = model(inputs)
        preds = (outputs >= 0.5).squeeze().long()
        all_inputs.append(inputs)
        all_labels.append(labels)
        all_preds.append(preds)
        if num_batches is not None and idx == (num_batches - 1):
            break
    
    return torch.cat(all_inputs).cpu(), torch.cat(all_labels).cpu(), torch.cat(all_preds).cpu()
    