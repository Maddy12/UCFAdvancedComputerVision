import multiprocessing
import time
import os
import shutil
import pdb
from  progress.bar import Bar
from datetime import datetime
from multiprocessing import cpu_count

# Pytorch
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter

# Local
from utils.metrics import *
from utils.model import MyModel, AlexNetFine
from utils.datasets import TinyImagenet, SVHN


def run_experiment(dataset, data_path, num_conv=3, fine_tune=False, perc_class=100, batch_size=128, epochs=100, num_classes=200, train=True, transform=None): 
    """
    Args: 
        num_conv: The number of graphical convolutions to run on the graphical representation of the ResNet output. 
        dataset: The dataset being used: either tiny imagenet or SVHN
        data_path: The root path of the data
        perc_class: The percentage of samples per class, for altering the data size. 
        epochs (int): the number of epochs to run for training, default is None
        num_classes: the number of classes
    """
    
    # Set up the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if fine_tune:
        if dataset == 'imagenet':
            nclasses = 200
        elif dataset == 'svhn':
            nclasses = 10
        model = AlexNetFine(nclasses)
        exp_name = 'finetune_alexnet'
        model = model.to(device)
    else:
        model = MyModel(num_conv)
        exp_name = 'convs'+str(num_conv)+'_perc'+str(perc_class)
        model = model.to(device)
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0

    # Load the data
    data_obj = {'imagenet': TinyImagenet, 'svhn': SVHN}
    train_dataset = data_obj[dataset](perc_per_class=perc_class, base_dir=data_path, split='train', transform=transform)
    val_dataset = data_obj[dataset](perc_per_class=perc_class, base_dir=data_path, split='val', transform=transform)
    dataloader = dict()
    dataloader['training'] = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=cpu_count(), shuffle=True)
    dataloader['training_length'] = len(train_dataset)
    dataloader['testing'] = DataLoader(dataset=val_dataset, batch_size=int(batch_size/2.0), num_workers=cpu_count(), shuffle=True)
    dataloader['testing_length'] = len(val_dataset)

    # Training, and if train is false, loads model through passed model path
    print("Running experiment {} with {} epochs".format(exp_name, epochs))
    if train:
        start = time.time()
        for epoch in range(epochs):
            # epoch += 1
            scheduler.step()
            avg_train_loss, avg_train_top1_prec = run_model(epoch, model, criterion, optimizer, dataloader['training'], dataloader['training_length'],  
                                                            train=True, device=device, num_classes=num_classes, regularizer=None)
            
            # save model
            current_date = str(datetime.now().year)+str(datetime.now().month)+str(datetime.now().day)
            checkpoint_path = os.path.join(os.getcwd(), '{}_{}_checkpoints'.format(exp_name, current_date))
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            is_best = avg_train_top1_prec > best_acc
            best_acc = max(avg_train_top1_prec, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': avg_train_top1_prec, 
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=checkpoint_path)
        end = time.time()
        print("Completed epochs {} | Train Time {} | Acc {} | Loss {} ".format(epochs, round(end-start, 2), avg_train_top1_prec,  avg_train_loss))

    # Validation
    start = time.time()
    avg_test_loss, avg_test_top1_prec = run_model(epoch, model, criterion, optimizer, dataloader['testing'], dataloader['testing_length'], 
                                                                train=False, device=device, num_classes=num_classes)
    end = time.time() - start
    print(" Test Time  {} | Acc {} | Loss {}".format(round(end-start, 2), avg_test_top1_prec, avg_test_loss, ))


def run_model(epoch, model, criterion, optimizer, dataloader,  datalength, device, train=True, regularizer=None, num_classes=10):
    """
    This function will run the model in either train or test mode returning the overall average loss and the top1 loss.

    Args:
        epoch (int): The epoch that the run is currently on. 
        model: The model being used. 
        criterion: The function that is used to evaluate loss.
        optimizer: The function that is used for backpropagation.
        dataloader: A generator object to iterate through the train/test dataset. 
        datalength (int): The number of batches in the data generator. 
        device (str): The device to run on, either 'cpu' or 'cuda'.
        train (bool): Whether the run is to train or not. Default is True.
        regularizer: A regularizer function if using custom regularizer to add to the global loss function. 
        num_classes (int): The number of classes that are being used for the multi-class problem.

    Returns: 
        Average loss, and top1k loss.
    """
    # Determine if we are in training mode or in validation mode
    if train:
        model.train()
        bar = Bar('Training', max=datalength)
    else:
        model.eval()
        bar = Bar('Validating', max=datalength)

    # Initialize model performance tracking metrics
    losses = Metrics()
    results = Metrics()
    # top5 = Metrics()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Send inputs and outputs to device which is either 'cpu' or 'cuda'
        inputs = inputs.to(device)
        inputs = torch.autograd.Variable(inputs)
        # Forward Pass,
        outputs = model(inputs)

        # measure accuracy and record loss
        labels = targets['label'].to(device).type(torch.long)
        if len(labels.shape) < 1:
            labels = labels.squeeze(1).type(torch.long)
        try:
            loss = criterion(outputs, labels)
        except RuntimeError:
            pdb.set_trace()
        losses.update(loss.data.item(), inputs.size(0))
        acc = accuracy(outputs.data, labels, topk=(1, 5))
        results.update(acc.item(), inputs.size(0))
        # top5.update(prec5.item(), inputs.size(0))
    
        # Backward Pass: compute gradient and do SGD step
        if train:   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        bar.suffix = '({batch}/{size}) | Total: {total:} | Epoch: {epoch:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(dataloader),
            total=bar.elapsed_td,
            epoch=epoch+1,
            loss=losses.avg,
            acc=results.avg,
        )
        bar.next() 
    bar.finish()
    return losses.avg, results.avg


def save_checkpoint(state, is_best, checkpoint):
    """
    This function will save a checkpoint based off of the passed checkpoint save path.
    If the model is the best so far, it will also save the full state dict of the model.
    The state wil have the keys: 
        * epoch: the epoch the model training is currently on
        * state_dict: the parameters of the model at the current epoch
        * acc: the average testing top1 precision
        * best_acc: the best accuracy the model has produced so far, may be from a different epoch
        * optimizer: the optimizer state dict

    Args: 
        state (dict): THe current state of the model.
        is_best (bool): If the model is the best so far with the current parameters.
        checkpoint (str): The directory to save the current model state to 
    """
    try:
        filename = 'checkpoint.pth.tar'
        filepath = os.path.join(checkpoint, str(state['epoch'])+'_'+str(datetime.now()).replace(' ', '')+'_'+filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    except Exception as e:
        print("There is an error saving checkpoints: {}".format(e))
        exit()


def test_functionality():
    num_conv = 3
    dataset =  'imagenet'
    data_path = 'data/tiny-imagenet-200'
    perc_class = 20
    batch_size = 12
    epochs = 1
    run_experiment(num_conv, dataset, data_path, perc_class=100, batch_size=64, epochs=100, num_classes=200, train=True)

if __name__=='__main__':
    from utils import plotting
    import json
    results = json.load(open('/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/plots/results.json', 'r'))
    outdir = '/home/mschiappa/PycharmProjects/UCFAdvancedComputerVision/plots'

    # Experiment 1
    for experiment in [1,2,3]:
        for query in ['Average Accuracy', 'Average Loss', 'Average Training Loss', 'Average Training Accuracy', 'Train Time']:
            plotting.plot(results, experiment, outdir, query)


    # # Experiment 2
    # plotting.plot(results, 2, outdir, loss=False)
    # plotting.plot(results, 2, outdir, loss=True)

    # # Experiment 3
    # plotting.plot(results, 3, outdir, loss=False)
    # plotting.plot(results, 3, outdir, loss=True)