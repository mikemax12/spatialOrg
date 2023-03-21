import copy
import os
import argparse
import os
import time
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
from helper import utils
from shutil import copyfile
import pickle
from torch.utils.data import Dataset

# from models.resnet import resnet50, resnet18
import torchvision.models as models
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
from tqdm import tqdm
import transformers
from transformers import AutoImageProcessor, ViTForImageClassification
import torch

import torch, gc
gc.collect()
torch.cuda.empty_cache()


def get_params(train=False):
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # general params
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset can be cifar10, svhn')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for training the model')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='device for training the model')
    parser.add_argument('--trial', type=str, default='test',
                        help='name for the experiment')
    parser.add_argument('--chkpt', type=str, default='',
                        help='checkpoint for resuming the training')

    # training params

    parser.add_argument('--bs', type=int, default=2,
                        help='batchsize')
    parser.add_argument('--nw', type=int, default=8,
                        help='number of workers for the dataloader')
    parser.add_argument('--save_dir', type=str, default='./logs',
                        help='directory to log and save checkpoint')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd')
    parser.add_argument('--epochs', default=90, type=int, help='training epochs')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60',
                        help='where to decay lr, can be a list')
    parser.add_argument('--datadir', type=str, default='/home/aldb/dataset/ILSVRC/Data/CLS-LOC',
                        help='directory of the data')

    args = parser.parse_args()

    if train and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = models.resnet18()

    def forward(self, x):
        return self.net(x)


def main(args):
    if 'aldb' in __file__:
        args.bs = 2
        args.nw = 1

    # copy this file to the log dir
    a = os.path.basename(__file__)

    file_name = a.split('.py')[0]

    exp_name = '%s/%s' % (file_name, args.trial)
    args.save_dir = os.path.join(args.save_dir, exp_name)

    # create the log folder
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    elif 'test' not in args.save_dir:
        ans = input('log dir exists, override? (yes=y)')

        print(ans, '<<<<<<<')

        if ans != 'y':
            exit()

    # creat logger
    logger = utils.Logger(args=args,
                          var_names=['Epoch', 'train_loss', 'train_acc', 'test_acc', 'best_acc', 'lr'],
                          format=['%02d', '%.4f', '%.4f', '%.3f', '%.3f', '%.6f'],
                          print_args=True)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    copyfile(os.path.join(dir_path, a), os.path.join(args.save_dir, a))

    # save args
    with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

    device = args.device

    acc_best = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    ######################################################################################

    # Data loader for general contrastive learning =====================================================================
    traindir = os.path.join(args.datadir, 'train')
    valdir = os.path.join(args.datadir, 'val2')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True,
        num_workers=args.nw, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False,
        num_workers=args.nw, pin_memory=True)

    ######################################################################################

    net = Net()

    print('# of params in bbone    : %d' % utils.count_parameters(net))

    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    if args.chkpt != '':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.chkpt)
        net.load_state_dict(checkpoint['net'], 'cpu')
        start_epoch = args.startepoch

    net = nn.DataParallel(net)
    net = net.to(device)

    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.momentum, weight_decay=args.wd)

    best_acc = 0

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    if start_epoch > 0:
        scheduler.step(start_epoch)

    for epoch in range(start_epoch, args.epochs):

        lr = optimizer.param_groups[0]['lr']

        # break
        print("\n epoch: %02d, learning rate: %.4f" % (epoch, lr))
        t0 = time.time()

        train_acc, train_loss = train(
            epoch,
            net,
            optimizer,
            train_loader,
            device,
            args)

        print('>>>>>>>', args.save_dir)

        scheduler.step()

        # compute acc on nat examples
        test_acc, test_loss = validate(epoch, net, val_loader, device, args)

        print('%s: test acc: %.2f, best acc: %.2f' % (
            args.trial, test_acc, best_acc))

        state = {'net': net.module.state_dict(),
                 'acc': test_acc}
        torch.save(state, os.path.join(args.save_dir, 'model_last.pt'))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(state, os.path.join(args.save_dir, 'model_best.pt'))

        # print('test acc nat: %2.2f, best acc: %.2f' % (test_acc, acc_best))
        #
        logger.store(
            [epoch, train_loss, train_acc, test_acc, best_acc,
             optimizer.param_groups[0]['lr']],
            log=True)

        t = time.time() - t0
        remaining = (args.epochs - epoch) * t
        print("epoch time: %.1f, rt:%s" % (t, utils.format_time(remaining)))

        # scheduler.step()

def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, device=torch.device('cpu')):
    metrics.reset()

    # training loop
    for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))

        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        metrics.update('acc1', acc1.item())
        metrics.update('acc5', acc5.item())

        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Acc@1: {:.2f}, Acc@5: {:.2f}"
                    .format(epoch, batch_idx, len(data_loader), loss.item(), acc1.item(), acc5.item()))
    return metrics.result()


def valid_epoch(epoch, model, data_loader, criterion, metrics, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
            acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))

            losses.append(loss.item())
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

    loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', acc1)
    metrics.update('acc5', acc5)
    return metrics.result()


def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str(save_dir + 'current.pth')
    torch.save(state, filename)

    if best:
        filename = str(save_dir + 'best.pth')
        torch.save(state, filename)

def train(epoch, net, optimizer, trainloader, device, args):
    net.train()

    

    prog_bar = tqdm(
        trainloader,
        ascii=True,
        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
    )

    correct = 0
    total = 0
    
    config = get_train_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
        momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        pct_start=config.warmup_steps / config.train_steps,
        total_steps=config.train_steps)

    criterion = nn.CrossEntropyLoss()
    epochs = 100
    


    #!pip install transformers
    import transformers
    from transformers import AutoImageProcessor, ViTForImageClassification
    import torch
    import datasets
    from datasets import load_dataset

    import time
    import json
    import copy
    import torch
    import numpy as np
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch import nn, optim
    from torch.optim import lr_scheduler


    from torchvision import transforms

    from torchvision import models
    from torchvision import datasets


    import kaggle
    import os
    from torch.utils.data import Dataset
    from PIL import Image
    import json
    class ImageNetKaggle(Dataset):
        def __init__(self, root, split, transform=None):
            self.samples = []
            self.targets = []
            self.transform = transform
            self.syn_to_class = {}
            with open(os.path.join("/shared/local_scratch/gpu1/c_mtmaxwel/ENETER/ILSRVC", "imagenet_class_index.json"), "rb") as f:
                        json_file = json.load(f)
                        for class_id, v in json_file.items():
                            self.syn_to_class[v[0]] = int(class_id)
            with open(os.path.join("/shared/local_scratch/gpu1/c_mtmaxwel/ENETER/ILSRVC", "ILSVRC2012_val_labels.json"), "rb") as f:
                        self.val_to_syn = json.load(f)
            samples_dir = os.path.join("/shared/local_scratch/gpu1/c_mtmaxwel/ENETER/", "ILSVRC/Data/CLS-LOC", split)
            for entry in os.listdir(samples_dir):
                if split == "train":
                    syn_id = entry
                    target = self.syn_to_class[syn_id]
                    syn_folder = os.path.join(samples_dir, syn_id)
                    for sample in os.listdir(syn_folder):
                        sample_path = os.path.join(syn_folder, sample)
                        self.samples.append(sample_path)
                        self.targets.append(target)
                elif split == "val":
                    syn_id = self.val_to_syn[entry]
                    target = self.syn_to_class[syn_id]
                    sample_path = os.path.join(samples_dir, entry)
                    self.samples.append(sample_path)
                    self.targets.append(target)
        def __len__(self):
                return len(self.samples)
        def __getitem__(self, idx):
                x = Image.open(self.samples[idx]).convert("RGB")
                if self.transform:
                    x = self.transform(x)
                return x, self.targets[idx]


    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torch
    import torchvision
    from tqdm import tqdm
    import transformers
    from transformers import AutoImageProcessor, ViTForImageClassification
    import torch

    #image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = net #ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    model.train().cuda()  # Needs CUDA, don't bother on CPUs
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    dataset = ImageNetKaggle("train", "", val_transform)
    dataloader = DataLoader(
                dataset,
                batch_size=64, # may need to reduce this depending on your GPU 
                num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                shuffle=False,
                drop_last=False,
                pin_memory=True
            )
    correct = 0
    total = 0


    '''
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            logits = model(x.cuda()).logits
            _, preds = torch.max(logits, 1)
            # model predicts one of the 1000 ImageNet classes
            #preds = logits.argmax(-1).item()
            #_, preds = torch.max(outputs, 1)
            #loss = criteria(outputs, labels)
            #loss = criteria(logits, labels)
            print("printing")
            print(torch.max(logits, 1))
            print(logits)
            print(preds)
            print(y)'''


    def train_model(model, criteria, optimizer, scheduler,    
                                          num_epochs=25, device='cuda'):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(1, 15):#num_epochs+1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)
            running_loss = 0.0
            running_corrects = 0
            # Each epoch has a training and validation phase
            for x, y in tqdm(dataloader):


                # Iterate over data.
                inputs = x
                labels = y
                inputs = inputs.to(device)
                labels = labels.to(device)

                #print(inputs.shape)
                # zero the parameter gradients
                optimizer.zero_grad()
                #print(inputs.shape)
                # forward
                # track history if only in train

                    #outputs = model(inputs)
                logits = model(inputs).logits
                _, preds = torch.max(logits, 1)
                # model predicts one of the 1000 ImageNet classes
                #preds = logits.argmax(-1).item()
                #_, preds = torch.max(outputs, 1)
                #loss = criteria(outputs, labels)
                loss = criteria(logits, labels)
                #print(loss)

                    # backward + optimize only if in training phase

                loss.backward() #computes gradients
                optimizer.step() #updates weights

                    # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                print(running_corrects)
                print(running_corrects)

                epoch_loss = running_loss / len(dataset)
                epoch_acc = running_corrects.double() / len(dataset)
                print(running_corrects / len(dataset))
                #epoch_acc = running_corrects/64

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    "train", epoch_loss, epoch_acc))

                # deep copy the model
                if epoch_acc > best_acc:
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

    criteria = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # Number of epochs
    eps=5

    model = train_model(net, criteria, optimizer, scheduler, eps, 'cuda') #try to look at why val loss is lower


def validate(epoch, net, valloader, device, args):
    net.eval()

    am_loss = utils.AverageMeter()
    am_acc = utils.AverageMeter()

    prog_bar = tqdm(
        valloader,
        ascii=True,
        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
    )

    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (img, target) in enumerate(prog_bar):
            img, target = img.cuda(), target.cuda()
            
            bs = len(img)

            logits = net(img)

            loss = criterion(logits, target)
            pred_lbl = logits.argmax(1)
            print("Targ, loss, logits")
            print(target)
            print(loss)
            print(logits)
            print(pred_lbl)
            correct += (pred_lbl == target).type(torch.float).sum()
            total += bs

            am_loss.update(loss.item())
            am_acc.update(correct / total)

            prog_bar.set_description(
                "eval: E{}/{}, loss:{:2.3f}, loss_aux:{:2.3f}, acc:{:2.2f}".format(
                    epoch, args.epochs, am_loss.avg, 0, correct * 100 / total))
    prog_bar.close()

    acc = correct * 100 / total

    return acc, am_loss.avg


if __name__ == "__main__":
    args = get_params(train=True)

    args.gpu_devices = [int(id) for id in args.gpu_id.split(',')]

    args.lr_decay_epochs = [int(e) for e in args.lr_decay_epochs.split(',')]

    print('gpu devices to use: ', args.gpu_devices)

    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    main(args)





