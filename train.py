import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from wide_resnet import wrn_22_10

import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--depth', default=22, help='')
parser.add_argument('--widen_factor', default=10, help='')
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--batch_size_test', default=128, help='')
parser.add_argument('--momentum', default=0.9, help='')
parser.add_argument('--weight_decay', default=5e-4, help='')
parser.add_argument('--epoch', default=200, help='')
parser.add_argument('--num_worker', default=4, help='')
parser.add_argument('--logdir', type=str, default='logs', help='')
args = parser.parse_args()
print("args: ", args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data...')

transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_train = CIFAR10(root='./data', train=True,
                        download=True, transform=transforms_train)
dataset_test = CIFAR10(root='./data', train=False,
                        download=True, transform=transforms_test)
train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_worker)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test,
                            shuffle=True, num_workers=args.num_worker)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Making model...')

model = wrn_22_10()
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                      momentum=args.momentum, weight_decay=args.weight_decay)

decay_epoch = [60, 120, 160]
step_lr_schedular = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.2)
writer = SummaryWriter(args.logdir)

def train(epoch, global_steps):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        global_steps += 1
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100 * correct / total
    print('train epoch: {} | loss: {:.3f} | acc: {:.3f}'.format(
        epoch, train_loss/(batch_idx+1), acc))
    
    writer.add_scalar('log/train error', 100 - acc, global_steps)
    return global_steps


def test(epoch, best_acc, global_steps):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100 * correct / total
    print('test epoch: {} [{} / {}] | loss: {:.3f} | acc: {:.3f}'.format(
        epoch, batch_idx, len(test_loader), test_loss/(batch_idx+1), acc))
    
    writer.add_scalar('log/train error', 100 - acc, global_steps)
    
    if acc > best_acc:
        print('==> Saving model...')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('save_model'):
            os.mkdir('save_model')
        torch.save(state, './save_model/ckpt.pth')
        best_acc = acc

    return best_acc

if __name__ == '__main__':
    best_acc = 0
    epoch = 0
    global_steps = 0

    for epoch in range(args.epoch):

        global_steps = train(epoch, global_steps)
        best_acc = test(epoch, best_acc, global_steps)
        step_lr_schedular.step()
        print('best test accuracy is ', best_acc)
