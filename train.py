import torch
import os
import ssl
import argparse
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from models import Cifar10Model, MnistModel


BATCH_SIZE = 128
LEARNING_RATE = 1E-2


def train(dataset_name, m):
    opt = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)
    loss_func = torch.nn.CrossEntropyLoss()

    root = os.path.join(
        'models',
        dataset_name,
        'with_batch_norm' if m.batch_norm else 'no_batch_norm',
        now.strftime('%m_%d_%Y'),
        now.strftime('%H_%M_%S')
    )
    weight_dir = os.path.join(root, 'weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    train_losses = np.empty((0, 2))
    test_losses = np.empty((0, 2))
    train_errors = np.empty((0, 2))
    test_errors = np.empty((0, 2))

    def save_metrics():
        np.save(os.path.join(root, 'train_losses'), train_losses)
        np.save(os.path.join(root, 'test_losses'), test_losses)
        np.save(os.path.join(root, 'train_errors'), train_errors)
        np.save(os.path.join(root, 'test_errors'), test_errors)

    for epoch in tqdm(range(160), desc='Epoch'):
        train_loss = 0
        train_acc = 0
        for data, labels in tqdm(train_loader, desc='Train', leave=False):
            data = data.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            predictions = m(data)
            loss = loss_func(predictions, labels)
            loss.backward()
            opt.step()

            train_loss += loss.item() / len(train_loader)
            train_acc += labels.eq(torch.argmax(predictions, 1)).sum().item() / len(train_set)
            del data, labels
        train_losses = np.append(train_losses, [[epoch, train_loss]])
        train_errors = np.append(train_errors, [[epoch, 1 - train_acc]])
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Error/train', 1 - train_acc, epoch)

        with torch.no_grad():
            test_loss = 0
            test_acc = 0
            for data, labels in tqdm(test_loader, desc='Test', leave=False):
                data = data.to(device)
                labels = labels.to(device)

                predictions = m(data)
                loss = loss_func(predictions, labels)

                test_loss += loss.item() / len(test_loader)
                test_acc += labels.eq(torch.argmax(predictions, 1)).sum().item() / len(test_set)
                del data, labels
            test_losses = np.append(test_losses, [[epoch, test_loss]])
            test_errors = np.append(test_errors, [[epoch, 1 - test_acc]])
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Error/test', 1 - test_acc, epoch)

            if epoch % 2 == 0:
                save_metrics()
            if epoch % 20 == 0:
                torch.save(m.state_dict(), os.path.join(weight_dir, f'cp_{epoch}'))
    save_metrics()
    torch.save(m.state_dict(), os.path.join(weight_dir, 'final'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=('cifar10', 'mnist'))
    parser.add_argument('-bn', '--batch-norm', action='store_true')
    args = parser.parse_args()

    writer = SummaryWriter()
    now = datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'cifar10':
        num_layers = 32
        transform = T.Compose([
            T.ToTensor(),
            lambda x: x - torch.mean(x, (1, 2), keepdim=True)
        ])

        ssl._create_default_https_context = ssl._create_unverified_context  # Patch expired certificate error
        train_set = CIFAR10(
            root='data',
            train=True,
            download=True,
            transform=transform
        )
        test_set = CIFAR10(
            root='data',
            train=False,
            transform=transform
        )
        model = Cifar10Model(num_layers, batch_norm=args.bn).to(device)
    else:
        train_set = MNIST(
            root='data',
            train=True,
            download=True,
            transform=T.ToTensor()
        )
        test_set = MNIST(
            root='data',
            train=False,
            transform=T.ToTensor()
        )
        model = MnistModel(batch_norm=args.bn).to(device)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    train(args.dataset, model)
