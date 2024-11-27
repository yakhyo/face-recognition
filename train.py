from utils import layer
import lfw_eval
from utils.dataset import ImageList
from models import net, mobilefacenet
import argparse
import os
import time
import torch

import torch.utils.data
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')
parser.add_argument('--root_path', type=str, default='', help='path to root path of images')
parser.add_argument('--database', type=str, default='WebFace', help='Which Database for train. (WebFace, VggFace2)')
parser.add_argument('--train_list', type=str, default=None, help='path to training list')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 512)')
parser.add_argument('--is_gray', action='store_true', help='Transform input image to gray')
parser.add_argument('--network', type=str, default='mobilenet', help='Which network for train.')
# parser.add_argument('--network', type=str, default='mobilenet', help='Which network for train.')
parser.add_argument('--num_class', type=int, default=None, help='number of people (class)')
parser.add_argument('--classifier_type', type=str, default='ARC', help='Which classifier for train.')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=None, help='lr decay step')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 0.0005)')
parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='checkpoint/', help='path to save checkpoint')
parser.add_argument('--no_cuda', action='store_true', help='disables CUDA training')
parser.add_argument('--workers', type=int, default=8, help='how many workers to load data')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

if args.database == 'WebFace':
    args.train_list = 'data/train/webface_train_ann.txt'
    args.num_class = 10572
    args.step_size = [10, 20]
elif args.database == 'VggFace2':
    args.train_list = '/home/wangyf/dataset/VGG-Face2/VGG-Face2-112X96.txt'
    args.num_class = 8069
    args.step_size = [80000, 120000, 140000]
else:
    raise ValueError("NOT SUPPORT DATABASE!")


def main():
    # Model selection based on arguments
    if args.network == 'sphere20':
        model = net.SphereNet(type=20, is_gray=args.is_gray)
    elif args.network == 'sphere64':
        model = net.SphereNet(type=64, is_gray=args.is_gray)
    elif args.network == 'LResNet50E_IR':
        model = net.LResNet50E_IR(is_gray=args.is_gray)
    elif args.network=="mobilenet":
        model = mobilefacenet.get_mbf(False, 512)
    else:
        raise ValueError("NOT SUPPORT NETWORK!")

    # No need for DataParallel, we are using a single GPU
    model = model.to(device)

    # Create save path if it does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Select classifier
    classifier = {
        'MCP': layer.MarginCosineProduct(512, args.num_class),
        'AL': layer.AngleLinear(512, args.num_class),
        'ARC': layer.ArcFace(512, args.num_class),
        'L': torch.nn.Linear(512, args.num_class, bias=False)
    }[args.classifier_type].to(device)

    # Transformations for images
    if args.is_gray:
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, fileList=args.train_list, transform=train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    

    print(f'Length of train dataset: {len(train_loader.dataset)}, Number of Identities: {args.num_class}')

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
        {'params': classifier.parameters()}
    ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step_size, gamma=0.1)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(train_loader, model, classifier, criterion, optimizer, epoch)
        save_path = os.path.join(args.save_path, f'{args.network}_{epoch}_checkpoint.pth')
        torch.save(model.state_dict(), save_path)
        scheduler.step()


def train(train_loader, model, classifier, criterion, optimizer, epoch):
    model.train()
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = classifier(output, target) if not isinstance(classifier, torch.nn.Linear) else classifier(output)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, Time: {elapsed_time:.4f}s')
            start_time = time.time()


if __name__ == '__main__':
    main()
