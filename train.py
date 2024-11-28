
import argparse
import os
import time
import torch

import torch.utils.data
from torchvision import transforms

from utils import layer
import lfw_eval
from utils.dataset import ImageList, ImageFolder

from utils.helper import AverageMeter, calculate_accuracy

from models import mobilefacenet
from models.sphereface import sphere20, sphere36, sphere64

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')
parser.add_argument('--root_path', type=str, default='data/train/webface_112x112', help='path to root path of images')
parser.add_argument('--database', type=str, default='WebFace', help='Which Database for train. (WebFace, VggFace2)')
parser.add_argument('--train_list', type=str, default=None, help='path to training list')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 512)')
parser.add_argument('--is_gray', action='store_true', help='Transform input image to gray')
parser.add_argument('--network', type=str, default='sphere20', help='Which network for train.')
parser.add_argument('--num_class', type=int, default=None, help='number of people (class)')
parser.add_argument('--classifier_type', type=str, default='MCP', help='Which classifier for train.')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=None, help='lr decay step')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 0.0005)')
parser.add_argument('--print-freq', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='checkpoint/', help='path to save checkpoint')
parser.add_argument('--workers', type=int, default=8, help='how many workers to load data')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.database == 'WebFace':
    args.train_list = 'data/train/webface_train_ann.txt'
    args.num_class = 10572
    args.step_size = [10, 20]
elif args.database == 'VggFace2':
    args.train_list = '/data/train/vggface2_train_ann.txt'
    args.num_class = 8069
    args.step_size = [80000, 120000, 140000]
else:
    raise ValueError("NOT SUPPORT DATABASE!")


def main():
    # Model selection based on arguments
    in_channels = 1 if args.is_gray else 3
    if args.network == 'sphere20':
        model = sphere20(embedding_dim=512, in_channels=in_channels)
        model_eval = sphere20(embedding_dim=512, in_channels=in_channels)
    elif args.network == 'sphere36':
        model = sphere36(embedding_dim=512, in_channels=in_channels)
        model_eval = sphere36(embedding_dim=512, in_channels=in_channels)
    elif args.network == 'sphere64':
        model = sphere64(embedding_dim=512, in_channels=in_channels)
        model_eval = sphere64(embedding_dim=512, in_channels=in_channels)
    elif args.network == "mobile":
        model = MobileNetV2(input_size=(112, 112))
        model_eval = MobileNetV2(input_size=(112, 112))
    else:
        raise ValueError("NOT SUPPORT NETWORK! ")

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
        ImageFolder(root=args.root_path, transform=train_transform),
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
        train_one_epoch(model, classifier, criterion, optimizer, train_loader, device, epoch)
        save_path = os.path.join(args.save_path, f'{args.network}_{epoch}_checkpoint.pth')
        torch.save(model.state_dict(), save_path)
        scheduler.step()
        lfw_eval.eval(model_eval, save_path, args.is_gray)


def validate():
    pass


def train_one_epoch(model, classifier, criterion, optimizer, data_loader, device, epoch) -> None:
    model.train()

    losses = AverageMeter("Avg Loss", ":6.3f")
    batch_time = AverageMeter("Batch Time", ":4.3f")
    accuracy_meter = AverageMeter("Accuracy", ":4.2f")

    start_time = time.time()
    for batch_idx, (images, target) in enumerate(data_loader):
        # Move data to device
        images = images.to(device)
        target = target.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(images)
        if isinstance(classifier, torch.nn.Linear):
            output = classifier(embeddings)
        else:
            output = classifier(embeddings, target)

        # Compute loss and accuracy
        loss = criterion(output, target)
        accuracy = calculate_accuracy(output, target)
        # print(accuracy)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        accuracy_meter.update(accuracy.item(), images.size(0))
        batch_time.update(time.time() - start_time)

        # Reset start time for the next batch
        start_time = time.time()

        # Log results at intervals
        if batch_idx % args.print_freq == 0:
            print(
                f'Epoch: [{epoch}][{batch_idx:05d}/{len(data_loader)}] '
                f'Loss: {losses.avg:6.3f}, '
                f'Accuracy: {accuracy_meter.avg:4.2f}%, '
                f'Time: {batch_time.avg:4.3f}s'
            )

    # End-of-epoch summary
    print(
        f'Epoch [{epoch}] Summary: '
        f'Loss: {losses.avg:6.3f}, '
        f'Accuracy: {accuracy_meter.avg:4.2f}%, '
        f'Total Time: {batch_time.sum:4.3f}s'
    )
    


if __name__ == '__main__':
    main()
