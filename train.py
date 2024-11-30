
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


def parse_arguments():
    parser = argparse.ArgumentParser(description=("Command-line arguments for training a face recognition model"))

    # Dataset and Paths
    parser.add_argument(
        '--root',
        type=str,
        default='data/train/webface_112x112/',
        help='Path to the root directory of training images.'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='WebFace',
        choices=['WebFace', 'VggFace2'],
        help='Database to use for training. Options: WebFace, VggFace2.'
    )

    # Model Settings
    parser.add_argument(
        '--network',
        type=str,
        default='sphere20',
        choices=['sphere20', 'sphere36', 'sphere64', 'mobile'],
        help='Network architecture to use. Options: sphere20, sphere36, sphere64, mobile.'
    )
    parser.add_argument(
        '--classifier',
        type=str,
        default='MCP',
        choices=['ARC', 'MCP', 'AL', 'L'],
        help='Type of classifier to use. Options: ARC (ArcFace), MCP (MarginCosineProduct), AL (SphereFace), L (Linear).'
    )

    # Training Hyperparameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size for training. Default: 512.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs for training. Default: 30.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='Initial learning rate. Default: 0.1.'
    )
    parser.add_argument(
        '--step-size',
        type=list,
        default=None,
        help='Milestones for learning rate decay in MultiStepLR. Default: None.'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum factor for SGD optimizer. Default: 0.9.'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        help='Weight decay for SGD optimizer. Default: 5e-4.'
    )

    # Data Transformations
    parser.add_argument(
        '--save-path',
        type=str,
        default='checkpoint/',
        help='Path to save model checkpoints. Default: checkpoint/.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of data loader workers. Default: 8.'
    )
    parser.add_argument(
        '--print-freq',
        type=int,
        default=100,
        help='Frequency (in batches) for printing training progress. Default: 100.'
    )

    return parser.parse_args()


# Define a function to select a classification head
def get_classification_head(classifier, embedding_dim, num_classes):
    classifiers = {
        'MCP': layer.MarginCosineProduct(embedding_dim, num_classes),
        'AL': layer.SphereFace(embedding_dim, num_classes),
        'ARC': layer.ArcFace(embedding_dim, num_classes),
        'L': torch.nn.Linear(embedding_dim, num_classes, bias=False)
    }

    if classifier not in classifiers:
        raise ValueError(f"Unsupported classifier type: {classifier}")

    return classifiers[classifier]


def train_one_epoch(model, classification_head, criterion, optimizer, data_loader, device, epoch, params) -> None:
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
        if isinstance(classification_head, torch.nn.Linear):
            output = classification_head(embeddings)
        else:
            output = classification_head(embeddings, target)

        # Compute loss and accuracy
        loss = criterion(output, target)
        accuracy = calculate_accuracy(output, target)

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
        if batch_idx % params.print_freq == 0:
            print(
                f'Epoch: [{epoch}/{params.epochs}][{batch_idx:04d}/{len(data_loader):04d}] '
                f'Loss: {losses.avg:6.3f}, '
                f'Accuracy: {accuracy_meter.avg:4.2f}%, '
                f'Time: {batch_time.avg:4.3f}s'
            )

    # End-of-epoch summary
    print(
        f'Epoch [{epoch}/{params.epochs}] Summary: '
        f'Loss: {losses.avg:6.3f}, '
        f'Accuracy: {accuracy_meter.avg:4.2f}%, '
        f'Total Time: {batch_time.sum:4.3f}s'
    )


def main(params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure database-specific settings
    db_config = {
        'WebFace': {
            'num_classes': 10572,
            'step_size': [10, 20],
            'mean': (0.5203, 0.4045, 0.3465),
            'std': (0.2417, 0.2076, 0.1948)
        },
        'VggFace2': {
            'num_classes': 8631,
            'step_size': [10, 15, 25],
            'mean': (0.5334, 0.4158, 0.3601),
            'std': (0.2467, 0.2135, 0.2010)
        },
    }

    if params.database not in db_config:
        raise ValueError("Unsupported database!")

    params.num_classes = db_config[params.database]['num_classes']
    params.step_size = db_config[params.database]['step_size']

    mean = db_config[params.database]['mean']
    std = db_config[params.database]['std']

    # Model selection based on arguments
    if params.network == 'sphere20':
        model = sphere20(embedding_dim=512, in_channels=3)
    elif params.network == 'sphere36':
        model = sphere36(embedding_dim=512, in_channels=3)
    elif params.network == 'sphere64':
        model = sphere64(embedding_dim=512, in_channels=3)
    elif params.network == "mobile":
        model = MobileNetV2(input_size=(112, 112))
    else:
        raise ValueError("Unsupported network!")

    # No need for DataParallel, we are using a single GPU
    model = model.to(device)

    # Create save path if it does not exist
    os.makedirs(params.save_path, exist_ok=True)

    # Select classification head
    classification_head = get_classification_head(params.classifier, embedding_dim=512, num_classes=params.num_classes)
    classification_head = classification_head.to(device)

    # Transformations for images
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # DataLoader
    train_dataset = ImageFolder(root=params.root, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        num_workers=params.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    print(f'Length of train dataset: {len(train_loader.dataset)}, Number of Identities: {params.num_classes}')

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
        {'params': classification_head.parameters()}
    ],
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.step_size, gamma=0.1)

    # Training loop
    for epoch in range(1, params.epochs + 1):
        train_one_epoch(
            model,
            classification_head,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            params
        )
        save_path = os.path.join(params.save_path, f'{params.network}_{epoch}_checkpoint.pth')
        torch.save(model.state_dict(), save_path)
        scheduler.step()
        lfw_eval.eval(model, save_path, device)


def validate():
    pass


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
