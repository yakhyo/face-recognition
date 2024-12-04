
import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


import lfw_eval_v2 as lfw_eval
from utils.dataset import ImageFolder
from utils.metrics import ArcFace, MarginCosineProduct, SphereFace
from utils.general import AverageMeter, calculate_accuracy, init_distributed_mode, reduce_tensor

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
        choices=['WebFace', 'VggFace2', "MS1M"],
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
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training. Default: 512.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training. Default: 30.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate. Default: 0.1.')
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
        default='weights',
        help='Path to save model checkpoints. Default: `weights`.'
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

    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")

    return parser.parse_args()


# Define a function to select a classification head
def get_classification_head(classifier, embedding_dim, num_classes):
    classifiers = {
        'MCP': MarginCosineProduct(embedding_dim, num_classes),
        'AL': SphereFace(embedding_dim, num_classes),
        'ARC': ArcFace(embedding_dim, num_classes),
        'L': torch.nn.Linear(embedding_dim, num_classes, bias=False)
    }

    if classifier not in classifiers:
        raise ValueError(f"Unsupported classifier type: {classifier}")

    return classifiers[classifier]


def get_dbconfig(database):
    db_config = {
        'WebFace': {
            'num_classes': 10572,
            'step_size': [10, 20, 25],
        },
        'VggFace2': {
            'num_classes': 8631,
            'step_size': [10, 15, 25],
        },
        'MS1M': {
            'num_classes': 85742,
            'step_size': [10, 15, 25],
        }
    }

    if database not in db_config:
        raise ValueError("Unsupported database!")

    return db_config[database]['num_classes'], db_config[database]['step_size']


def train_one_epoch(model, classification_head, criterion, optimizer, data_loader, device, epoch, params) -> None:
    model.train()

    losses = AverageMeter("Avg Loss", ":6.3f")
    batch_time = AverageMeter("Batch Time", ":4.3f")
    accuracy_meter = AverageMeter("Accuracy", ":4.2f")
    last_batch_idx = len(data_loader) - 1

    start_time = time.time()
    for batch_idx, (images, target) in enumerate(data_loader):
        last_batch = last_batch_idx == batch_idx

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

        if args.distributed:
            reduced_loss = reduce_tensor(loss, args.world_size)
            accuracy = reduce_tensor(accuracy, args.world_size)
        else:
            reduced_loss = loss

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Update metrics
        losses.update(reduced_loss.item(), images.size(0))
        accuracy_meter.update(accuracy.item(), images.size(0))
        batch_time.update(time.time() - start_time)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Reset start time for the next batch
        start_time = time.time()

        # Log results at intervals
        if batch_idx % params.print_freq == 0 or last_batch:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            print(
                f'Epoch: [{epoch}/{params.epochs}][{batch_idx:05d}/{len(data_loader):05d}] '
                f'Loss: {losses.avg:6.3f}, '
                f'Accuracy: {accuracy_meter.avg:4.2f}%, '
                f'LR: {lr:.5f} '
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
    init_distributed_mode(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Configure dataset-specific settings
    num_classes, step_size = get_dbconfig(params.database)

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

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    # Create save path if it does not exist
    os.makedirs(params.save_path, exist_ok=True)

    # Select classification head
    classification_head = get_classification_head(params.classifier, embedding_dim=512, num_classes=num_classes)
    classification_head = classification_head.to(device)

    # Transformations for images
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # DataLoader
    train_dataset = ImageFolder(root=params.root, transform=train_transform)

    if params.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=params.workers,
        pin_memory=True
    )

    print(f'Length of train dataset: {len(train_loader.dataset)}, Number of Identities: {num_classes}')

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
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=0.1)

    best_accuracy = 0.0
    # Training loop
    for epoch in range(1, params.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
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
        last_save_path = os.path.join(params.save_path, f'{params.network}_{params.classifier}_last.pth')
        torch.save(model_without_ddp.state_dict(), last_save_path)

        lr_scheduler.step()
        if torch.distributed.get_rank() == 0:
            accuracy, _ = lfw_eval.eval(model_without_ddp, last_save_path, device)

            # Save the best model if accuracy improves
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = os.path.join(params.save_path, f'{params.network}_{params.classifier}_best.pth')
                torch.save(model_without_ddp.state_dict(), best_model_path)
                print(f"New best accuracy: {best_accuracy:.4f}. Model saved to {best_model_path}")

        print(f"Epoch {epoch} completed. Latest model saved to {last_save_path}. Best accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
