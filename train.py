import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import lfw_eval
from utils.dataset import ImageFolder
from utils.metrics import ArcFace, MarginCosineProduct, SphereFace
from utils.general import (
    setup_seed,
    reduce_tensor,
    save_on_master,
    calculate_accuracy,
    init_distributed_mode,
    AverageMeter,
    EarlyStopping,
    LOGGER,
)

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
    # lr_scheduler configuration
    parser.add_argument(
        '--lr-scheduler',
        type=str,
        default='MultiStepLR',
        choices=['StepLR', 'MultiStepLR'],
        help='Learning rate scheduler type.'
    )
    parser.add_argument('--step-size', type=int, default=10, help='Period of learning rate decay for StepLR.')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='Multiplicative factor of learning rate decay for StepLR and ExponentialLR.'
    )
    parser.add_argument(
        '--milestones',
        type=int,
        nargs='+',
        default=[10, 20, 25],
        help='List of epoch indices to reduce learning rate for MultiStepLR (ignored if StepLR is used).'
    )
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD optimizer. Default: 0.9.')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        help='Weight decay for SGD optimizer. Default: 5e-4.'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='weights',
        help='Path to save model checkpoints. Default: `weights`.'
    )
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loader workers. Default: 8.')
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to continue training.")

    parser.add_argument(
        '--print-freq',
        type=int,
        default=100,
        help='Frequency (in batches) for printing training progress. Default: 100.'
    )

    parser.add_argument("--world-size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only."
    )

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


def train_one_epoch(
    model,
    classification_head,
    criterion, optimizer,
    data_loader,
    device,
    epoch,
    params
) -> None:
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

        # calculate_accuracy is a function to compute classification accuracy.
        accuracy = calculate_accuracy(output, target)

        if args.distributed:
            # reduce_tensor is used in distributed training to aggregate metrics (e.g., loss, accuracy)
            # across multiple GPUs. It ensures all devices contribute to the final metric computation.
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
            lr = optimizer.param_groups[0]['lr']
            log = (
                f'Epoch: [{epoch}/{params.epochs}][{batch_idx:05d}/{len(data_loader):05d}] '
                f'Loss: {losses.avg:6.3f}, '
                f'Accuracy: {accuracy_meter.avg:4.2f}%, '
                f'LR: {lr:.5f} '
                f'Time: {batch_time.avg:4.3f}s'
            )
            LOGGER.info(log)

    # End-of-epoch summary
    log = (
        f'Epoch [{epoch}/{params.epochs}] Summary: '
        f'Loss: {losses.avg:6.3f}, '
        f'Accuracy: {accuracy_meter.avg:4.2f}%, '
        f'Total Time: {batch_time.sum:4.3f}s'
    )
    LOGGER.info(log)


def main(params):
    init_distributed_mode(params)

    setup_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if params.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Configure dataset-specific settings
    db_config = {
        'WebFace': {
            'num_classes': 10572,
        },
        'VggFace2': {
            'num_classes': 8631,
        },
        'MS1M': {
            'num_classes': 85742,
        }
    }
    if params.database not in db_config:
        raise ValueError("Unsupported database!")

    num_classes = db_config[params.database]['num_classes']

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
    LOGGER.info('Loading training data.')
    train_dataset = ImageFolder(root=params.root, transform=train_transform)

    if params.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=params.num_workers,
        pin_memory=True
    )

    LOGGER.info(f'Length of training dataset: {len(train_loader.dataset)}, Number of Identities: {num_classes}')

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
    # Learning rate scheduler
    if params.lr_scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)
    elif params.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
    else:
        raise ValueError(f"Unsupported lr_scheduler type: {params.lr_scheduler}")

    start_epoch = 0
    if params.checkpoint and os.path.isfile(params.checkpoint):
        ckpt = torch.load(params.checkpoint, map_location=device, weights_only=True)

        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

        # Move optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = ckpt['epoch']
        LOGGER.info(f'Resumed training from {params.checkpoint}, starting at epoch {start_epoch}')

    best_accuracy = 0.0
    early_stopping = EarlyStopping(patience=10)

    # Training loop
    LOGGER.info(f'Training started for {params.network}, Classifier: {params.classifier}')
    for epoch in range(start_epoch, params.epochs):
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
        lr_scheduler.step()

        base_filename = f'{params.network}_{params.classifier}'

        last_save_path = os.path.join(params.save_path, f'{base_filename}_last.ckpt')

        # Save the last checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': params
        }

        save_on_master(checkpoint, last_save_path)

        if params.local_rank == 0:
            accuracy, _ = lfw_eval.eval(model_without_ddp, device=device)

        if early_stopping(epoch, accuracy):
            break

        # Save the best model if accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_on_master(checkpoint, os.path.join(params.save_path, f'{base_filename}_best.ckpt'))
            LOGGER.info(
                f"New best accuracy: {best_accuracy:.4f}."
                f"Model saved to {params.save_path} with `_best` postfix."
            )

        LOGGER.info(
            f"Epoch {epoch} completed. Latest model saved to {params.save_path} with `_last` postfix."
            f"Best accuracy: {best_accuracy:.4f}"
        )

    LOGGER.info('Training completed.')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
