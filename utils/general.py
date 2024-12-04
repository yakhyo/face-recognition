import os
import torch
import torch.distributed as distributed


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


@torch.no_grad()
def calculate_accuracy(output, target):
    """Computes the accuracy for the top-1 prediction."""
    batch_size = target.size(0)

    # Get the index of the max log-probability
    _, prediction = output.topk(1, 1, True, True)
    prediction = prediction.t()
    correct = prediction.eq(target.view(1, -1))

    # Calculate top-1 accuracy
    correct_count = correct.reshape(-1).float().sum(0, keepdim=True)
    accuracy = correct_count.mul_(100.0 / batch_size)

    return accuracy


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    """
    Initializes the distributed mode for multi-GPU training.

    Args:
        args: Argument parser object with the necessary attributes.
    """
    # Check for distributed environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Distributed mode not enabled. Falling back to single process.")
        args.distributed = False
        return

    args.distributed = True

    # Set the device
    torch.cuda.set_device(args.local_rank)
    print(f"| Distributed initialization (rank {args.rank}): env://", flush=True)

    # Initialize the process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank
    )
    setup_for_distributed(args.rank == 0)


def reduce_tensor(tensor, n):
    """Getting the average of tensors over multiple GPU devices
    Args:
        tensor: input tensor
        n: world size (number of gpus)
    Returns:
        reduced tensor
    """
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= n
    return rt
