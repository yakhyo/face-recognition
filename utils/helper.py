import torch


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
