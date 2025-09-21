import argparse
import time
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# Import PyTorch XLA libraries
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime

# FIX FOR ACCURACY: Version-compatible precision setting
import sys
import os

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"torch_xla: {torch_xla.__version__}")

# Try the new API first (PyTorch/XLA 2.2+)
precision_set = False
try:
    import torch_xla.backends
    torch_xla.backends.set_mat_mul_precision("high")
    print("✓ Using torch_xla.backends.set_mat_mul_precision('high')")
    precision_set = True
except (ImportError, AttributeError) as e:
    print(f"✗ torch_xla.backends not available: {e}")
    print("Using alternative approach...")

# Fallback approaches for older versions
if not precision_set:
    # Option 1: Environment variables (works across most versions)
    os.environ['XLA_USE_BF16'] = '0'  # Force FP32 instead of BF16
    print("✓ Set environment variable XLA_USE_BF16=0")

    # Option 2: Try PyTorch's built-in precision control
    try:
        torch.set_float32_matmul_precision('high')
        print("✓ Set PyTorch matmul precision to 'high'")
    except AttributeError:
        print("✗ torch.set_float32_matmul_precision not available")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

        nn.Linear(9216, 10)
        # example: batch_size = 128
        # (batch_size, 9216) @ (9216, 10) = (batch_size, 10)
        # (128, 9216) @ (9216, 10) = (128, 10)
        # output = XW + b


        nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    para_loader = pl.ParallelLoader(train_loader, [device])

    for batch_idx, (data, target) in enumerate(para_loader.per_device_loader(device)):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        # Use xm.optimizer_step to update weights on the TPU.
        xm.optimizer_step(optimizer)

        # Logging should only happen on the master process.
        if batch_idx % args.log_interval == 0 and xm.is_master_ordinal():
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) * torch_xla.runtime.world_size(), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    para_loader = pl.ParallelLoader(test_loader, [device])

    with torch.no_grad():
        for data, target in para_loader.per_device_loader(device):
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # CRITICAL: Reduce test loss and correct counts across all TPU cores.
    test_loss = xm.mesh_reduce('test_loss', test_loss, sum)
    correct = xm.mesh_reduce('correct', correct, sum)

    test_loss /= len(test_loader.dataset)

    # Print test results on the master process.
    if xm.is_master_ordinal():
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def _mp_fn(rank, args):
    """Main training function that will be spawned on each TPU core."""
    torch.manual_seed(args.seed)

    # Acquire the XLA device for the current process.
    device = torch_xla.device()

    # Start timer (per-process)
    start_time = time.time()

    # Move the model to the XLA device.
    model = Net().to(device)

    # Create distributed samplers for the datasets.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        num_replicas=torch_xla.runtime.world_size(),
        rank=torch_xla.runtime.global_ordinal(),
        shuffle=True)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        datasets.MNIST('../data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        num_replicas=torch_xla.runtime.world_size(),
        rank=torch_xla.runtime.global_ordinal(),
        shuffle=False)

    # CRITICAL FIX: Remove drop_last=True and increase batch size
    train_loader = torch.utils.data.DataLoader(
        train_sampler.dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=1,
        drop_last=False)  # CHANGED: Don't drop incomplete batches

    test_loader = torch.utils.data.DataLoader(
        test_sampler.dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=1,
        drop_last=False)  # CHANGED: Don't drop incomplete batches

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        # Set epoch for distributed sampler to ensure proper shuffling
        train_sampler.set_epoch(epoch)

        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # Save the model only on the master process to avoid race conditions.
    if args.save_model and xm.is_master_ordinal():
        xm.save(model.state_dict(), "mnist_cnn.pt")

    # Ensure all cores have finished computation and then compute elapsed.
    try:
        xm.rendezvous('end_training_sync')
    except Exception:
        pass

    end_time = time.time()
    elapsed = end_time - start_time

    # Reduce across all cores and take the maximum elapsed time (wall time).
    try:
        elapsed_tensor = torch.tensor(elapsed)
        total_elapsed_tensor = xm.mesh_reduce('total_elapsed_seconds', elapsed_tensor, max)
        if isinstance(total_elapsed_tensor, torch.Tensor):
            total_elapsed = float(total_elapsed_tensor.item())
        else:
            total_elapsed = float(total_elapsed_tensor)
    except Exception:
        total_elapsed = elapsed

    # Only master prints the final total time
    if xm.is_master_ordinal():
        formatted = str(timedelta(seconds=int(total_elapsed)))
        print(f"Total training time: {formatted} ({total_elapsed:.3f} seconds)")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example on XLA - FIXED for TPU Accuracy')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',  # CHANGED: Increased from 64
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Print batch size information for debugging
    print(f"Training batch size: {args.batch_size}")
    print(f"Per-core batch size: {args.batch_size // 8}")
    print(f"Test batch size: {args.test_batch_size}")

    # Use xmp.spawn to start the training on all available TPU cores.
    xmp.spawn(_mp_fn, args=(args,), nprocs=None, start_method='fork')


if __name__ == '__main__':
    main()