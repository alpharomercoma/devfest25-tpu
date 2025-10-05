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

# Use the newer API
try:
    from torch_xla import sync as xla_sync
    USE_NEW_SYNC_API = True
except ImportError:
    USE_NEW_SYNC_API = False

import sys
import os

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"torch_xla: {torch_xla.__version__}")

# Version-compatible precision setting
precision_set = False
try:
    import torch_xla.backends
    torch_xla.backends.set_mat_mul_precision("high")
    print("✓ Using torch_xla.backends.set_mat_mul_precision('high')")
    precision_set = True
except (ImportError, AttributeError) as e:
    print(f"✗ torch_xla.backends not available: {e}")

if not precision_set:
    os.environ['XLA_USE_BF16'] = '0'
    print("✓ Set environment variable XLA_USE_BF16=0")
    try:
        torch.set_float32_matmul_precision('high')
        print("✓ Set PyTorch matmul precision to 'high'")
    except AttributeError:
        pass


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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

    # Use MpDeviceLoader for better performance
    train_device_loader = para_loader.per_device_loader(device)

    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_device_loader):
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        # CRITICAL: Use xm.optimizer_step to update weights on TPU
        xm.optimizer_step(optimizer)

        # CRITICAL: Sync for XLA graph compilation efficiency
        if USE_NEW_SYNC_API:
            xla_sync()
        else:
            xm.mark_step()

        total_samples += len(data)

        # Logging - reduce frequency of .item() calls to minimize sync overhead
        if batch_idx % args.log_interval == 0 and xm.is_master_ordinal():
            # Only sync loss to CPU when we actually need to log it
            xm.add_step_closure(
                lambda loss_val=loss, batch=batch_idx, samples=total_samples:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    samples * torch_xla.runtime.world_size(),
                    len(train_loader.dataset),
                    100. * batch / len(train_loader),
                    loss_val.item()))
            )

            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    para_loader = pl.ParallelLoader(test_loader, [device])
    test_device_loader = para_loader.per_device_loader(device)

    # Accumulate on device to minimize host-device transfers
    test_loss_sum = torch.tensor(0.0, device=device)
    correct_count = torch.tensor(0, device=device)

    with torch.no_grad():
        for data, target in test_device_loader:
            output = model(data)

            # Accumulate loss on device
            test_loss_sum += F.nll_loss(output, target, reduction='sum')

            # Compute accuracy on device
            pred = output.argmax(dim=1, keepdim=True)
            correct_count += pred.eq(target.view_as(pred)).sum()

    # Sync after test loop
    if USE_NEW_SYNC_API:
        xla_sync()
    else:
        xm.mark_step()

    # CRITICAL: Reduce across all TPU cores
    test_loss_total = xm.mesh_reduce('test_loss', test_loss_sum, sum).item()
    correct_total = xm.mesh_reduce('correct', correct_count, sum).item()

    test_loss_avg = test_loss_total / len(test_loader.dataset)

    # Print on master process only
    if xm.is_master_ordinal():
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss_avg,
            int(correct_total),
            len(test_loader.dataset),
            100. * correct_total / len(test_loader.dataset)))


def _mp_fn(rank, args):
    """Main training function spawned on each TPU core."""
    torch.manual_seed(args.seed)

    # Acquire XLA device
    device = torch_xla.device()

    if xm.is_master_ordinal():
        print(f"\n{'='*60}")
        print(f"TPU Training Configuration:")
        print(f"  World size (TPU cores): {torch_xla.runtime.world_size()}")
        print(f"  Global batch size: {args.batch_size}")
        print(f"  Per-core batch size: {args.batch_size // torch_xla.runtime.world_size()}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning rate: {args.lr}")
        print(f"{'='*60}\n")

    start_time = time.time()

    # Move model to XLA device
    model = Net().to(device)

    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create datasets
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    # Create distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=torch_xla.runtime.world_size(),
        rank=torch_xla.runtime.global_ordinal(),
        shuffle=True,
        drop_last=False  # Don't drop incomplete batches
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=torch_xla.runtime.world_size(),
        rank=torch_xla.runtime.global_ordinal(),
        shuffle=False,
        drop_last=False
    )

    # Optimized DataLoader settings for TPU
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size // torch_xla.runtime.world_size(),  # Per-core batch size
        sampler=train_sampler,
        num_workers=4,  # Increased for better I/O
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2,  # Prefetch batches
        drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size // torch_xla.runtime.world_size(),
        sampler=test_sampler,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)  # Ensure proper shuffling

        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # Wait for all device operations to complete before timing
    xm.wait_device_ops()

    # Save model on master process only
    if args.save_model and xm.is_master_ordinal():
        xm.save(model.state_dict(), "mnist_cnn.pt")
        print("Model saved to mnist_cnn.pt")

    # Synchronize all cores
    xm.rendezvous('training_complete')

    end_time = time.time()
    elapsed = end_time - start_time

    # Get max elapsed time across all cores (wall clock time)
    elapsed_tensor = torch.tensor(elapsed, device=device)
    max_elapsed = xm.mesh_reduce('max_elapsed', elapsed_tensor, max).item()

    # Master process prints final timing
    if xm.is_master_ordinal():
        formatted = str(timedelta(seconds=int(max_elapsed)))
        print(f"\n{'='*60}")
        print(f"Total training time: {formatted} ({max_elapsed:.3f} seconds)")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST on TPU v4-8 (Fully Optimized)')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='global batch size for training (default: 512, 64 per core)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='global batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='batches between logging (default: 20, increased to reduce overhead)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save the trained model')
    args = parser.parse_args()

    # Spawn training on all TPU cores
    xmp.spawn(_mp_fn, args=(args,), nprocs=None, start_method='fork')


if __name__ == '__main__':
    main()