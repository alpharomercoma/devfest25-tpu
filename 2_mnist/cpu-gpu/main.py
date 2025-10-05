import argparse
import time
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.amp import autocast, GradScaler


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


def train(args, model, device, train_loader, optimizer, epoch, scaler=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Use mixed precision if enabled
        if scaler is not None:
            with autocast('cuda'):
                output = model(data)
                loss = F.nll_loss(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, scaler=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            # Use mixed precision for inference if enabled
            if scaler is not None:
                with autocast('cuda'):
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
            else:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example (Optimized)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--no-amp', action='store_true',
                        help='disables automatic mixed precision')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    args, _ = parser.parse_known_args()

    use_accel = not args.no_accel and torch.cuda.is_available()
    use_amp = use_accel and not args.no_amp

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.device("cuda")
        # Enable cuDNN benchmark mode for optimized convolution algorithms
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_accel:
        accel_kwargs = {
            'num_workers': args.num_workers,
            'persistent_workers': True if args.num_workers > 0 else False,
            'pin_memory': True,
            'shuffle': True,
            'prefetch_factor': 2 if args.num_workers > 0 else None
        }
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)
        test_kwargs['shuffle'] = False  # Don't shuffle test set

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (AMP)")

    # Start timing
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, scaler)
        test(model, device, test_loader, scaler)
        scheduler.step()

    # Synchronize device before measuring end time
    if use_accel:
        torch.cuda.synchronize()

    # End timing and print total
    total_time = time.time() - start_time
    formatted = str(timedelta(seconds=int(total_time)))
    print(f"\nTotal training time: {formatted} ({total_time:.3f} seconds)")

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print("Model saved to mnist_cnn.pt")


if __name__ == '__main__':
    main()