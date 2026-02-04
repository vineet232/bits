import torch
import numpy as np
from tqdm import tqdm


class EarlyStopping:
    # Utility class to stop training when validation loss
    # stops improving for a fixed number of epochs.
    

    def __init__(self, patience=7):
        # Number of consecutive epochs to wait before stopping
        self.patience = patience

        # Stores the best (lowest) validation loss seen so far
        self.best = None

        # Counts epochs without improvement
        self.counter = 0

        # Flag indicating whether training should stop
        self.stop = False

    def __call__(self, val_loss):
        # Updates early stopping state based on current validation loss.
     
        # First epoch or improvement in validation loss
        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            # No improvement observed
            self.counter += 1

            # Trigger early stopping if patience is exceeded
            if self.counter >= self.patience:
                self.stop = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    # Runs one full training epoch.
    # Performs forward pass, loss computation,
    # backpropagation, and parameter updates.

    # Enable training mode (dropout, batchnorm active)
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    # Progress bar for training batches
    pbar = tqdm(loader, desc="Training", leave=False)

    for x, y in pbar:
        # Move input data and labels to target device
        x, y = x.to(device), y.to(device)

        # Clear gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass
        out = model(x)

        # Compute loss
        loss = criterion(out, y)

        # Backward pass (gradient computation)
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate loss for epoch-level reporting
        total_loss += loss.item()

        # Compute predictions and accuracy
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        # Update progress bar with live metrics
        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    # Return average loss and accuracy for the epoch
    return total_loss / len(loader), correct / total


def eval_one_epoch(model, loader, criterion, device):
    # Runs one evaluation epoch without gradient updates.
    # Used for validation or testing.

    # Set model to evaluation mode
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    # Progress bar for validation batches
    pbar = tqdm(loader, desc="Validation", leave=False)

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for x, y in pbar:
            # Move data to target device
            x, y = x.to(device), y.to(device)

            # Forward pass only
            out = model(x)

            # Compute validation loss
            loss = criterion(out, y)

            # Accumulate loss and accuracy
            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            # Update progress bar
            pbar.set_postfix(loss=loss.item(), acc=correct / total)

    # Return average validation loss and accuracy
    return total_loss / len(loader), correct / total
