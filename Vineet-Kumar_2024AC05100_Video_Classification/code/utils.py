import torch
import numpy as np
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss):
        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct/total)

    return total_loss/len(loader), correct/total



def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            pbar.set_postfix(loss=loss.item(), acc=correct/total)

    return total_loss/len(loader), correct/total

