"""Shared PyTorch training for EEG classifiers with a temporal convolution."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


VOLTS_TO_MICROVOLTS = 1e6


def make_loader(X, y, indices, batch_size, shuffle_seed=None):
    signals = torch.from_numpy(X[indices].astype(np.float32)).mul_(VOLTS_TO_MICROVOLTS)
    labels = torch.as_tensor(y[indices], dtype=torch.long)
    generator = None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed)
    return DataLoader(TensorDataset(signals, labels), batch_size=batch_size,
                      shuffle=generator is not None, generator=generator)


def run_epoch(model, loader, criterion, device, optimizer=None, batch_check=None):
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    predictions = []

    with torch.set_grad_enabled(training):
        for batch, (signals, labels) in enumerate(loader):
            signals, labels = signals.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            if logits.shape != (len(labels), model.n_outputs):
                raise RuntimeError(f"Expected one logit per class and trial, got {tuple(logits.shape)}")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite {'training' if training else 'evaluation'} loss")

            if training:
                check_update = batch == 0 and batch_check is not None
                loss.backward()
                if check_update:
                    weight = model.conv_temporal.weight
                    before = weight.detach().clone()
                    gradient = weight.grad
                    if gradient is None or not torch.isfinite(gradient).all() or not gradient.any():
                        raise RuntimeError("Expected finite, nonzero first-batch gradients")
                optimizer.step()
                if check_update:
                    if torch.equal(weight, before):
                        raise RuntimeError("Adam did not update the temporal convolution")
                    batch_check.update(
                        input_shape=list(signals.shape), input_dtype=str(signals.dtype),
                        target_shape=list(labels.shape), target_dtype=str(labels.dtype),
                        logits_shape=list(logits.shape), loss=loss.item(),
                        temporal_weight_grad_norm=gradient.norm().item(), optimizer_update=True,
                    )

            predicted = logits.detach().argmax(dim=1)
            predictions.append(predicted)
            total_loss += loss.detach() * len(labels)
            correct += (predicted == labels).sum()

    count = len(loader.dataset)
    return total_loss.item() / count, correct.item() / count, torch.cat(predictions).cpu().numpy()


def fit(model, train, val, settings, device, checkpoint):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    history, batch_check = [], {}
    best_loss, best_epoch = float("inf"), 0
    stopped_early = False
    for epoch in range(1, settings["max_epochs"] + 1):
        train_loss, train_accuracy, _ = run_epoch(
            model, train, criterion, device, optimizer, batch_check if epoch == 1 else None,
        )
        if epoch == 1:
            print(f"  First training batch: {batch_check}", flush=True)
        val_loss, val_accuracy, _ = run_epoch(model, val, criterion, device)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_accuracy, "val_accuracy": val_accuracy,
        })
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), checkpoint)

        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: train loss={train_loss:.4f}, val loss={val_loss:.4f}, "
                  f"train accuracy={train_accuracy:.4f}, val accuracy={val_accuracy:.4f}", flush=True)
        if epoch - best_epoch >= settings["patience"]:
            stopped_early = epoch < settings["max_epochs"]
            break

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    selection = {
        "best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_run": len(history),
        "stopped_early": stopped_early, "first_training_batch": batch_check,
    }
    return pd.DataFrame(history), selection
