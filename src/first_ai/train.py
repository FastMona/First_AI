"""Generic training utilities (scaffold).

Provide a minimal, reusable training loop that future classifier trainers can use.
"""

from typing import Callable, Dict, Any

try:
    import torch
except Exception:
    torch = None  # type: ignore


def fit(
    model,
    train_loader,
    loss_fn: Callable,
    optimizer,
    device: str,
    scaler=None,
) -> float:
    """Run one training epoch; return average loss."""
    if torch is None:
        raise RuntimeError("torch not available")

    model.train()
    running = 0.0
    count = 0
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(X)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
        running += loss.item() * X.size(0)
        count += X.size(0)
    return running / max(count, 1)


def evaluate(model, loader, loss_fn: Callable, device: str) -> Dict[str, Any]:
    """Evaluate model; return dict with avg_loss."""
    if torch is None:
        raise RuntimeError("torch not available")

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            loss = loss_fn(out, y)
            total += loss.item() * X.size(0)
            count += X.size(0)
    return {"avg_loss": total / max(count, 1)}
