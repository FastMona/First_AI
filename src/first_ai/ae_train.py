"""Autoencoder training utilities (scaffold)."""

try:
    import torch
except Exception:
    torch = None  # type: ignore


def train_autoencoder(autoencoder, loader, optimizer, device: str, epochs: int = 5, scaler=None) -> None:
    if torch is None:
        raise RuntimeError("torch not available")
    for _ in range(epochs):
        autoencoder.train()
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    reconstruction = autoencoder(X, y)
                    loss = torch.nn.functional.mse_loss(reconstruction, X)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                reconstruction = autoencoder(X, y)
                loss = torch.nn.functional.mse_loss(reconstruction, X)
                loss.backward()
                optimizer.step()
