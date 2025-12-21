# pytorch_env — project environment

Quick setup & usage

1) Activate the conda environment

```powershell
conda activate pytorch
```

2) Verify PyTorch + CUDA

```powershell
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

3) Recreate this environment

- Reproducible (minimal pinned) environment is in `environment.yml` in this repo:

```powershell
conda env create -f environment.yml
conda activate pytorch
```

4) VS Code

- I added a `.vscode/settings.json` that points VS Code to the environment's Python interpreter. If VS Code doesn't auto-select it, set the interpreter to:

```
C:\Users\David\Miniconda3\envs\pytorch\python.exe
```

Notes

- This environment uses PyTorch 2.5.1 built for CUDA 12.4. Your machine uses NVIDIA driver 591.44 and an RTX 4070, which is compatible.
- If you prefer a different PyTorch/CUDA combo, activate the env and install with `mamba`/`conda` using the `pytorch` and `nvidia` channels.
