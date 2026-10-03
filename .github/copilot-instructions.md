# Copilot instructions

## Project shape

- This is a small, script-oriented PyTorch GPT training repository, not an installable Python package. Run commands from the repository root; scripts use paths relative to the current working directory.
- `model.py` defines the GPT model, causal attention, optimizer setup, GPT-2 weight import, and autoregressive generation. `train.py` owns the training/evaluation loop and supports single-process and PyTorch DDP training. `sample.py` loads a trained checkpoint or pretrained GPT-2 weights for generation; `bench.py` isolates model benchmarking.
- Data preparation lives under `data/<dataset>/prepare.py`. The scripts write `train.bin` and `val.bin` as `uint16` token IDs; `train.py` reads these with NumPy memory maps. Character-level Shakespeare also writes `meta.pkl` with its vocabulary mappings, which `sample.py` needs to decode its outputs. GPT-2 BPE datasets instead use `tiktoken` at sampling time.
- Configurations under `config/` are Python scripts executed into the calling script's global namespace by `configurator.py`; they are not imported as config objects. Apply a config first, then override existing settings with `--key=value` arguments. CLI values are parsed as Python literals where possible and must have the same type as the setting they override.
- Training checkpoints are saved as `<out_dir>/ckpt.pt`. Resume depends on the saved model, optimizer, iteration, best validation loss, model arguments, and config; sampling loads the model and model arguments, and may use the config's dataset name to find character vocabulary metadata. Preserve these fields when changing checkpoint serialization.
- DDP initialization uses the environment supplied by `torchrun`; only the master rank logs and saves checkpoints. Keep single-process behavior usable as well as the distributed path.

## Setup and commands

Install the dependencies listed by the README. The Shakespeare preparation scripts also import `requests`:

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm requests
```

Prepare the small character-level Shakespeare dataset and start its example training run:

```sh
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

On CPU, disable compilation as in the README (`torch.compile` may not be supported on every platform):

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Prepare OpenWebText and launch the README's 8-GPU GPT-2 reproduction command:

```sh
python data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

OpenWebText preparation downloads and processes a large dataset; it is not needed for the Shakespeare examples. Other dataset-specific training, evaluation, and fine-tuning configurations are in `config/`. The README documents sampling from pretrained GPT-2 variants and fine-tuning.

## Analysis notebooks and benchmarks

- `scaling_laws.ipynb` explores parameter/FLOP estimates and compute-optimal model/data tradeoffs from Chinchilla; `transformer_sizing.ipynb` estimates model parameters, checkpoint and optimizer memory, FLOPs, and MFU. These are analysis notebooks, not part of the training runtime or an automated test suite. When changing an assumption used by a calculation, rerun the dependent cells and keep it consistent with the formulas in `model.py` and `train.py`.
- `bench.py` is a standalone forward/backward and optimizer-step benchmark. It uses `configurator.py` overrides like the training scripts, defaults to `data/openwebtext/train.bin`, and synchronizes CUDA, so run it with CUDA available. Use `--real_data=False` to benchmark with generated token IDs instead of preparing OpenWebText; `--profile=True` enables PyTorch profiling.

Example synthetic-data benchmark on a CUDA machine:

```sh
python bench.py --real_data=False --device=cuda --compile=False --batch_size=2 --block_size=128
```

## Validation

There is no configured test suite, per-test runner, or linter in this repository. For a small end-to-end CPU smoke check, first prepare `shakespeare_char`, then run:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --dtype=float32 --eval_iters=1 --eval_interval=1 --log_interval=1 --batch_size=2 --block_size=32 --n_layer=1 --n_head=1 --n_embd=32 --max_iters=1 --lr_decay_iters=1 --always_save_checkpoint=True --out_dir=out-smoke
```

This exercises data loading, forward/backward training, evaluation, and checkpoint writing without requiring a GPU.
