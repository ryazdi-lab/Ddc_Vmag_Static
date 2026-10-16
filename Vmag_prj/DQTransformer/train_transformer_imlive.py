"""
Training script for the pure-transformer DQ models (transformer_model.py).

Supports all three models:
    --model dq        -> DQTransformer              (global attention, plain MSE+MAE loss)
    --model axial     -> AxialTransformerDQNet       (row/col attention, plain MSE+MAE loss)
    --model centrality-> CentralityAwareTransformer   (row/col attention + centrality heads,
                                                        multi-task CentralityAwareLoss)

Mirrors the style of your existing training code: dataset.py for loading,
metrics.py for the plain loss/metrics, centrality_loss.py for the
multi-task loss/metrics when --model centrality is used.

Usage:
    python train_transformer.py --model axial --case 14 \
        --train dataset_train.npz --val dataset_val.npz --test dataset_test.npz

    python train_transformer.py --model centrality --case 30 \
        --train dataset_train.npz --val dataset_val.npz --test dataset_test.npz \
        --epochs 100 --lr 5e-4
"""

import argparse
import time
import threading

import torch
import torch.nn as nn

from dataset import get_dataloaders
from transformer_model import (
    DQTransformer, AxialTransformerDQNet, CentralityAwareTransformer,
    DQTransformerVmagNet, AxialTransformerVmagNet,
)
from metrics import CombinedMSEMAELoss, compute_metrics, aggregate_metrics


# ---------------------------------------------------------------------------
# Heartbeat (keep-alive prints for long-running remote sessions)
# ---------------------------------------------------------------------------

class Heartbeat:
    """Background daemon thread that prints a keep-alive message every
    `interval` seconds, so long epochs don't look "dead" to a server that
    kills idle/silent sessions."""

    def __init__(self, interval=120, msg="I'm alive"):
        self.interval = interval
        self.msg = msg
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop_event.wait(self.interval):
            print(f"[heartbeat] {self.msg} ({time.strftime('%H:%M:%S')})", flush=True)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, target_key: str, n_buses: int, d_model: int, depth: int, num_heads: int):
    if target_key == 'Vmag':
        if name == 'dq':
            return DQTransformerVmagNet(n_buses=n_buses, d_model=d_model, depth=depth, num_heads=num_heads)
        if name == 'axial':
            return AxialTransformerVmagNet(n_buses=n_buses, d_model=d_model, depth=depth, num_heads=num_heads)
        raise ValueError(
            f"--target_key Vmag requires --model dq or --model axial (got --model {name!r}); "
            f"the centrality architecture's heads are built around the DQ matrix decomposition "
            f"and have no per-bus vector head."
        )
    if name == 'dq':
        return DQTransformer(n_buses=n_buses, d_model=d_model, depth=depth, num_heads=num_heads)
    if name == 'axial':
        return AxialTransformerDQNet(n_buses=n_buses, d_model=d_model, depth=depth, num_heads=num_heads)
    if name == 'centrality':
        return CentralityAwareTransformer(n_buses=n_buses, d_model=d_model, depth=depth, num_heads=num_heads)
    raise ValueError(f"Unknown --model '{name}'")


# ---------------------------------------------------------------------------
# One epoch of training / evaluation
# ---------------------------------------------------------------------------

def run_epoch(
    model:      nn.Module,
    loader,
    device:     torch.device,
    is_centrality: bool,
    criterion,
    optimizer=None,   # None => eval mode, no backward pass
    scaler=None,      # torch.cuda.amp.GradScaler, or None to disable AMP
    checkpoint_fn=None,        # optional callable(batch_idx) for mid-epoch checkpointing
    checkpoint_every_n_batches: int = 0,   # 0 disables mid-epoch checkpointing
):
    train_mode = optimizer is not None
    model.train(train_mode)
    use_amp = scaler is not None and scaler.is_enabled()

    running_loss = 0.0
    n_samples    = 0
    batch_metrics = []

    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if is_centrality:
                    outputs = model(X)
                    loss, _ = criterion(outputs, y)
                else:
                    pred = model(X)
                    loss = criterion(pred, y)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item() * X.size(0)
        n_samples    += X.size(0)

        with torch.no_grad():
            if is_centrality:
                from centrality_loss import compute_full_metrics
                m = compute_full_metrics(outputs, y)
                batch_metrics.append({
                    'r2':   m['matrix_r2'],
                    'rmse': m['matrix_rmse'],
                    'mae':  m['matrix_mae'],
                })
            else:
                batch_metrics.append(compute_metrics(pred, y))

        if (
            train_mode
            and checkpoint_fn is not None
            and checkpoint_every_n_batches > 0
            and (batch_idx + 1) % checkpoint_every_n_batches == 0
        ):
            checkpoint_fn(batch_idx + 1)

    avg_loss = running_loss / n_samples
    avg_metrics = aggregate_metrics(batch_metrics)
    return avg_loss, avg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a pure-transformer DQ estimation model.")

    parser.add_argument('--model', choices=['dq', 'axial', 'centrality'], default='axial',
                         help="Which transformer architecture to train.")
    parser.add_argument('--target_key', choices=['y', 'Vmag'], default='y',
                         help="Which target to predict: 'y' for the DQ matrix "
                              "(works with any --model), 'Vmag' for the per-bus "
                              "voltage magnitude vector (requires --model dq or "
                              "--model axial; centrality has no vector head).")
    parser.add_argument('--case', type=int, default=14,
                         help="Number of buses (matrix side length), e.g. 14, 30, 39, 57, 118.")

    parser.add_argument('--train', type=str, default='dataset_train.npz')
    parser.add_argument('--val',   type=str, default='dataset_val.npz')
    parser.add_argument('--test',  type=str, default='dataset_test.npz')

    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience',   type=int,   default=35,
                         help="Early-stopping patience (epochs without val R2 improvement).")

    parser.add_argument('--d_model',   type=int, default=96)
    parser.add_argument('--depth',     type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=6)

    parser.add_argument('--checkpoint', type=str, default=None,
                         help="Path to save the best model's state_dict. "
                              "Defaults to '{model}_case{N}_best.pt'.")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                         help="Path to a full training-state checkpoint (model+optimizer+scheduler+"
                              "epoch etc.) to resume from. Defaults to '{model}_case{N}_resume.pt'.")
    parser.add_argument('--resume', action='store_true',
                         help="Resume training from --resume_checkpoint if it exists.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--heartbeat_interval', type=int, default=120,
                         help="Seconds between 'I'm alive' keep-alive prints during training "
                              "(0 disables it).")

    parser.add_argument('--amp', dest='amp', action='store_true', default=True,
                         help="Use automatic mixed precision (fp16) on CUDA. On by default "
                              "when a GPU is available; roughly halves memory and can "
                              "meaningfully speed up attention-heavy models. No effect on CPU.")
    parser.add_argument('--no_amp', dest='amp', action='store_false',
                         help="Disable mixed precision.")
    parser.add_argument('--checkpoint_every_n_batches', type=int, default=0,
                         help="If >0, save the full resumable checkpoint every N training "
                              "batches (not just at epoch end). Use this for very long epochs "
                              "so a crash mid-epoch doesn't cost you the whole epoch. A "
                              "mid-epoch checkpoint resumes at the START of that same epoch "
                              "(not the exact batch), but keeps the weight/optimizer progress "
                              "made up to that point.")
    parser.add_argument('--compile', action='store_true', default=False,
                         help="Wrap the model with torch.compile() for kernel fusion (PyTorch "
                              "2.0+, CUDA). First batch is slower (compilation warmup) but "
                              "subsequent batches are typically 20-40%% faster. Falls back to "
                              "an uncompiled model with a warning if compilation fails.")

    args = parser.parse_args()
    run_tag = args.model if args.target_key == 'y' else f"{args.model}_{args.target_key.lower()}"
    if args.checkpoint is None:
        args.checkpoint = f"{run_tag}_case{args.case}_best.pt"
    if args.resume_checkpoint is None:
        args.resume_checkpoint = f"{run_tag}_case{args.case}_resume.pt"

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    use_amp = args.amp and device.type == 'cuda'
    print(f"Device: {device} | Model: {args.model} | Case: {args.case} | AMP: {use_amp}\n")

    # --- Data ---
    train_loader, val_loader, test_loader = get_dataloaders(
        args.train, args.val, args.test, batch_size=args.batch_size,
        target_key=args.target_key, expected_n_buses=args.case,
    )

    # --- Model ---
    model = build_model(args.model, args.target_key, args.case, args.d_model, args.depth, args.num_heads).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile() enabled.\n")
        except Exception as e:
            print(f"torch.compile() failed ({e}); continuing without compilation.\n")

    is_centrality = (args.model == 'centrality')
    if is_centrality:
        from centrality_loss import CentralityAwareLoss
        criterion = CentralityAwareLoss()
    else:
        criterion = CombinedMSEMAELoss(alpha=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Linear warmup -> cosine decay (transformers are sensitive to LR at the start)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training loop with early stopping on val R2 ---
    best_val_r2 = -float('inf')
    epochs_without_improvement = 0
    start_epoch = 0

    if args.resume:
        try:
            ckpt = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if use_amp and ckpt.get('scaler_state_dict') is not None:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            best_val_r2 = ckpt['best_val_r2']
            epochs_without_improvement = ckpt['epochs_without_improvement']
            start_epoch = ckpt['epoch'] + 1
            try:
                torch.set_rng_state(ckpt['rng_state'].cpu().to(torch.uint8))
                if torch.cuda.is_available() and ckpt.get('cuda_rng_state') is not None:
                    torch.cuda.set_rng_state_all(
                        [t.cpu().to(torch.uint8) for t in ckpt['cuda_rng_state']]
                    )
            except Exception as e:
                print(f"Warning: could not restore RNG state ({e}); continuing with fresh RNG state.")
            print(
                f"Resumed from '{args.resume_checkpoint}' at epoch {start_epoch} "
                f"(best_val_r2={best_val_r2:.4f}, epochs_without_improvement={epochs_without_improvement})\n"
            )
        except FileNotFoundError:
            print(
                f"--resume was set but no resume checkpoint found at '{args.resume_checkpoint}'. "
                f"Starting from scratch.\n"
            )

    heartbeat = None
    if args.heartbeat_interval > 0:
        heartbeat = Heartbeat(interval=args.heartbeat_interval).start()

    try:
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()

            def _mid_epoch_checkpoint(batch_idx, _epoch=epoch):
                # Resumes at the START of this same epoch (start_epoch = _epoch, not _epoch+1),
                # but keeps whatever weight/optimizer progress was made up to this batch.
                torch.save({
                    'epoch': _epoch - 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if use_amp else None,
                    'best_val_r2': best_val_r2,
                    'epochs_without_improvement': epochs_without_improvement,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'args': vars(args),
                }, args.resume_checkpoint)
                print(f"  [mid-epoch checkpoint] saved after {batch_idx} batches of epoch {_epoch + 1}", flush=True)

            train_loss, train_metrics = run_epoch(
                model, train_loader, device, is_centrality, criterion, optimizer=optimizer,
                scaler=scaler,
                checkpoint_fn=_mid_epoch_checkpoint,
                checkpoint_every_n_batches=args.checkpoint_every_n_batches,
            )
            val_loss, val_metrics = run_epoch(
                model, val_loader, device, is_centrality, criterion, optimizer=None,
            )
            scheduler.step()

            dt = time.time() - t0
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"train_loss={train_loss:.5f} r2={train_metrics['r2']:.4f} | "
                f"val_loss={val_loss:.5f} r2={val_metrics['r2']:.4f} rmse={val_metrics['rmse']:.5f} "
                f"mae={val_metrics['mae']:.5f} | {dt:.1f}s"
            )

            if val_metrics['r2'] > best_val_r2:
                best_val_r2 = val_metrics['r2']
                epochs_without_improvement = 0
                torch.save(model.state_dict(), args.checkpoint)
            else:
                epochs_without_improvement += 1

            # Save full resumable state after every epoch (cheap insurance against crashes).
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_amp else None,
                'best_val_r2': best_val_r2,
                'epochs_without_improvement': epochs_without_improvement,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'args': vars(args),
            }, args.resume_checkpoint)

            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping: no val R2 improvement for {args.patience} epochs.")
                break
    finally:
        if heartbeat is not None:
            heartbeat.stop()

    # --- Final test evaluation using the best checkpoint ---
    print(f"\nLoading best checkpoint (val R2={best_val_r2:.4f}) from '{args.checkpoint}'")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_loss, test_metrics = run_epoch(
        model, test_loader, device, is_centrality, criterion, optimizer=None,
    )

    print("\n" + "=" * 60)
    print(f"TEST RESULTS  [{args.model}, case{args.case}]")
    print("=" * 60)
    print(f"  loss : {test_loss:.5f}")
    print(f"  R2   : {test_metrics['r2']:.4f}")
    print(f"  RMSE : {test_metrics['rmse']:.6f}")
    print(f"  MAE  : {test_metrics['mae']:.6f}")

    if is_centrality:
        from centrality_loss import compute_full_metrics
        # One more pass to report the full centrality breakdown on the whole test set
        model.eval()
        all_outputs = {k: [] for k in ['dq_pred', 'betweenness_decoder', 'closeness_decoder',
                                        'betweenness_head', 'closeness_head']}
        all_targets = []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                for k in all_outputs:
                    all_outputs[k].append(out[k])
                all_targets.append(y)
        all_outputs = {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}
        all_targets = torch.cat(all_targets, dim=0)

        full_metrics = compute_full_metrics(all_outputs, all_targets)
        print("\n  Centrality-level metrics (betweenness + closeness):")
        print(f"    R2   : {full_metrics['centrality_r2']:.4f}")
        print(f"    RMSE : {full_metrics['centrality_rmse']:.6f}")
        print(f"    MAE  : {full_metrics['centrality_mae']:.6f}")
        print("\n  FC-head centrality metrics:")
        print(f"    R2   : {full_metrics['head_r2']:.4f}")
        print(f"    RMSE : {full_metrics['head_rmse']:.6f}")
        print(f"    MAE  : {full_metrics['head_mae']:.6f}")


if __name__ == "__main__":
    main()
