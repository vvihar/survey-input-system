from __future__ import annotations

import argparse
import csv
import pickle
import random
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "torchvision is required for MNIST training. Install it with: pip install torchvision"
    ) from exc


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MnistNet(nn.Module):
    """
    Architecture intentionally matches survey_pipeline/digit_model.py.
    Input:  N x 1 x 28 x 28, normalized by (x - 0.1307) / 0.3081
    Output: N x 10 logits
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    device: torch.device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_mnist_loaders(
    root: Path,
    batch_size: int,
    num_workers: int,
    val_size: int = 5000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=12,
                translate=(0.12, 0.12),
                scale=(0.85, 1.15),
                shear=6,
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    full_train = datasets.MNIST(
        root=str(root), train=True, download=True, transform=train_tf
    )
    full_train_for_val = datasets.MNIST(
        root=str(root), train=True, download=True, transform=eval_tf
    )

    n_train = len(full_train) - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, _ = random_split(full_train, [n_train, val_size], generator=generator)
    _, val_subset = random_split(
        full_train_for_val, [n_train, val_size], generator=generator
    )

    test_ds = datasets.MNIST(
        root=str(root), train=False, download=True, transform=eval_tf
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def normalize_digit_image_for_pipeline(img: np.ndarray) -> np.ndarray:
    """
    Convert a labeled crop into the same 28x28 style used by
    survey_pipeline.digit_model.normalize_digit_for_mnist:
    black background, white digit, float32 in [0, 1].
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Infer polarity. If background is mostly white, invert after thresholding.
    # Otsu with both polarities is enough for debug digit crops and full crop images.
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 127:
        th = 255 - th

    # Remove small speckles.
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    coords = cv2.findNonZero(th)
    if coords is None:
        return np.zeros((28, 28), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    digit = th[y : y + h, x : x + w]

    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x0 = (28 - new_w) // 2
    y0 = (28 - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas.astype(np.float32) / 255.0


class LabeledDigitCropDataset(Dataset[tuple[torch.Tensor, int]]):
    """
    Optional fine-tuning dataset for real survey crops.

    Supported formats:
      1. directory format:
         labeled_digits/0/*.png
         labeled_digits/1/*.png
         ...
         labeled_digits/9/*.png

      2. CSV format with columns:
         path,label
         debug_digits/a.png,3
         debug_digits/b.png,7

    The images may be either already-normalized 28x28 digit images or larger
    crop images. They are converted to the pipeline's 28x28 representation.
    """

    def __init__(self, root: Path | None = None, csv_path: Path | None = None) -> None:
        self.samples: list[tuple[Path, int]] = []

        if root is not None:
            root = root.expanduser().resolve()
            for label in range(10):
                label_dir = root / str(label)
                if not label_dir.exists():
                    continue
                for p in sorted(label_dir.glob("**/*")):
                    if p.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                        ".bmp",
                        ".tif",
                        ".tiff",
                    }:
                        self.samples.append((p, label))

        if csv_path is not None:
            csv_path = csv_path.expanduser().resolve()
            base = csv_path.parent
            with csv_path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                if "path" not in fieldnames or "label" not in fieldnames:
                    raise ValueError("fine-tune CSV must have columns: path,label")
                for row in reader:
                    p = Path(row["path"])
                    if not p.is_absolute():
                        p = base / p
                    self.samples.append((p, int(row["label"])))

        if not self.samples:
            raise ValueError("No labeled crop images found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        arr = normalize_digit_image_for_pipeline(img)
        x = torch.from_numpy(arr).float().unsqueeze(0)
        x = (x - MNIST_MEAN) / MNIST_STD
        return x, label


def make_crop_loaders(
    dataset: Dataset[Any],
    batch_size: int,
    num_workers: int,
    val_ratio: float = 0.2,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    if not isinstance(dataset, Sized):
        raise TypeError("dataset must implement __len__")

    n_val = max(1, int(round(len(dataset) * val_ratio)))
    n_train = len(dataset) - n_val
    if n_train <= 0:
        raise ValueError("Need at least two labeled crop images for fine-tuning.")
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)  # type: ignore[arg-type]
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    )


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> tuple[int, int]:
    pred = logits.argmax(dim=1)
    correct = int((pred == y).sum().item())
    total = int(y.numel())
    return correct, total


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    correct_sum = 0
    total_sum = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            if is_train:
                opt = optimizer
                assert opt is not None
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        correct, total = accuracy_from_logits(logits, y)
        loss_sum += float(loss.item()) * total
        correct_sum += correct
        total_sum += total

    return loss_sum / total_sum, correct_sum / total_sum


def save_checkpoint(
    path: Path, model: nn.Module, epoch: int, val_acc: float, args: argparse.Namespace
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": "MnistNet-v1-compatible-with-survey_pipeline.digit_model",
            "epoch": epoch,
            "val_acc": val_acc,
            "normalization": {"mean": MNIST_MEAN, "std": MNIST_STD},
            "args": vars(args),
        },
        path,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    out_path: Path,
    args: argparse.Namespace,
    phase_name: str,
) -> float:
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs)
    )

    best_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, cfg.device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, cfg.device, optimizer=None)
        scheduler.step()

        print(
            f"[{phase_name}] epoch={epoch:03d}/{cfg.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(out_path, model, epoch, val_acc, args)
            print(f"  saved best: {out_path} val_acc={val_acc:.4f}")

    return best_acc


def torch_load_compatible(path: Path, device: torch.device) -> Any:
    """
    PyTorch 2.6+ の weights_only=True 問題に対応した読み込み。
    自分で作成した checkpoint だけ fallback を許す。
    """
    try:
        return torch.load(path, map_location=device)
    except pickle.UnpicklingError as exc:
        msg = str(exc)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        return torch.load(path, map_location=device, weights_only=False)


def load_checkpoint_if_given(
    model: nn.Module,
    path: Path | None,
    device: torch.device,
) -> None:
    if path is None or not path.exists():
        return

    obj = torch_load_compatible(path, device)

    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj

    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)


def export_onnx(model: nn.Module, out_path: Path, device: torch.device) -> None:
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_any: Any = dummy
    torch.onnx.export(
        model,
        dummy_any,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"saved ONNX: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MnistNet for the survey input system. Saves a model compatible with survey_pipeline.digit_model.load_model()."
    )
    parser.add_argument("--out", type=Path, default=Path("models/mnist.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path(".cache/mnist"))
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Optional checkpoint to resume/fine-tune from.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=int, default=5000)

    parser.add_argument(
        "--fine-tune-dir",
        type=Path,
        default=None,
        help="Optional labeled crop directory: root/0/*.png ... root/9/*.png",
    )
    parser.add_argument(
        "--fine-tune-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns path,label",
    )
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-lr", type=float, default=2e-4)
    parser.add_argument("--fine-tune-val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--skip-mnist",
        action="store_true",
        help="Use only the labeled crop dataset. Requires --pretrained or enough crop data.",
    )

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--export-onnx", type=Path, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"device={device}")

    model = MnistNet().to(device)
    load_checkpoint_if_given(model, args.pretrained, device)

    if not args.skip_mnist:
        train_loader, val_loader, test_loader = make_mnist_loaders(
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_size=args.val_size,
        )
        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            device=device,
        )
        train_model(
            model, train_loader, val_loader, cfg, args.out, args, phase_name="mnist"
        )

        # Reload best before test / fine-tuning.
        load_checkpoint_if_given(model, args.out, device)
        test_loss, test_acc = run_epoch(model, test_loader, device, optimizer=None)
        print(f"[mnist] test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    if args.fine_tune_dir is not None or args.fine_tune_csv is not None:
        crop_ds = LabeledDigitCropDataset(
            root=args.fine_tune_dir, csv_path=args.fine_tune_csv
        )
        crop_train, crop_val = make_crop_loaders(
            crop_ds,
            batch_size=min(args.batch_size, max(1, len(crop_ds))),
            num_workers=args.num_workers,
            val_ratio=args.fine_tune_val_ratio,
        )
        cfg = TrainConfig(
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            lr=args.fine_tune_lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            device=device,
        )
        train_model(
            model, crop_train, crop_val, cfg, args.out, args, phase_name="fine-tune"
        )

    if args.export_onnx is not None:
        load_checkpoint_if_given(model, args.out, device)
        export_onnx(model, args.export_onnx, device)

    print(f"done. compatible checkpoint: {args.out}")


if __name__ == "__main__":
    main()
