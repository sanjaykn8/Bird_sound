from __future__ import annotations

import os
import gc
import math
import random
import warnings
import time
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


class CFG:
    COMP_DIR = Path(os.environ.get("BIRDCLEF_DIR", "/kaggle/input/competitions/birdclef-2026"))
    TRAIN_AUDIO_DIR = COMP_DIR / "train_audio"
    TRAIN_SOUNDSCAPE_DIR = COMP_DIR / "train_soundscapes"
    TRAIN_CSV = COMP_DIR / "train.csv"
    SOUNDSCAPE_CSV = COMP_DIR / "train_soundscapes_labels.csv"
    TEST_DIR = COMP_DIR / "test_soundscapes"
    SAMPLE_SUB = COMP_DIR / "sample_submission.csv"
    OUTPUT_DIR = Path(os.environ.get("BIRDCLEF_OUT", "/kaggle/working"))

    SR = 32_000
    DURATION = 3
    N_SAMPLES = SR * DURATION

    N_FFT = 1024
    HOP_LENGTH = 256
    N_MELS = 96
    FMIN = 20
    FMAX = 16_000

    SEED = 42
    FOLDS = 3
    TRAIN_FOLDS = [0]
    EPOCHS = 1
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    LR = 1e-3
    WEIGHT_DECAY = 1e-2

    MIXUP_ALPHA = 0.3
    MIXUP_PROB = 0.3

    TTA_SHIFTS = [0.0]
    USE_AMP = torch.cuda.is_available()

    DEBUG = True
    N_DEBUG = 200
    RUN_INFERENCE = True



def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True



def find_comp_dir() -> Path:
    candidates = [
        CFG.COMP_DIR,
        Path("/kaggle/input/birdclef-2026"),
        Path("./birdclef-2026"),
        Path("./data"),
        Path(".")
    ]
    for p in candidates:
        if p and p.exists() and (p / "train.csv").exists():
            return p
    raise FileNotFoundError(
        "Could not find the BirdCLEF dataset. Set BIRDCLEF_DIR or place the competition files in a visible path."
    )


seed_everything(CFG.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG.COMP_DIR = find_comp_dir()
CFG.TRAIN_AUDIO_DIR = CFG.COMP_DIR / "train_audio"
CFG.TRAIN_SOUNDSCAPE_DIR = CFG.COMP_DIR / "train_soundscapes"
CFG.TRAIN_CSV = CFG.COMP_DIR / "train.csv"
CFG.SOUNDSCAPE_CSV = CFG.COMP_DIR / "train_soundscapes_labels.csv"
CFG.TEST_DIR = CFG.COMP_DIR / "test_soundscapes"
CFG.SAMPLE_SUB = CFG.COMP_DIR / "sample_submission.csv"
CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Competition dir: {CFG.COMP_DIR}")


train_df = pd.read_csv(CFG.TRAIN_CSV)
sample_sub = pd.read_csv(CFG.SAMPLE_SUB)

SPECIES = [c for c in sample_sub.columns if c not in ("row_id", "filename", "end_time")]
NUM_CLASSES = len(SPECIES)
SPECIES2IDX = {s: i for i, s in enumerate(SPECIES)}
IDX2SPECIES = {i: s for s, i in SPECIES2IDX.items()}

if CFG.DEBUG:
    train_df = train_df.sample(n=min(CFG.N_DEBUG, len(train_df)), random_state=CFG.SEED).reset_index(drop=True)

train_df["filepath"] = train_df["filename"].apply(lambda x: str(CFG.TRAIN_AUDIO_DIR / x))
train_df = train_df[train_df["filepath"].map(lambda p: Path(p).exists())].reset_index(drop=True)

print(f"Classes: {NUM_CLASSES}")
print(f"Train rows: {len(train_df)}")
print(train_df.head(2))



def _clean_token(x) -> str:
    return str(x).strip().strip("'").strip('"')



def parse_label_list(value) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [_clean_token(x) for x in value if str(x).strip()]
    s = str(value).strip()
    if not s or s in {"[]", "nan", "None"}:
        return []
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [_clean_token(x) for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [_clean_token(x) for x in s.split(";") if x.strip()]



def encode_labels(row) -> np.ndarray:
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    primary = row.get("primary_label", "")
    if isinstance(primary, str) and primary in SPECIES2IDX:
        vec[SPECIES2IDX[primary]] = 1.0
    for sp in parse_label_list(row.get("secondary_labels", "")):
        if sp in SPECIES2IDX:
            vec[SPECIES2IDX[sp]] = 1.0
    return vec


train_df["label_vec"] = train_df.apply(encode_labels, axis=1)

if "author" in train_df.columns:
    groups = train_df["author"].fillna("unknown").astype(str)
elif "site" in train_df.columns:
    groups = train_df["site"].fillna("unknown").astype(str)
else:
    lat = train_df.get("latitude", pd.Series(np.zeros(len(train_df)))).fillna(0)
    lon = train_df.get("longitude", pd.Series(np.zeros(len(train_df)))).fillna(0)
    groups = (lat // 5).astype(str) + "_" + (lon // 5).astype(str)

train_df["fold"] = 0
n_groups = int(pd.Series(groups).nunique())
if len(train_df) >= 2 and n_groups >= 2:
    n_splits = min(CFG.FOLDS, n_groups)
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (_, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
            train_df.loc[val_idx, "fold"] = fold

label_matrix = np.stack(train_df["label_vec"].values)
class_counts = label_matrix.sum(axis=0).clip(min=1)
class_weights = torch.tensor((len(train_df) / (NUM_CLASSES * class_counts)).astype(np.float32), device=DEVICE)

print("Fold counts:")
print(train_df["fold"].value_counts().sort_index())
print(f"Class weight range: {class_weights.min().item():.2f} - {class_weights.max().item():.2f}")



def load_audio(path: str, sr: int = CFG.SR, duration: float | None = None) -> np.ndarray:
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    except Exception as exc:
        print(f"[WARN] load_audio failed for {path}: {exc}")
        audio = np.zeros(CFG.N_SAMPLES, dtype=np.float32)
    return audio.astype(np.float32)



def center_crop(audio: np.ndarray, n_samples: int = CFG.N_SAMPLES) -> np.ndarray:
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    start = max(0, (len(audio) - n_samples) // 2)
    return audio[start:start + n_samples]



def random_crop(audio: np.ndarray, n_samples: int = CFG.N_SAMPLES) -> np.ndarray:
    if len(audio) < n_samples:
        repeats = math.ceil(n_samples / max(len(audio), 1))
        audio = np.tile(audio, repeats)
    start = random.randint(0, len(audio) - n_samples)
    return audio[start:start + n_samples]



def audio_to_melspec(audio: np.ndarray, sr: int = CFG.SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=CFG.N_FFT,
        hop_length=CFG.HOP_LENGTH,
        n_mels=CFG.N_MELS,
        fmin=CFG.FMIN,
        fmax=CFG.FMAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    mel_norm = (mel_db + 80.0) / 80.0
    return mel_norm.astype(np.float32)



def time_shift(audio: np.ndarray, max_shift_frac: float = 0.1) -> np.ndarray:
    shift = int(random.uniform(0, max_shift_frac) * len(audio))
    return np.roll(audio, shift)



def add_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    return audio + np.random.randn(len(audio)).astype(np.float32) * noise_level



def random_gain(audio: np.ndarray, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    return audio * random.uniform(low, high)



def spec_augment(mel: np.ndarray, time_mask_param: int = 50, freq_mask_param: int = 16, num_masks: int = 2) -> np.ndarray:
    mel = mel.copy()
    T = mel.shape[1]
    F = mel.shape[0]
    for _ in range(num_masks):
        t = random.randint(0, min(time_mask_param, T))
        t0 = random.randint(0, max(T - t, 1))
        mel[:, t0:t0 + t] = 0.0
        f = random.randint(0, min(freq_mask_param, F))
        f0 = random.randint(0, max(F - f, 1))
        mel[f0:f0 + f, :] = 0.0
    return mel


class BirdClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio = load_audio(row["filepath"])

        if self.mode == "train":
            audio = random_crop(audio, CFG.N_SAMPLES)
            audio = time_shift(audio)
            if random.random() < 0.3:
                audio = add_noise(audio)
            if random.random() < 0.3:
                audio = random_gain(audio)
        else:
            audio = center_crop(audio, CFG.N_SAMPLES)

        mel = audio_to_melspec(audio)
        if self.mode == "train" and random.random() < 0.5:
            mel = spec_augment(mel)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(row["label_vec"], dtype=torch.float32)
        return mel, label



def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = CFG.MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam



def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class BirdModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.backbone(x)
        return self.head(feats)


class WeightedBCELoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
        return loss.mean()



def compute_macro_auc(targets: np.ndarray, probs: np.ndarray, species: list = SPECIES) -> float:
    aucs = []
    for i in range(len(species)):
        y_true = targets[:, i]
        if y_true.sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y_true, probs[:, i]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for mels, labels in tqdm(loader, desc="valid", leave=False):
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast(enabled=CFG.USE_AMP and DEVICE.type == "cuda"):
            logits = model(mels)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return total_loss / max(len(loader), 1), compute_macro_auc(all_labels, all_probs)



def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, scaler) -> float:
    model.train()
    total_loss = 0.0
    for mels, labels in tqdm(loader, desc="train", leave=False):
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if random.random() < CFG.MIXUP_PROB:
            mels, y_a, y_b, lam = mixup_data(mels, labels)
            with autocast(enabled=CFG.USE_AMP and DEVICE.type == "cuda"):
                logits = model(mels)
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
        else:
            with autocast(enabled=CFG.USE_AMP and DEVICE.type == "cuda"):
                logits = model(mels)
                loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)



def run_training(backbone_name: str) -> Path:
    best_auc = -1.0
    ckpt_path = CFG.OUTPUT_DIR / f"{backbone_name}_fold0.pt"

    for fold in CFG.TRAIN_FOLDS:
        tr_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        vl_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

        tr_ds = BirdClipDataset(tr_df, mode="train")
        vl_ds = BirdClipDataset(vl_df, mode="valid")

        tr_loader = DataLoader(
            tr_ds,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        vl_loader = DataLoader(
            vl_ds,
            batch_size=CFG.BATCH_SIZE,
            shuffle=False,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        model = BirdModel(backbone_name=backbone_name).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        criterion = WeightedBCELoss(class_weights=class_weights)
        scaler = GradScaler(enabled=CFG.USE_AMP and DEVICE.type == "cuda")

        print(f"=== {backbone_name} | fold={fold} ===")
        for epoch in range(CFG.EPOCHS):
            t0 = time.time()
            tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion, scaler)
            vl_loss, vl_auc = validate(model, vl_loader, criterion)
            elapsed = time.time() - t0
            if vl_auc > best_auc:
                best_auc = vl_auc
                torch.save(model.state_dict(), ckpt_path)
            print(f"epoch {epoch+1}/{CFG.EPOCHS} | tr_loss={tr_loss:.4f} | vl_loss={vl_loss:.4f} | auc={vl_auc:.4f} | {elapsed:.0f}s")

        del model, optimizer, scaler, tr_loader, vl_loader, tr_ds, vl_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved: {ckpt_path} | best_auc={best_auc:.4f}")
    return ckpt_path


@torch.no_grad()
def predict_soundscape(model: nn.Module, audio_path: Path, tta_shifts: list = CFG.TTA_SHIFTS) -> dict[int, np.ndarray]:
    clip_secs = 60
    total_samples = CFG.SR * clip_secs
    audio = load_audio(str(audio_path), sr=CFG.SR)
    if len(audio) < total_samples:
        audio = np.pad(audio, (0, total_samples - len(audio)))
    else:
        audio = audio[:total_samples]

    model.eval()
    results: dict[int, np.ndarray] = {}
    for end_time in range(5, 65, 5):
        start_sec = end_time - CFG.DURATION
        probs_list = []
        for shift in tta_shifts:
            start_sample = int((start_sec + shift) * CFG.SR)
            start_sample = max(0, min(start_sample, total_samples - CFG.N_SAMPLES))
            crop = audio[start_sample:start_sample + CFG.N_SAMPLES]
            if len(crop) < CFG.N_SAMPLES:
                crop = np.pad(crop, (0, CFG.N_SAMPLES - len(crop)))
            mel = audio_to_melspec(crop)
            mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            logits = model(mel_t)
            probs_list.append(torch.sigmoid(logits).cpu().numpy()[0])
        results[end_time] = np.mean(probs_list, axis=0)
    return results



def run_inference(ckpt_paths: list[Path]) -> pd.DataFrame:
    test_files = sorted(CFG.TEST_DIR.glob("*.ogg")) or sorted(CFG.TEST_DIR.glob("*.wav"))
    print(f"Test soundscapes: {len(test_files)}")
    if not test_files:
        raise FileNotFoundError(f"No test soundscape files found in {CFG.TEST_DIR}")

    loaded_models = []
    for path in ckpt_paths:
        if not path.exists():
            print(f"[WARN] Missing checkpoint: {path.name}")
            continue
        backbone_name = path.stem.replace("_fold0", "")
        model = BirdModel(backbone_name=backbone_name).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        loaded_models.append(model)
        print(f"Loaded {path.name}")

    if not loaded_models:
        raise RuntimeError("No checkpoints found for inference.")

    rows = []
    for audio_path in tqdm(test_files, desc="inference"):
        filename = audio_path.name
        model_preds = [predict_soundscape(m, audio_path) for m in loaded_models]
        for end_time in range(5, 65, 5):
            probs = np.mean([pred[end_time] for pred in model_preds], axis=0)
            row = {"row_id": f"{filename}_{end_time}", "filename": filename, "end_time": end_time}
            for i, sp in enumerate(SPECIES):
                row[sp] = float(probs[i])
            rows.append(row)

    sub = pd.DataFrame(rows)
    sub_path = CFG.OUTPUT_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission: {sub_path}")
    print(sub.head(3))
    return sub



def main() -> None:
    resnet_ckpt = run_training("resnet18")
    effnet_ckpt = run_training("efficientnet_b0")
    if CFG.RUN_INFERENCE:
        run_inference([resnet_ckpt, effnet_ckpt])


if __name__ == "__main__":
    main()
