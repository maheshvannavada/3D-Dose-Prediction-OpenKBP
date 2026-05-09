# phase4_train.py  — MASKED LOSS VERSION
#
# KEY FIX: We now use possible_dose_mask.csv to compute loss
# ONLY inside the patient body — not on air voxels.
# This forces the model to actually learn the dose distribution
# instead of hiding by predicting zero everywhere.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from phase2_dataloader import load_patient, load_sparse_csv_to_volume
from phase3_unet import UNet3D


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    'train_folder'  : 'provided-data/train-pats',
    'valid_folder'  : 'provided-data/validation-pats',
    'num_train_pts' : 10,
    'num_valid_pts' : 3,
    'crop_size'     : 64,
    'num_epochs'    : 8,
    'learning_rate' : 1e-3,
    'batch_size'    : 1,
    'base_filters'  : 16,
    'save_path'     : 'best_model.pth',
}


# ─────────────────────────────────────────────
# HELPER: center crop
# ─────────────────────────────────────────────

def center_crop(tensor, crop_size=64):
    """Crop center of 128³ volume to crop_size³."""
    start = (128 - crop_size) // 2   # = 32
    end   = start + crop_size         # = 96
    return tensor[:, start:end, start:end, start:end]


# ─────────────────────────────────────────────
# DATASET — now loads possible_dose_mask too
# ─────────────────────────────────────────────

class OpenKBPDataset(Dataset):

    def __init__(self, patient_folder, num_patients, crop_size=64):
        all_paths = sorted([
            os.path.join(patient_folder, pt)
            for pt in os.listdir(patient_folder)
            if os.path.isdir(os.path.join(patient_folder, pt))
        ])
        self.patient_paths = all_paths[:num_patients]
        self.crop_size     = crop_size
        print(f"  Dataset: {len(self.patient_paths)} patients loaded")

    def __len__(self):
        return len(self.patient_paths)

    def __getitem__(self, index):
        patient_path = self.patient_paths[index]

        # Load CT, organ masks, dose from Phase 2
        ct_tensor, mask_tensor, dose_tensor = load_patient(patient_path)

        # NEW: Load possible_dose_mask
        # This binary volume marks which voxels are inside the patient body.
        # 1 = inside body (dose matters here)
        # 0 = air/outside (always zero, ignore during loss computation)
        dose_mask_raw    = load_sparse_csv_to_volume(
            os.path.join(patient_path, 'possible_dose_mask.csv')
        )
        # Convert to binary tensor with channel dim: (1, 128, 128, 128)
        dose_mask_tensor = torch.from_numpy(
            (dose_mask_raw > 0).astype(np.float32)
        ).unsqueeze(0)

        # Stack CT + organ masks → model input: (11, 128, 128, 128)
        model_input = torch.cat([ct_tensor, mask_tensor], dim=0)

        # Center crop ALL tensors to 64³
        model_input      = center_crop(model_input,      self.crop_size)
        dose_tensor      = center_crop(dose_tensor,      self.crop_size)
        mask_tensor      = center_crop(mask_tensor,      self.crop_size)
        dose_mask_tensor = center_crop(dose_mask_tensor, self.crop_size)

        return model_input, dose_tensor, mask_tensor, dose_mask_tensor


# ─────────────────────────────────────────────
# LOSS — Masked MAE
# ─────────────────────────────────────────────

def masked_mae_loss(predicted_dose, true_dose, possible_dose_mask):
    """
    Computes MAE only on voxels INSIDE the patient body.

    Why this matters:
      A 64³ volume has ~262,000 voxels.
      ~240,000 are air → dose = 0 always.
      Predicting zero everywhere gives MAE ≈ 7 Gy (looks like learning!)
      but the model hasn't learned anything useful.

    By masking, we force the model to focus on the
    ~20,000 body voxels where dose actually varies.

    Args:
        predicted_dose     : (B, 1, D, H, W)
        true_dose          : (B, 1, D, H, W)
        possible_dose_mask : (B, 1, D, H, W) — 1=body, 0=air

    Returns:
        scalar MAE loss (Gy) computed only inside the body
    """
    # Number of body voxels (add epsilon to avoid division by zero)
    num_body_voxels = possible_dose_mask.sum() + 1e-8

    # Compute error only where mask = 1
    masked_error = torch.abs(predicted_dose - true_dose) * possible_dose_mask

    return masked_error.sum() / num_body_voxels


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, device, epoch_num):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch_num}/{CONFIG['num_epochs']} [Train]",
        leave=True
    )

    # Unpack 4 items now (added dose_mask)
    for model_input, true_dose, organ_masks, dose_mask in progress_bar:

        model_input = model_input.to(device)
        true_dose   = true_dose.to(device)
        dose_mask   = dose_mask.to(device)

        # Forward pass
        predicted_dose = model(model_input)

        # Masked MAE loss — only inside the body
        loss = masked_mae_loss(predicted_dose, true_dose, dose_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'masked_MAE_Gy': f'{loss.item():.3f}'})

    return total_loss / len(dataloader)


# ─────────────────────────────────────────────
# VALIDATION LOOP
# ─────────────────────────────────────────────

def validate_one_epoch(model, dataloader, device, epoch_num):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch_num}/{CONFIG['num_epochs']} [Valid]",
            leave=True
        )
        for model_input, true_dose, organ_masks, dose_mask in progress_bar:

            model_input = model_input.to(device)
            true_dose   = true_dose.to(device)
            dose_mask   = dose_mask.to(device)

            predicted_dose = model(model_input)
            loss = masked_mae_loss(predicted_dose, true_dose, dose_mask)

            total_loss += loss.item()
            progress_bar.set_postfix({'val_masked_MAE': f'{loss.item():.3f}'})

    return total_loss / len(dataloader)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  Phase 4 — Masked MAE Training (10 patients, 8 epochs)")
    print("="*55 + "\n")

    device = torch.device("cpu")
    print(f"  Device : CPU")
    print(f"  Volume : {CONFIG['crop_size']}³ voxels")
    print(f"  Loss   : Masked MAE (body voxels only)\n")

    # Datasets
    print("Loading datasets...")
    train_dataset = OpenKBPDataset(
        CONFIG['train_folder'],
        num_patients=CONFIG['num_train_pts'],
        crop_size=CONFIG['crop_size']
    )
    valid_dataset = OpenKBPDataset(
        CONFIG['valid_folder'],
        num_patients=CONFIG['num_valid_pts'],
        crop_size=CONFIG['crop_size']
    )

    train_loader = DataLoader(train_dataset, batch_size=1,
                              shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1,
                              shuffle=False, num_workers=0)

    # Model
    print("\nBuilding model...")
    model = UNet3D(in_channels=11, base_filters=CONFIG['base_filters']).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {params:,} ({params/1e6:.2f}M)\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    history = {'train_mae': [], 'valid_mae': []}
    best_valid_loss = float('inf')

    print(f"Starting {CONFIG['num_epochs']} epochs...\n")
    print("  NOTE: Loss values will be HIGHER now (20-30 Gy at start)")
    print("  This is correct — we are now measuring only body voxels")
    print("  where dose actually varies. Watch for the TREND downward.\n")

    for epoch in range(1, CONFIG['num_epochs'] + 1):

        train_mae = train_one_epoch(model, train_loader, optimizer, device, epoch)
        valid_mae = validate_one_epoch(model, valid_loader, device, epoch)

        # Save best model
        if valid_mae < best_valid_loss:
            best_valid_loss = valid_mae
            torch.save({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_mae'           : valid_mae,
                'config'              : CONFIG,
            }, CONFIG['save_path'])
            print(f"  💾 Best model saved (epoch {epoch}, "
                  f"val_MAE={valid_mae:.3f} Gy)")

        print(f"\n  ── Epoch {epoch} Summary ──")
        print(f"     Train masked MAE : {train_mae:.3f} Gy")
        print(f"     Valid masked MAE : {valid_mae:.3f} Gy")
        print(f"     Best so far      : {best_valid_loss:.3f} Gy\n")

        history['train_mae'].append(train_mae)
        history['valid_mae'].append(valid_mae)

    # Plot
    epochs_range = range(1, CONFIG['num_epochs'] + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, history['train_mae'], 'b-o', label='Train Masked MAE (Gy)')
    plt.plot(epochs_range, history['valid_mae'], 'r-o', label='Valid Masked MAE (Gy)')
    plt.title('Phase 4 — Masked MAE Training Curve\n(body voxels only)')
    plt.xlabel('Epoch')
    plt.ylabel('Masked MAE (Gy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()

    print("="*55)
    print("   Phase 4 COMPLETE!")
    print(f"     Best validation MAE : {best_valid_loss:.3f} Gy")
    print(f"     Model saved to      : {CONFIG['save_path']}")
    print("="*55)
    print("\n  Expected masked MAE ranges:")
    print("     Epoch 1  : 20–35 Gy  (model just starting)")
    print("     Epoch 8  : 12–20 Gy  (model improving)")
    print("     After this, run phase5_evaluate.py to see the dose blob!")