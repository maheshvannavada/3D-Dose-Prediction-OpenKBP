# phase2_dataloader.py
# Phase 2: Load and preprocess one OpenKBP patient into clean PyTorch tensors.
# FIXED VERSION — all three bugs corrected:
#   Fix 1: Organ CSVs are in the patient folder directly (no 'structures' subfolder)
#   Fix 2: CT HU range adjusted to [0, 4000] because OpenKBP pre-shifts HU by +1000
#   Fix 3: PTV70 added to organ list (it IS present in the data)

import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Every patient volume in OpenKBP is exactly this size
VOLUME_SHAPE = (128, 128, 128)  # (Depth, Height, Width) in voxels

# FIX 2: OpenKBP pre-adds 1000 to all HU values before saving.
# So what should be air (-1000 HU) is stored as 0,
# and bone (~700 HU) is stored as ~1700.
# We clip at 4000 because the max we saw was 3976.
CT_MIN_HU = 0
CT_MAX_HU = 4000

# FIX 3: PTV70 added — it is the PRIMARY tumor target (highest dose region)
# and it IS present in the data. Esophagus and Larynx may be missing
# for some patients — our loader handles that gracefully with zeros.
OAR_NAMES = [
    'Brainstem',
    'SpinalCord',
    'RightParotid',
    'LeftParotid',
    'Esophagus',     # may be absent for some patients — returns zeros if missing
    'Larynx',        # may be absent for some patients — returns zeros if missing
    'Mandible',
    'PTV56',         # Planning Target Volume — low dose ring
    'PTV63',         # Planning Target Volume — medium dose ring
    'PTV70',         # Planning Target Volume — HIGH dose core (tumor itself)
]


# ─────────────────────────────────────────────────────────────
# FUNCTION 1: Read one sparse CSV file → full 3D numpy volume
# ─────────────────────────────────────────────────────────────

def load_sparse_csv_to_volume(csv_path, volume_shape=VOLUME_SHAPE):
    """
    OpenKBP uses TWO different sparse CSV formats:
    
    Format A — CT and Dose (two columns):
        ,data
        1073061, 127.0      ← index, value
    
    Format B — Organ masks (one real column):
        ,data
        1073061,            ← index only, value is always 1.0
    
    We handle both formats automatically.
    """

    volume = np.zeros(volume_shape, dtype=np.float32)

    if not os.path.exists(csv_path):
        print(f"  ⚠  Missing (normal for some patients): {os.path.basename(csv_path)}")
        return volume

    data = pd.read_csv(csv_path, header=0)

    if data.empty:
        return volume

    # Column 0 is always the flat voxel index
    flat_indices = data.iloc[:, 0].values.astype(int)

    # Check if column 1 has real values or is all empty (NaN)
    if data.shape[1] > 1 and not data.iloc[:, 1].isnull().all():
        # Format A: CT or Dose — use the actual values
        values = data.iloc[:, 1].values.astype(np.float32)
    else:
        # Format B: Organ mask — every listed voxel = 1.0
        values = np.ones(len(flat_indices), dtype=np.float32)

    z_coords, y_coords, x_coords = np.unravel_index(flat_indices, volume_shape)
    volume[z_coords, y_coords, x_coords] = values

    return volume


# ─────────────────────────────────────────────────────────────
# FUNCTION 2: Normalize CT volume from raw HU range to [0.0, 1.0]
# ─────────────────────────────────────────────────────────────

def normalize_ct(ct_volume, min_hu=CT_MIN_HU, max_hu=CT_MAX_HU):
    """
    Neural networks train poorly on large raw numbers like 0–4000.
    We rescale everything to [0.0, 1.0].

    Step 1 — Clip: values outside [min_hu, max_hu] are noise, clamp them.
    Step 2 — Scale: (value - min) / (max - min)

    Example: stored HU = 1040 (which is real-world soft tissue ~40 HU)
             normalized = (1040 - 0) / (4000 - 0) = 0.26
    """

    ct_clipped    = np.clip(ct_volume, min_hu, max_hu)
    ct_normalized = (ct_clipped - min_hu) / (max_hu - min_hu)

    return ct_normalized.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# FUNCTION 3: Load all organ masks → stacked tensor (10, D, H, W)
# ─────────────────────────────────────────────────────────────

def load_organ_masks(patient_folder, oar_names=OAR_NAMES):
    """
    Loads each organ's binary mask and stacks them into one array.

    FIX 1 APPLIED: organ CSVs live directly in the patient folder,
    NOT in a 'structures' subfolder. We pass patient_folder directly.

    Output shape: (10, 128, 128, 128)
      channel 0 = Brainstem mask   (1.0 where brainstem is, 0.0 elsewhere)
      channel 1 = SpinalCord mask
      ...
      channel 9 = PTV70 mask       (the tumor target)

    This is like a 10-page book where each page is one organ's silhouette.
    """

    all_masks = []

    for organ_name in oar_names:
        # FIX 1: look for the CSV directly in patient_folder
        mask_path  = os.path.join(patient_folder, f"{organ_name}.csv")
        organ_mask = load_sparse_csv_to_volume(mask_path)

        # Convert to binary: 1.0 = organ present, 0.0 = not present
        organ_mask = (organ_mask > 0).astype(np.float32)

        all_masks.append(organ_mask)

    # Stack list of (128,128,128) arrays → one (10, 128,128,128) array
    stacked_masks = np.stack(all_masks, axis=0)

    return stacked_masks


# ─────────────────────────────────────────────────────────────
# FUNCTION 4: Master loader — one patient → three clean tensors
# ─────────────────────────────────────────────────────────────

def load_patient(patient_folder):
    """
    Loads CT, dose, and organ masks for one patient.
    Returns PyTorch tensors ready to be fed into a neural network.

    Args:
        patient_folder : e.g. 'provided-data/train-pats/pt_1'

    Returns:
        ct_tensor   : shape (1,  128, 128, 128)  — CT scan
        mask_tensor : shape (10, 128, 128, 128)  — organ masks
        dose_tensor : shape (1,  128, 128, 128)  — ground truth dose
    """

    print(f"\nLoading patient: {os.path.basename(patient_folder)}")

    # ── CT scan ──
    ct_raw        = load_sparse_csv_to_volume(os.path.join(patient_folder, 'ct.csv'))
    ct_normalized = normalize_ct(ct_raw)
    print(f"  CT loaded    → shape {ct_raw.shape}, "
          f"raw range [{ct_raw.min():.0f}, {ct_raw.max():.0f}] "
          f"→ normalized [{ct_normalized.min():.2f}, {ct_normalized.max():.2f}]")

    # ── Dose map (ground truth label) ──
    dose_volume = load_sparse_csv_to_volume(os.path.join(patient_folder, 'dose.csv'))
    print(f"  Dose loaded  → shape {dose_volume.shape}, "
          f"range [{dose_volume.min():.1f}, {dose_volume.max():.1f}] Gy")

    # ── Organ masks — FIX 1: pass patient_folder directly ──
    organ_masks = load_organ_masks(patient_folder)
    print(f"  Masks loaded → shape {organ_masks.shape}  "
          f"({len(OAR_NAMES)} organs × 128³ voxels)")

    # ── Convert numpy arrays → PyTorch tensors ──
    # unsqueeze(0) adds the channel dimension: (D,H,W) → (1,D,H,W)
    # PyTorch Conv3d always expects (batch, channels, D, H, W)
    ct_tensor   = torch.from_numpy(ct_normalized).unsqueeze(0)  # (1, 128, 128, 128)
    dose_tensor = torch.from_numpy(dose_volume).unsqueeze(0)    # (1, 128, 128, 128)
    mask_tensor = torch.from_numpy(organ_masks)                 # (10, 128, 128, 128)

    print(f"\n  ✅ Final tensor shapes:")
    print(f"     ct_tensor   : {ct_tensor.shape}")
    print(f"     mask_tensor : {mask_tensor.shape}")
    print(f"     dose_tensor : {dose_tensor.shape}")

    return ct_tensor, mask_tensor, dose_tensor


# ─────────────────────────────────────────────────────────────
# CHECKPOINT 2 — ONE single main block (this was the bug before)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    patient_path = "provided-data/train-pats/pt_1"

    # ── Step A: Show what files are actually in the folder ──
    print("=== Folder contents of pt_1 ===")
    for f in sorted(os.listdir(patient_path)):
        print(f"  {f}")

    # ── Step B: Load the patient ──
    ct_tensor, mask_tensor, dose_tensor = load_patient(patient_path)

    # ── Step C: Print non-zero voxel counts per organ ──
    print("\n=== Organ mask statistics ===")
    for i, name in enumerate(OAR_NAMES):
        nonzero = mask_tensor[i].sum().item()
        status  = "✅" if nonzero > 0 else "⚠  absent"
        print(f"  {name:15s}: {nonzero:>8.0f} non-zero voxels  {status}")

    # ── Step D: Visualize the middle axial slice ──
    slice_idx = 64

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Phase 2 Checkpoint — Real OpenKBP Patient (pt_1)",
                 fontsize=13, fontweight='bold')

    # Plot 1 — CT scan
    im0 = axes[0].imshow(ct_tensor[0, slice_idx].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('CT Scan (normalized 0→1)')
    plt.colorbar(im0, ax=axes[0], label='0=air, 1=bone')

    # Plot 2 — All organ masks summed (so we see all organs at once)
    combined_mask = mask_tensor[:, slice_idx].sum(axis=0).numpy()
    im1 = axes[1].imshow(combined_mask, cmap='hot')
    axes[1].set_title('All Organ Masks (summed)')
    plt.colorbar(im1, ax=axes[1], label='organ overlap count')

    # Plot 3 — Ground truth dose
    im2 = axes[2].imshow(dose_tensor[0, slice_idx].numpy(), cmap='jet')
    axes[2].set_title('Ground Truth Dose (Gy)')
    plt.colorbar(im2, ax=axes[2], label='Dose (Gy)')

    plt.tight_layout()
    plt.savefig('checkpoint_2_output.png', dpi=150)
    plt.show()

    print(" Phase 2 COMPLETE — All three panels should now show real data!")
    print("   If organ masks panel is no longer blank orange → you passed! 🎉")