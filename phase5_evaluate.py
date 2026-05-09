# phase5_evaluate.py
# Phase 5: Clinical Evaluation & Heatmap Visualization
#
# This script loads the trained model and produces:
#   1. Side-by-side predicted vs true dose heatmaps
#   2. Dose error map (where did we go wrong?)
#   3. DVH curves per organ (the clinical gold standard plot)
#   4. Summary MAE score per patient

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from phase2_dataloader import load_patient, OAR_NAMES
from phase3_unet import UNet3D


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    'model_path'   : 'best_model.pth',
    'test_folder'  : 'provided-data/validation-pats',
    'num_patients' : 3,        # evaluate on 3 validation patients
    'crop_size'    : 64,       # must match training crop size
    'base_filters' : 16,       # must match training base_filters
}


# ─────────────────────────────────────────────
# HELPER: same center crop from Phase 4
# ─────────────────────────────────────────────

def center_crop(tensor, crop_size=64):
    """Crop center of 128³ volume to crop_size³."""
    start = (128 - crop_size) // 2
    end   = start + crop_size
    return tensor[:, start:end, start:end, start:end]


# ─────────────────────────────────────────────
# STEP 1: Load trained model from checkpoint
# ─────────────────────────────────────────────

def load_trained_model(model_path, base_filters=16):
    """
    Loads the saved model weights from Phase 4.

    During training we saved:
        torch.save({ 'model_state_dict': ... }, path)

    Now we rebuild the same architecture and pour the
    saved weights back in — like reloading a saved game.
    """
    device = torch.device('cpu')

    # Rebuild the exact same architecture as training
    model = UNet3D(in_channels=11, base_filters=base_filters).to(device)

    # Load the saved checkpoint file
    checkpoint = torch.load(model_path, map_location=device)

    # Pour the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode — disables BatchNorm's training behaviour
    model.eval()

    saved_epoch = checkpoint.get('epoch', '?')
    saved_mae   = checkpoint.get('valid_mae', '?')
    print(f"  Model loaded from epoch {saved_epoch} "
          f"(saved val_MAE = {saved_mae:.3f} Gy)")

    return model, device


# ─────────────────────────────────────────────
# STEP 2: Run inference on one patient
# ─────────────────────────────────────────────

def predict_dose(model, patient_path, device, crop_size=64):
    """
    Loads one patient, runs the model, returns:
        predicted_dose : (64, 64, 64) numpy array in Gy
        true_dose      : (64, 64, 64) numpy array in Gy
        ct_volume      : (64, 64, 64) numpy array normalized [0,1]
        mask_tensor    : (10, 64, 64, 64) organ masks
    """

    # Load full patient data
    ct_tensor, mask_tensor, dose_tensor = load_patient(patient_path)

    # Build model input: CT + masks → (11, 128, 128, 128)
    model_input = torch.cat([ct_tensor, mask_tensor], dim=0)

    # Center crop to 64³
    model_input  = center_crop(model_input,  crop_size)
    dose_tensor  = center_crop(dose_tensor,  crop_size)
    mask_tensor  = center_crop(mask_tensor,  crop_size)
    ct_cropped   = center_crop(ct_tensor,    crop_size)

    # Add batch dimension: (11, 64, 64, 64) → (1, 11, 64, 64, 64)
    model_input_batch = model_input.unsqueeze(0).to(device)

    # Run inference — no gradients needed
    with torch.no_grad():
        predicted_batch = model(model_input_batch)

    # Remove batch dimension and convert to numpy
    predicted_dose = predicted_batch[0, 0].cpu().numpy()   # (64, 64, 64)
    true_dose      = dose_tensor[0].numpy()                 # (64, 64, 64)
    ct_volume      = ct_cropped[0].numpy()                  # (64, 64, 64)
    masks_numpy    = mask_tensor.numpy()                    # (10, 64, 64, 64)

    # Compute MAE for this patient
    mae = np.mean(np.abs(predicted_dose - true_dose))

    return predicted_dose, true_dose, ct_volume, masks_numpy, mae


# ─────────────────────────────────────────────
# STEP 3: Plot heatmaps for one patient
# ─────────────────────────────────────────────

def plot_dose_heatmaps(predicted_dose, true_dose, ct_volume,
                       patient_name, save_path):
    """
    Creates a 4-panel figure:
      Panel 1: CT scan (anatomical reference)
      Panel 2: Ground truth dose (what a real plan looks like)
      Panel 3: Predicted dose (what our model produced)
      Panel 4: Error map (difference — where we went wrong)
    """

    # Pick the slice with the highest true dose — most informative slice
    slice_idx = int(np.argmax(true_dose.max(axis=(1, 2))))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f'Phase 5 — Dose Prediction: {patient_name}',
        fontsize=13, fontweight='bold'
    )

    # Shared dose color scale across panels 2 and 3
    dose_max = max(true_dose.max(), predicted_dose.max())
    dose_norm = Normalize(vmin=0, vmax=dose_max)

    # ── Panel 1: CT scan ──
    im0 = axes[0].imshow(ct_volume[slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('CT Scan')
    axes[0].set_xlabel(f'Axial slice {slice_idx}')
    plt.colorbar(im0, ax=axes[0], label='Normalized HU')

    # ── Panel 2: Ground truth dose ──
    im1 = axes[1].imshow(
        true_dose[slice_idx], cmap='jet', norm=dose_norm
    )
    axes[1].set_title('Ground Truth Dose')
    axes[1].set_xlabel('Gy')
    plt.colorbar(im1, ax=axes[1], label='Dose (Gy)')

    # ── Panel 3: Predicted dose ──
    im2 = axes[2].imshow(
        predicted_dose[slice_idx], cmap='jet', norm=dose_norm
    )
    axes[2].set_title('Predicted Dose (Our Model)')
    axes[2].set_xlabel('Gy')
    plt.colorbar(im2, ax=axes[2], label='Dose (Gy)')

    # ── Panel 4: Error map (absolute difference) ──
    error_map = np.abs(predicted_dose[slice_idx] - true_dose[slice_idx])
    im3 = axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=dose_max * 0.3)
    axes[3].set_title('Absolute Error Map')
    axes[3].set_xlabel('Bright = large error')
    plt.colorbar(im3, ax=axes[3], label='|Error| (Gy)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f" Heatmap saved: {save_path}")


# ─────────────────────────────────────────────
# STEP 4: Plot DVH curves
# ─────────────────────────────────────────────

def plot_dvh_curves(predicted_dose, true_dose, organ_masks,
                    patient_name, save_path):
    """
    Dose-Volume Histogram (DVH) — the gold standard clinical plot.

    For each organ, we ask:
    'What fraction of this organ's volume receives at least X Gy?'

    We plot this for both predicted and true dose.
    A good model's curves overlap closely with the ground truth.

    Solid line  = ground truth
    Dashed line = our prediction
    """

    dose_thresholds = np.linspace(0, 80, 100)  # 0 to 80 Gy in 100 steps

    # Colors for up to 10 organs
    colors = [
        '#E24B4A', '#185FA5', '#2E9E4F', '#BA7517', '#7B3FA0',
        '#C45CA0', '#4AABBA', '#8B6914', '#5D7A8C', '#A0522D'
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        f'DVH Curves — {patient_name}\n'
        f'(solid=ground truth, dashed=predicted)',
        fontsize=12, fontweight='bold'
    )

    plotted_any = False

    for organ_idx, organ_name in enumerate(OAR_NAMES):

        organ_mask = organ_masks[organ_idx]          # (64, 64, 64)
        organ_voxels = organ_mask.sum()

        # Skip organs absent in this patient
        if organ_voxels < 10:
            continue

        color = colors[organ_idx % len(colors)]

        # Compute DVH for true dose
        true_dvh = []
        pred_dvh = []

        for threshold in dose_thresholds:
            # Fraction of organ volume receiving >= threshold Gy
            true_frac = ((true_dose >= threshold) & (organ_mask > 0)).sum() \
                        / organ_voxels
            pred_frac = ((predicted_dose >= threshold) & (organ_mask > 0)).sum() \
                        / organ_voxels

            true_dvh.append(true_frac * 100)   # convert to percentage
            pred_dvh.append(pred_frac * 100)

        # Plot ground truth (solid) and prediction (dashed)
        ax.plot(dose_thresholds, true_dvh,
                color=color, linewidth=2,
                label=f'{organ_name}')
        ax.plot(dose_thresholds, pred_dvh,
                color=color, linewidth=1.5, linestyle='--')

        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, 'No organ masks found in this crop region.\n'
                'This is normal — organs may be outside the 64³ center crop.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='gray')
    else:
        ax.legend(loc='upper right', fontsize=8, ncol=2)

    ax.set_xlabel('Dose (Gy)', fontsize=11)
    ax.set_ylabel('Volume (%)', fontsize=11)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f" DVH plot saved: {save_path}")


# ─────────────────────────────────────────────
# STEP 5: Summary table across all patients
# ─────────────────────────────────────────────

def print_summary_table(results):
    """
    Prints a clean summary table of MAE per patient.
    This is what you'd report in a research paper or
    in your Fraunhofer project documentation.
    """
    print("\n" + "="*45)
    print("  Clinical Evaluation Summary")
    print("="*45)
    print(f"  {'Patient':<15} {'MAE (Gy)':>10}")
    print("-"*45)

    all_maes = []
    for patient_name, mae in results:
        print(f"  {patient_name:<15} {mae:>10.3f} Gy")
        all_maes.append(mae)

    print("-"*45)
    print(f"  {'Mean MAE':<15} {np.mean(all_maes):>10.3f} Gy")
    print(f"  {'Std MAE':<15} {np.std(all_maes):>10.3f} Gy")
    print(f"  {'Best MAE':<15} {np.min(all_maes):>10.3f} Gy")
    print("="*45)
    print("\n  Benchmark reference (from OpenKBP paper):")
    print("    Top teams achieved ~2.5 Gy MAE (200 pts, GPU, 200 epochs)")
    print("    Our result with 10 pts, CPU, 3 epochs is expected at 7-10 Gy")
    print("    → Scaling up = direct path to competitive performance\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  Phase 5 — Clinical Evaluation & Visualization")
    print("="*55 + "\n")

    # ── Load model ──
    print("Loading trained model...")
    model, device = load_trained_model(
        CONFIG['model_path'],
        base_filters=CONFIG['base_filters']
    )

    # ── Get validation patient paths ──
    all_val_paths = sorted([
        os.path.join(CONFIG['test_folder'], pt)
        for pt in os.listdir(CONFIG['test_folder'])
        if os.path.isdir(os.path.join(CONFIG['test_folder'], pt))
    ])
    eval_paths = all_val_paths[:CONFIG['num_patients']]

    results = []

    # ── Evaluate each patient ──
    for patient_path in eval_paths:

        patient_name = os.path.basename(patient_path)
        print(f"\n{'─'*45}")
        print(f"  Evaluating: {patient_name}")
        print(f"{'─'*45}")

        # Run model inference
        predicted_dose, true_dose, ct_volume, organ_masks, mae = predict_dose(
            model, patient_path, device, CONFIG['crop_size']
        )
        print(f"  MAE for this patient: {mae:.3f} Gy")

        # Plot heatmaps
        plot_dose_heatmaps(
            predicted_dose, true_dose, ct_volume,
            patient_name,
            save_path=f'heatmap_{patient_name}.png'
        )

        # Plot DVH curves
        plot_dvh_curves(
            predicted_dose, true_dose, organ_masks,
            patient_name,
            save_path=f'dvh_{patient_name}.png'
        )

        results.append((patient_name, mae))

    # ── Print summary table ──
    print_summary_table(results)

    print(" Phase 5 COMPLETE — All outputs saved!")
    print("\n  Files produced:")
    for patient_name, _ in results:
        print(f"    heatmap_{patient_name}.png")
        print(f"    dvh_{patient_name}.png")
    print("\n  You now have a complete end-to-end medical AI pipeline!")


    '''
    Reading Your Output
Panel 1 — CT Scan ✅
The patient's head anatomy is clearly visible. Good.
Panel 2 — Ground Truth Dose ✅
Beautiful IMRT dose distribution. Red core = 70 Gy at tumor, smooth blue falloff. This is what a real treatment plan looks like.
Panel 3 — Predicted Dose ⚠️
Almost completely blue (near zero). The model "knows" most voxels should be zero, but hasn't learned to place the high-dose blob correctly yet.
Panel 4 — Error Map
The bright white region shows exactly where the model failed — it missed the tumor dose entirely.
    '''