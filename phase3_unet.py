# phase3_unet.py
# Phase 3: The 3D U-Net — the brain of our dose prediction pipeline.
#
# INPUT : (batch, 11, 128, 128, 128)
#          → 1 CT channel + 10 organ mask channels stacked together
# OUTPUT: (batch,  1, 128, 128, 128)
#          → predicted dose map in Gy

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# BUILDING BLOCK: ConvBlock
# Two 3D convolutions with BatchNorm and ReLU activation.
# This is the basic "think about the data" unit used everywhere.
# ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Two rounds of: Conv3D → BatchNorm → ReLU

    Why two convolutions? One conv looks at immediate neighbors.
    Two convs look at neighbors-of-neighbors — a wider receptive field
    without using a bigger (slower) kernel.

    Why BatchNorm? It keeps the numbers from exploding or vanishing
    during training. Think of it as re-centering the data after each layer.

    Why ReLU? It adds non-linearity — without it, stacking layers
    is mathematically identical to one layer. ReLU lets the network
    learn complex curved relationships, not just straight lines.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            # First convolution: learn local 3D patterns
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,    # 3×3×3 cube of neighbors
                padding=1,        # 'same' padding — output same size as input
                bias=False        # BatchNorm handles bias implicitly
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            # Second convolution: refine what was just learned
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_tensor):
        return self.block(input_tensor)


# ─────────────────────────────────────────────────────────────
# ENCODER BLOCK: ConvBlock + MaxPool3D
# Shrinks spatial size by 2× while doubling channel depth.
# "Zoom out to understand the big picture."
# ─────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    Applies ConvBlock, then saves the result as a 'skip feature'
    for later, then downsamples with MaxPool.

    MaxPool3d(2): takes every 2×2×2 cube and keeps only the max value.
    This halves each spatial dimension: 128→64→32→16
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv  = ConvBlock(in_channels, out_channels)
        self.pool  = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input_tensor):
        # Run convolutions first — this is what we'll skip-connect
        skip_features = self.conv(input_tensor)
        # Then downsample for the next encoder level
        downsampled   = self.pool(skip_features)
        return downsampled, skip_features


# ─────────────────────────────────────────────────────────────
# DECODER BLOCK: Upsample + Concatenate skip + ConvBlock
# Grows spatial size by 2× and uses skip connection.
# "Zoom back in using what we remembered earlier."
# ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Step 1 — Upsample: doubles each spatial dimension (16→32→64→128)
             using learned transposed convolution (not simple interpolation).

    Step 2 — Concatenate: paste the skip features from the matching encoder
             level alongside the upsampled features. This restores fine
             spatial detail that was lost during downsampling.

    Step 3 — ConvBlock: fuse the combined features into a refined output.

    The skip connection is the "U" in U-Net — it's the key innovation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # ConvTranspose3d: learnable upsampling (opposite of MaxPool)
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        # After concatenation, channel count doubles → we halve it back
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, decoder_tensor, skip_tensor):
        # Step 1: upsample the decoder tensor
        upsampled = self.upsample(decoder_tensor)

        # Step 2: concatenate along the channel dimension (dim=1)
        # upsampled shape: (B, C, D, H, W)
        # skip_tensor shape: (B, C, D, H, W)  ← same spatial size now
        # combined shape:   (B, 2C, D, H, W)
        combined = torch.cat([upsampled, skip_tensor], dim=1)

        # Step 3: fuse with two convolutions
        output = self.conv(combined)
        return output


# ─────────────────────────────────────────────────────────────
# THE FULL 3D U-NET
# ─────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    Full 3D U-Net for radiation dose prediction.

    Architecture:
      Input  (B, 11, 128, 128, 128)   ← 1 CT + 10 organ channels
        ↓ Encoder 1                    → 32 feature maps, 128³
        ↓ Encoder 2                    → 64 feature maps, 64³
        ↓ Encoder 3                    → 128 feature maps, 32³
        ↓ Bottleneck                   → 256 feature maps, 16³
        ↑ Decoder 3 + skip from Enc 3  → 128 feature maps, 32³
        ↑ Decoder 2 + skip from Enc 2  → 64 feature maps, 64³
        ↑ Decoder 1 + skip from Enc 1  → 32 feature maps, 128³
        → Final 1×1×1 Conv             → 1 feature map, 128³ (dose)
        → ReLU                         → only positive dose values
      Output (B, 1, 128, 128, 128)    ← predicted dose in Gy
    """

    def __init__(self, in_channels=11, base_filters=32):
        """
        Args:
            in_channels  : 11 = 1 CT + 10 organ mask channels
            base_filters : 32 = number of feature maps in first encoder layer.
                           Doubles at each level: 32 → 64 → 128 → 256
        """
        super().__init__()

        f = base_filters  # shorthand: 32

        # ── Encoder (left side of the U) ──
        self.encoder1 = EncoderBlock(in_channels, f)       # 11 → 32
        self.encoder2 = EncoderBlock(f,           f * 2)   # 32 → 64
        self.encoder3 = EncoderBlock(f * 2,       f * 4)   # 64 → 128

        # ── Bottleneck (bottom of the U) ──
        self.bottleneck = ConvBlock(f * 4, f * 8)          # 128 → 256

        # ── Decoder (right side of the U) ──
        self.decoder3 = DecoderBlock(f * 8, f * 4)         # 256 → 128
        self.decoder2 = DecoderBlock(f * 4, f * 2)         # 128 → 64
        self.decoder1 = DecoderBlock(f * 2, f)             # 64  → 32

        # ── Final layer: squeeze 32 channels → 1 dose channel ──
        # kernel_size=1 means "mix channels without looking at neighbors"
        self.final_conv = nn.Conv3d(f, 1, kernel_size=1)

        # ── ReLU to ensure no negative dose predictions ──
        # Dose is always ≥ 0 Gy physically
        self.final_relu = nn.ReLU()

    def forward(self, input_tensor):
        """
        The data flows down the encoder, across the bottleneck,
        and back up the decoder with skip connections bridging each level.
        """

        # ── Encoding path ──
        # Each encoder returns: (downsampled_tensor, skip_features)
        enc1_down, enc1_skip = self.encoder1(input_tensor)
        enc2_down, enc2_skip = self.encoder2(enc1_down)
        enc3_down, enc3_skip = self.encoder3(enc2_down)

        # ── Bottleneck ──
        bottleneck_features = self.bottleneck(enc3_down)

        # ── Decoding path ──
        # Each decoder takes: (current_tensor, matching_skip_tensor)
        dec3_out = self.decoder3(bottleneck_features, enc3_skip)
        dec2_out = self.decoder2(dec3_out,            enc2_skip)
        dec1_out = self.decoder1(dec2_out,            enc1_skip)

        # ── Final prediction ──
        dose_logits    = self.final_conv(dec1_out)
        dose_predicted = self.final_relu(dose_logits)

        return dose_predicted


# ─────────────────────────────────────────────────────────────
# CHECKPOINT 3 — Verify the network builds and runs
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=== Phase 3 Checkpoint: Building 3D U-Net ===\n")

    # Build the model
    model = UNet3D(in_channels=11, base_filters=32)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model built successfully")
    print(f"   Trainable parameters: {total_params:,}")
    print(f"   (~{total_params / 1e6:.1f} million)\n")

    # Create a fake batch of size 1 to test the forward pass
    # Shape: (batch=1, channels=11, D=128, H=128, W=128)
    fake_input = torch.zeros(1, 11, 128, 128, 128)
    print(f"Input tensor shape  : {fake_input.shape}")

    # Run a forward pass (no training, just checking shapes)
    with torch.no_grad():
        fake_output = model(fake_input)

    print(f"Output tensor shape : {fake_output.shape}")
    print(f"Output value range  : [{fake_output.min():.3f}, {fake_output.max():.3f}]")
    print(f" Phase 3 PASSED — U-Net forward pass successful!")
    print(f"   Input  (B, 11, 128, 128, 128) → CT + 10 organ channels")
    print(f"   Output (B,  1, 128, 128, 128) → predicted dose map")