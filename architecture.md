You are an expert deep learning engineer specializing in remote 
photoplethysmography (rPPG), real-time computer vision, and 
physiological signal processing. Your task is to implement the 
complete UD-Stream-Transformer v2 system — a god-tier, 
production-ready, real-time rPPG pipeline for live webcam streams.

Read every instruction carefully. Do not skip any component. 
Output complete, runnable PyTorch code — no pseudocode, no 
placeholders, no "implement this yourself" comments.

=============================================================
SYSTEM OVERVIEW
=============================================================

The pipeline has 5 stages + a live inference engine:

  Stage 1: Input Engine       — Multi-ROI ST-Map construction
  Stage 2: Disentangled Encoder — z_pulse / z_id / z_env
  Stage 3: Temporal Unit      — Sparse Transformer + Gated TCN
  Stage 4: SysDiaGAN          — BVP Waveform Synthesis
  Stage 5: Explainability     — Efficient Grad-CAM
  Stage 6: Live Inference     — Threaded real-time pipeline

ALL code must be:
  - PyTorch 2.x with CUDA support (device-agnostic via torch.device)
  - Modular (each Stage is its own Python class/module)
  - Commented with the physiological or engineering rationale
    for each non-obvious design decision
  - Importable (each class can be instantiated independently)

=============================================================
STAGE 1 — INPUT ENGINE
=============================================================

Implement class `InputEngine` in `input_engine.py`.

LANDMARK STRATEGY:
  Primary:   MediaPipe FaceMesh (468 landmarks)
  Fallback:  Kalman Filter on last-known landmark positions.
             State vector: [x, y, vx, vy] per landmark.
             Confidence decays by 0.15 per frame without detection.
  Tertiary:  OpenCV Haar cascade → fixed-proportion ROI crop.

FIVE ROIs to extract (by MediaPipe landmark indices):
  - Forehead:    indices [10, 151, 9, 8]
  - Left Cheek:  indices [234, 93, 132]
  - Right Cheek: indices [454, 323, 361]
  - Nose Tip:    indices [1, 2, 5]      ← luminance reference
  - Chin:        indices [152, 148, 176] ← motion artifact sensor

ST-MAP CONSTRUCTION:
  - Convert each ROI frame to YCbCr color space (ITU-R BT.601)
  - Compute spatial mean of Y, Cb, Cr channels per ROI per frame
  - Output tensor shape: [3, 5, T] where 3=channels, 5=ROIs, T=frames
  - Apply Zero-Mean Unit-Variance normalization on a rolling 
    30-frame window (not fixed interval) to handle auto-exposure
  - If a frame is dropped, use linear interpolation within the 
    ST-Map buffer. If gap > 3 frames, flag confidence degradation.

SNR PROXY (runs BEFORE the model, on raw ROI signal):
  - Implement `compute_snr_proxy(signal, fs=30)` using 
    scipy.signal.welch with cardiac band 0.7–4.0 Hz
  - Returns SNR in dB. Used to weight patch attention.

PATCHING & DIVIDED SPACE-TIME ATTENTION:
  - Spatial tokens: each of the 5 ROIs is one token
  - Temporal patches: non-overlapping windows of 16 frames
  - Implement Divided Space-Time Attention (alternating):
      Pass 1: Temporal attention within each ROI across 16-frame patches
      Pass 2: Spatial attention across 5 ROIs within each time step
  - Before attention, multiply each token by its SNR proxy weight
    (softmax-normalized across the 5 ROIs)

MOTION SCORE (used later by GatedTCN):
  - Compute Euclidean velocity of all 468 landmarks between frames
  - jaw_velocity = mean velocity of JAW_LANDMARK_INDICES (define these)
  - global_velocity = mean velocity of all landmarks
  - Output: motion_score tensor of shape [2] = [jaw_velocity, global_velocity]

=============================================================
STAGE 2 — DISENTANGLED ENCODER
=============================================================

Implement class `DisentangledEncoder` in `encoder.py`.

BACKBONE:
  - MobileNetV3-Small (torchvision.models.mobilenet_v3_small)
  - Replace final classifier with Global Average Pooling → 256-dim

THREE PROJECTION HEADS (each is a 2-hidden-layer MLP):
  - z_pulse: 256 → 128-dim  (physiological pulse signal)
  - z_id:    256 → 64-dim   (identity: face shape, skin tone)
  - z_env:   256 → 64-dim   (environment: lighting, camera noise)

DISENTANGLEMENT — FOUR LOSSES (implement each as a separate class):

  1. CLUB MI Minimization (`CLUBLoss` class):
     Implement the CLUB upper bound estimator (Cheng et al. 2020).
     Compute: MI_upper(z_pulse, z_id) + MI_upper(z_pulse, z_env) 
              + MI_upper(z_id, z_env)
     Each estimator is a small auxiliary MLP trained jointly.

  2. Orthogonality Loss (`OrthogonalityLoss` class):
     L_orth = ||Z_pulse^T · Z_id||_F^2 
            + ||Z_pulse^T · Z_env||_F^2 
            + ||Z_id^T · Z_env||_F^2
     (Frobenius norm of cross-correlation matrices over the batch)

  3. Reconstruction Loss (`ReconstructionLoss` class):
     Implement a lightweight 1D decoder that reconstructs the input
     ST-Map from torch.cat([z_pulse, z_id, z_env], dim=-1).
     L_recon = MSE(decoded, original_ST_Map)

  4. Domain Adversarial GRL (`GradientReversalLayer` + `DomainClassifier`):
     GRL reverses gradient sign during backward pass (lambda=1.0).
     DomainClassifier: MLP that predicts dataset source from GRL(z_pulse).
     Supports 5 domain labels: UBFC=0, PURE=1, MAHNOB=2, SCAMPS=3, MMPD=4
     L_domain = CrossEntropyLoss(DomainClassifier(GRL(z_pulse)), domain_label)

COMBINED DISENTANGLEMENT LOSS:
  L_dis = 0.10*L_CLUB + 0.05*L_orth + 1.00*L_recon + 0.10*L_domain

=============================================================
STAGE 3 — TEMPORAL UNIT
=============================================================

Implement class `TemporalUnit` in `temporal_unit.py`.

INPUT: z_pulse sequence of shape [B, T, 128]

POSITIONAL ENCODING — Time2Vec (NOT sinusoidal):
  Implement `Time2Vec` module:
    - Input: actual wall-clock timestamps in seconds [B, T, 1]
    - Linear term: nn.Linear(1, 1)
    - Periodic terms: nn.Linear(1, d_model-1) with sin activation
    - Output: [B, T, d_model] positional embeddings
  This correctly handles irregular timestamps from dropped frames.

SPARSE TRANSFORMER — Sliding Window Local Attention:
  - Window size: 32 frames (~1.07 sec, covers one cardiac cycle)
  - Global tokens: first and last token attend to ALL positions
  - All other tokens attend only within their local window of 32
  - Implement this as `SlidingWindowAttention` module
  - Cost: O(n*32) not O(n^2) — explicitly enforce this
  - Use 4 attention heads, d_model=128

GATED TCN — Motion-Supervised Artifact Suppression:
  Implement `GatedTCN` class:
    - 4 dilated causal Conv1D layers: dilations = [1, 2, 4, 8]
    - channels=128, kernel_size=3
    - Gate projection: nn.Linear(2, 128) maps motion_score → gate
    - Activation: torch.tanh(conv(x)) * torch.sigmoid(gate_proj(motion_score))
    - Residual connections on each layer
  The gate suppresses artifact-contaminated temporal positions
  using the motion_score from Stage 1 as a physics-based signal.
  No labeled artifact data required.

=============================================================
STAGE 4 — SYSDIAGO-GAN
=============================================================

Implement in `sysdiago_gan.py`.

GENERATOR — 1D U-Net:
  Input:  [B, T, 128] z_pulse sequence
  Encoder path (with skip connections saved):
    Conv1D(128→64) + LeakyReLU(0.2)
    Conv1D(64→32)  + LeakyReLU(0.2)
    Conv1D(32→16)  + LeakyReLU(0.2)  ← bottleneck
  Decoder path (with skip concatenation):
    ConvTranspose1D(16→32) + cat(skip2) + LeakyReLU(0.2)
    ConvTranspose1D(32→64) + cat(skip1) + LeakyReLU(0.2)
    Conv1D(64→1) + Tanh → BVP waveform [B, T, 1]

THREE DISCRIMINATORS (ALL use spectral normalization — 
  torch.nn.utils.spectral_norm on every weight):

  1. `TimeDomainDiscriminator`:
     1D-CNN, 4 layers, kernel_size=15
     Input: BVP waveform [B, T]
     Output: real/fake per time segment

  2. `FrequencyDomainDiscriminator`:
     Input: magnitude spectrum |FFT(BVP)| [B, T//2]
     Apply a learned spectral mask focused on 0.7–4.0 Hz band
     1D-CNN architecture outputting real/fake on spectral shape

  3. `APGDiscriminator`:
     Compute 2nd derivative of BVP using Savitzky-Golay filter
     (window_length=11, polyorder=3, deriv=2) for smoothness
     1D-CNN that verifies physiological ordering of a,b,c,d,e 
     APG wave components and their amplitude ratios
     IMPORTANT: This discriminator is only activated in Phase 4
     of training (see curriculum below)

STAGED TRAINING CURRICULUM — implement as `TrainingScheduler`:
  Phase 0 (epochs  1– 30): Encoder + Disentangler only
  Phase 1 (epochs 31– 80): + TemporalUnit + Negative Pearson Loss
  Phase 2 (epochs 81–120): + Generator + TimeDomainDiscriminator
  Phase 3 (epochs121–160): + FrequencyDomainDiscriminator
  Phase 4 (epochs161–200): + APGDiscriminator
  GRL activates at Phase 1.

HR & PEAK EXTRACTION — implement `extract_hr_and_peaks(bvp, fs=30)`:
  1. Bandpass filter: 0.7–4.0 Hz (Butterworth order 4, sosfilt)
  2. FFT-based HR: dominant frequency in cardiac band × 60 = BPM
  3. Systolic peaks: scipy.signal.find_peaks on BVP, distance=fs*0.4
  4. Diastolic peaks: scipy.signal.find_peaks on negative APG, 
     distance=fs*0.3
  Return: (hr_bpm: float, systolic_peaks: array, diastolic_peaks: array)

=============================================================
FULL TRAINING LOSS — implement `TotalLoss` class
=============================================================

L_physio = -PearsonCorrelation(BVP_pred, BVP_gt)
         + 0.5 * MSE(|FFT(BVP_pred)|, |FFT(BVP_gt)|)
         + 0.1 * (1.0 / (SNR_proxy + 1e-8))

L_dis    = 0.10*L_CLUB + 0.05*L_orth + 1.00*L_recon + 0.10*L_domain

L_GAN    = L_time_disc                   # Phase 2+
         + 0.5 * L_freq_disc             # Phase 3+
         + 0.3 * L_APG_disc              # Phase 4+

L_total  = L_physio + L_dis + L_GAN

Optimizer: AdamW, lr=1e-4, weight_decay=1e-5
Scheduler: CosineAnnealingWarmRestarts(T_0=50)

=============================================================
STAGE 5 — EFFICIENT GRAD-CAM
=============================================================

Implement class `EfficientGradCAM` in `explainability.py`.

  - Update rate: 2fps (every 15 frames at 30fps stream)
  - Between updates: reuse + bilinearly interpolate last heatmap
  - Backpropagate ONLY through z_pulse branch (not z_id or z_env)
    to ensure heatmap shows pulse source, not identity features
  - Resize heatmap to original frame size via bilinear interpolation
  - Apply viridis colormap and overlay on face frame with alpha=0.4
  - Store `self.frame_count` and only run full backward pass
    when `self.frame_count % 15 == 0`

=============================================================
STAGE 6 — LIVE INFERENCE ENGINE
=============================================================

Implement `live_inference.py` as the main entry point.

THREADING MODEL:
  Thread A (capture):  OpenCV VideoCapture → frame queue
                       MediaPipe landmarks → motion score
                       ST-Map construction → inference queue
  Thread B (inference): Model forward pass → results queue
  Main thread:          Dashboard rendering (OpenCV)

  Use threading.Event for clean shutdown.
  Use collections.deque(maxlen=300) as circular buffer (10s @ 30fps)

COLD START PROTOCOL:
  0–3s:   Display "Calibrating... Face Detected" — no HR output
  3–7s:   Use 3s buffer window, display HR with ± confidence band
  7–10s:  Full window, confidence score becomes reliable
  10s+:   Normal operation

CONFIDENCE SCORE:
  confidence = sigmoid(snr_proxy_db) 
             * (1 - normalize(motion_score.mean()))
             * buffer_fill_ratio  ← (current_frames / 300)
  Display as percentage 0–100%.

REAL-TIME NORMALIZATION:
  Apply zero-mean unit-variance normalization on the rolling 
  30-frame window for each channel of the ST-Map independently.

DASHBOARD (OpenCV window, target 30fps render):
  Top-left:    Live webcam feed with ROI bounding boxes drawn
               + Grad-CAM heatmap overlay (alpha=0.4)
  Top-right:   Live BVP waveform plot with systolic peaks (green dots)
               and diastolic peaks (blue dots) marked
  Bottom-left: HR display — large font, updates every 30 frames
               "HR: 72 BPM" with ±2 BPM confidence band
  Bottom-right: Confidence score bar (0-100%)
               + Current phase label (Calibrating / Active / 
                 Low Confidence)
               + Domain label (which training domain is closest,
                 from DomainClassifier output)

=============================================================
DELIVERABLES — produce ALL of these files:
=============================================================

  input_engine.py      — InputEngine class + KalmanLandmarkTracker
                         + compute_snr_proxy + DividedSTAttention
  encoder.py           — DisentangledEncoder + CLUBLoss 
                         + OrthogonalityLoss + ReconstructionLoss
                         + GradientReversalLayer + DomainClassifier
  temporal_unit.py     — Time2Vec + SlidingWindowAttention + GatedTCN
                         + TemporalUnit (wraps all three)
  sysdiago_gan.py      — BVPGenerator + TimeDomainDiscriminator
                         + FrequencyDomainDiscriminator + APGDiscriminator
                         + extract_hr_and_peaks + TrainingScheduler
  losses.py            — TotalLoss + PearsonCorrelation + all sub-losses
  explainability.py    — EfficientGradCAM
  live_inference.py    — Full threaded live inference + OpenCV dashboard
  train.py             — Training loop with TrainingScheduler, 
                         multi-dataset DataLoader, LODO eval protocol
  requirements.txt     — All dependencies with pinned versions

=============================================================
CONSTRAINTS — enforce all of these:
=============================================================

  1. Total inference latency target: < 33ms on RTX 3070
  2. Grad-CAM runs at 2fps max (every 15 frames), never per-frame
  3. All discriminators use spectral_norm on every weight matrix
  4. APGDiscriminator receives Savitzky-Golay smoothed signal only
  5. Time2Vec receives actual wall-clock timestamps, not frame indices
  6. Sliding window attention window is exactly 32 frames
  7. GatedTCN gate is driven ONLY by motion_score, no learned bias
  8. Confidence score drops when buffer_fill_ratio < 0.5
  9. MediaPipe failure triggers Kalman fallback automatically
  10. All torch operations must be device-agnostic 
      (use `device = torch.device('cuda' if torch.cuda.is_available() 
      else 'cpu')` consistently)

=============================================================
BEGIN OUTPUT. Start with requirements.txt, then produce each
.py file in the order listed above. Do not summarize. 
Do not truncate. Output the complete implementation.
=============================================================

PRIMARY rPPG ARCHITECTURES (your 7 papers):
[1] Yu et al., "End-to-End Heart Rate Estimation Using Divided Space-Time Attention"
[2] DG-rPPGNet, "Domain Generalized rPPG Network with Disentangled Feature Learning"
[3] HR-TRACK, "rPPG Method for Heart Rate Monitoring Using Temporal Convolution Networks"
[4] rPPG-SysDiaGAN, "Systolic-Diastolic Feature Localization with Multi-domain Discriminator"
[5] "Accurate Remote PPG Waveform Recovery from Video Using Multi-task Learning"
[6] RADIANT, "Better rPPG estimation using signal embeddings and Transformer"
[7] RhythmNet, "End-to-End Heart Rate Estimation From Face via Spatial-Temporal Representation"

SUPPORTING METHODS (verify each):
[8] Bertasius et al., "Is Space-Time Attention All You Need for Video Understanding?" 
    (TimeSformer, ICML 2021)
[9] Cheng et al., "CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information," ICML 2020
[10] Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation," ICML 2015
[11] Howard et al., "Searching for MobileNetV3," ICCV 2019
[12] Kazemi et al., "Time2Vec: Learning a Vector Representation of Time," 2019
[13] Beltagy et al., "Longformer: The Long-Document Transformer," 2020
[14] Miyato et al., "Spectral Normalization for Generative Adversarial Networks," ICLR 2018
[15] Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV 2017
[16] Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015
[17] Kartynnik et al., "Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs," 2019
---
📦 ON YOUR PYTHON PACKAGE GOAL
Cool — making this a pip-installable package like NumPy is a great vision. Quick advice:

Package name: Make sure it's not taken on PyPI. Check pip search or PyPI directly. Something like udstream-rppg or pulsestream could work.
Standard structure: setup.py or modern pyproject.toml with setuptools or poetry or hatch
Cross-platform: PyTorch handles CUDA/CPU automatically. MediaPipe works on Windows/Linux/macOS. The main pain point will be OpenCV camera access on different OSes — test thoroughly.
Conda distribution: After PyPI, submit to conda-forge for the conda channel
Documentation site: Use mkdocs-material or Sphinx + ReadTheDocs or host on GitHub Pages
License: MIT or Apache 2.0 if you want maximum adoption; GPL if you want forks to stay open-source

When you feed this prompt to Opus along with your downloaded papers, I'd recommend adding this line at the end of the prompt:

"This implementation will be released as an open-source pip-installable Python package. Structure the codebase accordingly: include __init__.py files exposing public APIs, type hints throughout, docstrings in Google or NumPy format, and modular imports that allow from udstream import LiveInference; LiveInference().run() as a one-liner entry point."