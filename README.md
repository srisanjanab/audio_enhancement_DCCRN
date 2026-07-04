# 🎙️ DCCRN — Speech Enhancement on VoiceBank-DEMAND

> Deep Learning Assignment — Noise-Robust Audio Signal Enhancement  
> Built with PyTorch + HuggingFace Datasets on Google Colab (GPU)

---

## 📌 Overview

This project implements a **Deep Complex Convolutional Recurrent Network (DCCRN)** for speech enhancement. The model takes noisy speech as input and outputs a cleaner, denoised version by learning to suppress background noise while preserving speech quality.

The key advantage over a standard U-Net is that DCCRN processes both the **real and imaginary** parts of the spectrogram together, recovering phase information that real-valued networks discard.

---

## 📂 Dataset

**VoiceBank-DEMAND** (16 kHz) loaded directly from HuggingFace:

```python
ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
```

| Split | Samples |
|-------|---------|
| Train | 11,572  |
| Test  | 824     |

Each sample contains a pair of `noisy` and `clean` waveforms at 16 kHz. Training noise types include 28 different real-world noise environments at SNR levels of 0, 5, 10, and 15 dB.

---

## 🏗️ Model Architecture

```
Noisy waveform  [B, L]
      │
   STFT  (FFT=512, hop=128, hann window)
      │
  [Re, Im]  →  stacked as 2-channel input
      │
  Encoder  (3 × Conv2d + BatchNorm + PReLU)
      │
  LSTM bottleneck  (2 layers, hidden=256)
      │
  Decoder  (3 × ConvTranspose2d + BatchNorm + PReLU)
      │
  Enhanced spectrogram  [Re, Im]
      │
   iSTFT  →  Enhanced waveform  [B, L]
```

**STFT settings:**
- FFT size: 512
- Hop length: 128
- Window: Hann
- Output bins: 257 frequency bins

---

## ⚙️ Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Batch size | 2 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Epochs | 10 |
| Loss function | SI-SNR |
| Hardware | Google Colab (T4 GPU) |

### SI-SNR Loss

Scale-Invariant Signal-to-Noise Ratio — handles varying recording levels without normalisation:

```python
def si_snr_loss(est, target):
    target = target - target.mean(dim=-1, keepdim=True)
    est    = est    - est.mean(dim=-1, keepdim=True)
    s_target = torch.sum(est * target, dim=-1, keepdim=True) * target / (...)
    e_noise  = est - s_target
    loss = -10 * torch.log10(sum(s_target²) / sum(e_noise²))
    return loss.mean()
```

---

## 📈 Training Results

Loss decreases consistently across all 10 epochs, showing the model is learning to suppress noise:

| Epoch | SI-SNR Loss |
|-------|-------------|
| 1  | -8.0138 |
| 2  | -8.3447 |
| 3  | -8.5710 |
| 4  | -8.7434 |
| 5  | -8.8660 |
| 6  | -8.9445 |
| 7  | -9.0188 |
| 8  | -9.0783 |
| 9  | -9.1399 |
| 10 | -9.1769 |

> Loss is negative because we minimise −SI-SNR (i.e., maximise SNR). More negative = better enhancement.

---

## 📊 Spectrogram Visualisation

The notebook plots three spectrograms side-by-side for visual comparison:

| Noisy Input | Clean Reference | DCCRN Enhanced |
|:-----------:|:---------------:|:--------------:|
| Background noise visible across all frequencies | Clean speech harmonics only | Noise suppressed, speech structure preserved |

---

## 🔊 Audio Evaluation

The notebook plays back audio for both **seen (train)** and **unseen (test)** samples:

**Training sample (seen data):**
- 🔊 Noisy input
- 🔊 Enhanced output
- 🔊 Clean reference

**Test sample (unseen data):**
- 🔊 Noisy input  
- 🔊 Enhanced output
- 🔊 Clean reference

Output WAV files saved: `enhanced.wav`, `noisy.wav`

---

## 🚀 How to Run

### On Google Colab (recommended)

1. Open the notebook in Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells top to bottom — dataset downloads automatically

### Install dependencies

```bash
pip install datasets pesq pystoi
```

> **Windows users:** `pesq` requires Microsoft C++ Build Tools to install.  
> On Colab (Linux) it installs without any issues.

---

## 📁 Project Structure

```
├── DeepLearning_assign.ipynb   # Main Colab notebook (complete pipeline)
└── README.md
```

The notebook contains everything in one place:
- Data loading
- STFT/iSTFT utilities
- DCCRN model definition
- SI-SNR loss
- Training loop
- Spectrogram plots
- Audio playback (seen + unseen samples)

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Model, training, STFT ops |
| `torchaudio` | Audio I/O |
| `datasets` | HuggingFace VoiceBank-DEMAND loader |
| `pesq` | Perceptual speech quality metric |
| `pystoi` | Intelligibility metric |
| `matplotlib` | Spectrogram plots |

---

## 📚 Reference

- Hu et al. (2020). *DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement.* INTERSPEECH 2020. [arXiv:2008.00264](https://arxiv.org/abs/2008.00264)
- VoiceBank-DEMAND dataset: Valentini-Botinhao et al. (2016)
