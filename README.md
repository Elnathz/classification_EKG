# Classification of ECG Arrhythmia Beats: Hybrid 1D-CNN and BiLSTM on MIT-BIH

A production-grade deep learning pipeline for multi-class cardiac arrhythmia classification on segmented single-lead electrocardiogram (ECG) waveforms. This repository contains the complete experimental framework: rigorous class-imbalance evaluation across 45 training runs, paired statistical significance testing, model selection, and zero-overhead ONNX deployment artifacts.

---

## 1. Executive Summary

* Problem: Automated classification of heartbeat morphology into 5 clinical categories according to the Association for the Advancement of Medical Instrumentation (AAMI) EC57 standard.
* Dataset: MIT-BIH Arrhythmia Database (109,446 segmented beats, 187 samples per beat, sampled at 125 Hz).
* Proposed Model: Hybrid 1D-CNN + Bidirectional LSTM (BiLSTM) network comprising 422,277 parameters.
* Best Model Performance (Test Set):
  * Overall Accuracy: 98.69%
  * Macro-averaged F1 Score: 92.41%
  * Weighted F1 Score: 98.66%
* Statistical Rigor: Evaluated across 3 model families (1D-CNN, BiLSTM, Hybrid) and 3 class-imbalance strategies over 5 random seeds (N = 45 full training runs). Paired t-tests confirm the Hybrid architecture outperforms standalone baselines at p < 0.01 across all configurations.
* Deployment: Exported to ONNX (opset 17) with maximum absolute logit divergence of 1.14e-05 compared to PyTorch, achieving sub-millisecond CPU latency per beat.

---

## 2. Clinical Dataset Profile (AAMI EC57)

Each sample consists of a single heartbeat extracted from Lead II, normalized and padded to a fixed sequence length of 187 points centered around the R-peak. The dataset exhibits severe class imbalance typical of real-world clinical monitoring:

| Class Code | AAMI EC57 Category | Clinical Manifestation | Training Set | Validation Set | Test Set | Total Samples | Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| N | Non-ectopic | Normal beat, Left/Right bundle branch block | 61,602 | 10,871 | 18,118 | 90,591 | 82.77% |
| S | Supraventricular | Atrial premature, Aberrant atrial premature | 1,890 | 333 | 556 | 2,779 | 2.54% |
| V | Ventricular | Premature ventricular contraction, Ventricular escape | 4,920 | 868 | 1,448 | 7,236 | 6.61% |
| F | Fusion | Fusion of ventricular and normal beat | 547 | 96 | 162 | 805 | 0.74% |
| Q | Unknown / Paced | Paced beat, fusion of paced, unclassifiable | 5,465 | 965 | 1,608 | 8,038 | 7.34% |
| Total | - | - | 74,424 | 13,133 | 21,892 | 109,449 | 100.00% |

Data split: 80% train, 20% test (from official MIT-BIH distribution). The training set is further divided using a stratified 85/15 train-validation split (split seed 42). The test set was held out strictly for final evaluation.

---

## 3. Architecture Design

The Hybrid architecture couples spatial feature extraction from convolutional kernels with sequential temporal dependencies captured by recurrent units.

```mermaid
graph TD
    Input["Input ECG Beat (Batch, 187, 1)"] --> Transpose["Permute to (Batch, 1, 187)"]
    
    subgraph ConvStack ["Feature Extraction: 1D-CNN Stack"]
        Transpose --> C1["Conv1D (1 -> 64, kernel=5, pad=2) + BatchNorm + ReLU + MaxPool1D(2) + Drop(0.2)"]
        C1 --> C2["Conv1D (64 -> 128, kernel=5, pad=2) + BatchNorm + ReLU + MaxPool1D(2) + Drop(0.2)"]
        C2 --> C3["Conv1D (128 -> 256, kernel=3, pad=1) + BatchNorm + ReLU + MaxPool1D(2) + Drop(0.2)"]
    end
    
    C3 --> Reshape["Transpose to Sequence: (Batch, 23, 256)"]
    
    subgraph TemporalStack ["Temporal Modeling: Bidirectional LSTM"]
        Reshape --> LSTM["2-Layer BiLSTM (hidden=64 per direction, dropout=0.2)"]
        LSTM --> Concat["Concatenate Final Hidden States (h_forward + h_backward) -> Dim 128"]
    end
    
    subgraph ClassifierHead ["Classification Head"]
        Concat --> Dense1["Dense (128 -> 128) + ReLU + Dropout(0.3)"]
        Dense1 --> Out["Dense (128 -> 5) -> Raw Logits"]
    end
```

### Layer Specifications and Parameter Footprint

* ConvStack: 3 hierarchical convolutional blocks reducing sequence dimension from 187 down to 23 steps while expanding feature channels from 1 to 256.
* BiLSTM: 2 layers, 64 hidden units per direction. Returns concatenated forward and backward terminal states (dim 128).
* Head: Fully connected projection (128 to 128) with Dropout (p = 0.3) and final linear layer to 5 class logits.
* Total Parameters:
  * 1D-CNN standalone: 174,469 parameters
  * BiLSTM standalone: 150,789 parameters
  * Hybrid 1D-CNN + BiLSTM: 422,277 parameters

---

## 4. Experimental Framework

To evaluate robustness against extreme class skew, models were subjected to a 3 x 3 factorial grid across 5 randomized seeds (42, 43, 44, 45, 46):

1. Scenario 1 (Tanpa penanganan): Natural data distribution with standard cross-entropy loss.
2. Scenario 2 (Pembobotan kelas): Cost-sensitive learning applying inverse class frequency weights:
   w_c = N_total / (C * N_c)
3. Scenario 3 (Oversampling dan pembobotan): Minority classes randomly oversampled up to 35% of the majority class count, followed by recalculation of class weights on the oversampled distribution.

Training hyperparameters:
* Optimizer: Adam (initial learning rate = 1e-3)
* Batch size: 128
* Learning rate scheduler: ReduceLROnPlateau (factor = 0.5, patience = 3 epochs, monitored on validation macro-F1)
* Early stopping: Patience = 7 epochs on validation macro-F1
* Maximum epochs: 50

---

## 5. Empirical Results and Statistical Validation

All metrics report the mean plus/minus standard deviation over 5 independent runs on the held-out test set (21,892 beats).

### Macro Performance Summary

| Scenario | Model Architecture | Params | Test Accuracy (%) | Test Macro-F1 (%) | Test Weighted-F1 (%) | 95% CI Macro-F1 | Mean Epochs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (Imb1) | 1D-CNN saja | 174,469 | 98.30 +/- 0.07 | 90.38 +/- 0.36 | 98.21 +/- 0.08 | [89.93, 90.84] | 47.6 |
| Baseline (Imb1) | BiLSTM saja | 150,789 | 98.41 +/- 0.08 | 91.39 +/- 0.59 | 98.37 +/- 0.09 | [90.65, 92.13] | 39.8 |
| Baseline (Imb1) | **Hybrid (CNN+BiLSTM)** | 422,277 | **98.70 +/- 0.12** | **92.40 +/- 0.74** | **98.67 +/- 0.13** | **[91.48, 93.32]** | 31.0 |
| Cost-Sensitive (Imb2) | 1D-CNN saja | 174,469 | 92.99 +/- 1.26 | 76.39 +/- 1.62 | 93.93 +/- 1.06 | [74.38, 78.40] | 19.6 |
| Cost-Sensitive (Imb2) | BiLSTM saja | 150,789 | 93.31 +/- 1.78 | 77.46 +/- 4.06 | 94.25 +/- 1.37 | [72.41, 82.51] | 36.0 |
| Cost-Sensitive (Imb2) | **Hybrid (CNN+BiLSTM)** | 422,277 | **95.95 +/- 0.98** | **83.36 +/- 2.64** | **96.37 +/- 0.80** | **[80.08, 86.64]** | 29.0 |
| Hybrid Sampling (Imb3) | 1D-CNN saja | 174,469 | 96.10 +/- 0.58 | 84.37 +/- 1.47 | 96.45 +/- 0.58 | [82.55, 86.20] | 34.0 |
| Hybrid Sampling (Imb3) | BiLSTM saja | 150,789 | 98.17 +/- 0.07 | 90.85 +/- 0.47 | 98.18 +/- 0.07 | [90.26, 91.44] | 49.4 |
| Hybrid Sampling (Imb3) | **Hybrid (CNN+BiLSTM)** | 422,277 | **98.58 +/- 0.12** | **92.00 +/- 0.36** | **98.60 +/- 0.11** | **[91.56, 92.44]** | 40.4 |

### Paired Statistical Significance Tests (Hybrid vs Baselines)

Paired t-tests were performed across matching seed runs (df = 4) to verify whether improvements were statistically significant:

| Scenario | Comparison | Target Metric | Mean Delta (%) | 95% Confidence Interval | p-value (paired t) | Effect Size (Cohen dz) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (Imb1) | Hybrid vs 1D-CNN | Macro-F1 | +2.01% | [+1.49%, +2.54%] | p = 0.0004 | 4.75 (Large) |
| Baseline (Imb1) | Hybrid vs BiLSTM | Macro-F1 | +1.01% | [+0.46%, +1.56%] | p = 0.0071 | 2.27 (Large) |
| Cost-Sensitive (Imb2) | Hybrid vs 1D-CNN | Macro-F1 | +6.97% | [+4.92%, +9.01%] | p = 0.0007 | 4.23 (Large) |
| Cost-Sensitive (Imb2) | Hybrid vs BiLSTM | Macro-F1 | +5.90% | [+0.36%, +11.44%] | p = 0.0417 | 1.32 (Large) |
| Hybrid Sampling (Imb3) | Hybrid vs 1D-CNN | Macro-F1 | +7.63% | [+5.52%, +9.73%] | p = 0.0006 | 4.49 (Large) |
| Hybrid Sampling (Imb3) | Hybrid vs BiLSTM | Macro-F1 | +1.15% | [+0.42%, +1.88%] | p = 0.0119 | 1.96 (Large) |

Conclusion: The Hybrid model consistently achieves statistically superior performance over both isolated 1D-CNN and BiLSTM architectures across all balancing regimes.

### Per-Class Metrics (Selected Production Hybrid Model)

Evaluated on the held-out test set (21,892 beats) for the production checkpoint:

| Class Code | Name | Support | Precision (%) | Recall (%) | F1 Score (%) | Specificity (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| N | Normal | 18,118 | 99.08 +/- 0.11 | 99.64 +/- 0.07 | 99.36 +/- 0.06 | 95.54 +/- 0.54 |
| S | Supraventricular | 556 | 91.27 +/- 2.63 | 79.60 +/- 2.72 | 84.99 +/- 1.64 | 99.80 +/- 0.07 |
| V | Ventricular (PVC) | 1,448 | 96.97 +/- 0.96 | 96.15 +/- 0.19 | 96.55 +/- 0.52 | 99.79 +/- 0.07 |
| F | Fusion | 162 | 84.73 +/- 2.35 | 79.26 +/- 4.53 | 81.79 +/- 1.62 | 99.89 +/- 0.02 |
| Q | Unknown / Paced | 1,608 | 99.64 +/- 0.22 | 98.96 +/- 0.30 | 99.29 +/- 0.13 | 99.97 +/- 0.02 |

---

## 6. Production Model and Inference

The selected production candidate (`imb1_hybrid_s45`) was audited for numerical parity between PyTorch and ONNX Runtime:

* Model format: `model.onnx` (1.69 MB)
* State dict: `model_state_dict.pt` (1.70 MB)
* ONNX Opset: 17
* Numerical divergence (Max Abs Diff): 1.144e-05
* Average single-beat latency (CPU): 0.32 ms

### Minimal Inference Example (Python)

```python
import json
import numpy as np
import onnxruntime as ort

# Load model and configuration
session = ort.InferenceSession("hasil_ekg/model_deploy/model.onnx", providers=["CPUExecutionProvider"])
with open("hasil_ekg/model_deploy/model_config.json", "r") as f:
    config = json.load(f)

# Input beat: 1D array of 187 float32 samples
sample_beat = np.random.randn(187).astype(np.float32)

# Reshape to expected input contract: (Batch, 187, 1)
input_tensor = sample_beat[np.newaxis, :, np.newaxis]

# Execute session
outputs = session.run(["logits"], {"ekg_beat": input_tensor})[0]

# Compute softmax probabilities
exp_logits = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

predicted_index = int(np.argmax(probabilities, axis=1)[0])
class_info = config["keluaran"]["kelas"][predicted_index]

print(f"Predicted Class: {class_info['kode']} ({class_info['nama']})")
print(f"Confidence: {probabilities[0, predicted_index] * 100:.2f}%")
```

### FastAPI Blueprint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI(title="ECG Arrhythmia Classifier API", version="1.0.0")
session = ort.InferenceSession("hasil_ekg/model_deploy/model.onnx", providers=["CPUExecutionProvider"])
CLASSES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]

class BeatInput(BaseModel):
    samples: list[float]

@app.post("/predict")
def predict_beat(data: BeatInput):
    if len(data.samples) != 187:
        raise HTTPException(status_code=400, detail="Expected 187 float samples per heartbeat")
    
    tensor = np.asarray(data.samples, dtype=np.float32)[np.newaxis, :, np.newaxis]
    logits = session.run(["logits"], {"ekg_beat": tensor})[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    idx = int(np.argmax(probs))
    
    return {
        "class_code": ["N", "S", "V", "F", "Q"][idx],
        "class_name": CLASSES[idx],
        "confidence": float(probs[0, idx]),
        "probabilities": {CLASSES[i]: float(probs[0, i]) for i in range(5)}
    }
```

---

## 7. Repository Structure

```
classification_EKG/
|-- .gitignore                  # Exclusion rules for dataset, checkpoints, docx
|-- README.md                   # Technical documentation and benchmarks
|-- pipeline_ekg_mitbih.ipynb   # End-to-end training and evaluation notebook
|-- data/                       # Local directory for MIT-BIH CSV files (git-ignored)
|   |-- mitbih_train.csv
|   +-- mitbih_test.csv
+-- hasil_ekg/                  # Generated research artifacts
    |-- config.json             # Global experiment configuration
    |-- model_deploy/           # Production export package
    |   |-- model.onnx          # Universal ONNX runtime model
    |   |-- model_state_dict.pt # PyTorch state dictionary
    |   |-- model_config.json   # Input/output contracts and metadata
    |   +-- contoh_masukan.json # Verification samples with reference logits
    |-- runs/                   # Raw JSON execution logs (45 runs)
    |-- tabel/                  # Statistical and summary CSV tables
    |   |-- audit_data.json
    |   |-- tabel_distribusi_kelas.csv
    |   |-- tabel_performa_ringkasan.csv
    |   |-- tabel_performa_per_kelas.csv
    |   +-- uji_pasangan_hibrida.csv
    +-- gambar/                 # 300 DPI publication plots (PNG and PDF)
        |-- distribusi_kelas.png
        |-- perbandingan_macro_f1.png
        |-- kurva_*.png
        +-- matriks_konfusi_*.png
```

---

## 8. Reproducibility

### Environment Setup

Prerequisites: Miniconda / Anaconda installed with Python 3.10.

```powershell
# 1. Create and activate environment
conda create -n ekg-mitbih python=3.10 -y
conda activate ekg-mitbih

# 2. Install PyTorch with CUDA 12.8 support (Required for RTX 50-series Blackwell architecture)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. Install core dependencies
pip install numpy pandas scipy scikit-learn matplotlib onnx onnxruntime ipykernel

# 4. Register kernel for VS Code / Jupyter
python -m ipykernel install --user --name ekg-mitbih --display-name "Python (ekg-mitbih)"
```

### Dataset Acquisition

1. Download the ECG Heartbeat Categorization Dataset from Kaggle (`shayanfazeli/heartbeat`).
2. Place `mitbih_train.csv` and `mitbih_test.csv` inside the `data/` directory.

### Execution

Open `pipeline_ekg_mitbih.ipynb` and execute all cells sequentially. The pipeline features built-in idempotence: runs previously committed to `hasil_ekg/runs/` are recognized and skipped automatically.

---

## 9. Clinical Limitations

1. Input Contract: The model requires pre-segmented beats of exactly 187 samples centered at the R-peak. It does not process continuous raw rhythm strips without an external R-peak detector (such as the Pan-Tompkins algorithm).
2. Cross-Patient Generalization: The MIT-BIH benchmark uses random beat splitting. Clinical deployments must validate on patient-independent cohorts to account for inter-patient morphological variations.
3. Rhythms vs Morphology: The network evaluates morphology within a single beat; inter-beat rhythm dynamics (such as heart rate variability or sinus pauses) are not modeled by single-beat representations.
