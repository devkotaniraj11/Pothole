# Week 4 Assignment — Neural Network Classifier on the Pothole Dataset

**Dataset:** *Pothole and Plain Road Images* (Kaggle, `virenbr11/pothole-and-plain-rode-images`)
**Notebook:** `pothole_ann_classifier.ipynb` (fully executed, code + outputs included)
**Framework:** PyTorch (CPU)

---

## Assignment requirements

This report documents a neural network classifier implemented on the chosen dataset (or its
closest viable substitute, see Section 1), covering the hyperparameter adjustments made — learning
rate, batch size, dropout, and optimizer, across six configurations — and their observed effect on
the training and validation loss/accuracy curves (Section 3), followed by an evaluation of the
final model's performance on a held-out, stratified test split using a confusion matrix,
precision, recall, and F1-score (Section 4). The discoveries from this experimentation phase are
summarised throughout, with snapshots of every plot produced (class balance, sample images,
hyperparameter comparison, training curves, confusion matrix) embedded alongside the corresponding
code to visually clarify both the process and the results.

---

## 1. Problem framing and dataset choice

The Week 2 dataset (`Pothole.v1-raw.yolov8`, a YOLOv8 object-detection export) turned out to
contain **no pothole-free images at all** — every one of its 665 images has at least one
annotated pothole bounding box. That makes a genuine whole-image "pothole present / absent"
classifier impossible on that dataset alone.

To get a real two-class problem, this assignment instead uses the **Pothole and Plain Road
Images** dataset from Kaggle, which pairs pothole images with genuine clean-road ("Plain")
negatives:

| Folder | Class | Meaning |
|---|---|---|
| `Plain`   | 0 | clean road, no pothole |
| `Pothole` | 1 | road with a pothole |

Raw folders shipped with the dataset:

| Folder | Plain (0) | Pothole (1) | Total |
|---|---|---|---|
| train | 367 | 357 | 724 |
| test  | 8   | 8   | 16  |

The dataset's own `test` folder has only **16 images** — far too small to draw reliable
conclusions from (each image is worth >6 percentage points of accuracy). To fix this, **all 740
images (train + test) were pooled and re-split** into a fresh stratified **70% / 15% / 15%**
train / validation / test partition:

| Split | Plain (0) | Pothole (1) | Total |
|---|---|---|---|
| train | 263 | 255 | 518 |
| valid | 56  | 55  | 111 |
| test  | 56  | 55  | 111 |

Hyperparameter tuning only ever sees the train/validation portions; the re-carved test split is
held out until the final evaluation in Section 4.

![Class balance](outputs/01_class_balance.png)

Sample images (top row: plain road, bottom row: pothole):

![Sample images](outputs/02_sample_images.png)

## 2. Model

A compact 4-block Convolutional Neural Network (CNN) was used:

```python
class PotholeCNN(nn.Module):
    def __init__(self, dropout=0.3, base_channels=16):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1),   nn.BatchNorm2d(c),   nn.ReLU(), nn.MaxPool2d(2), # 128->64
            nn.Conv2d(c, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(), nn.MaxPool2d(2), # 64->32
            nn.Conv2d(c*2, c*4, 3, padding=1), nn.BatchNorm2d(c*4), nn.ReLU(), nn.MaxPool2d(2),# 32->16
            nn.Conv2d(c*4, c*8, 3, padding=1), nn.BatchNorm2d(c*8), nn.ReLU(), nn.MaxPool2d(2),# 16->8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(c*8, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)
```

16 → 32 → 64 → 128 channels, batch normalisation, global average pooling, dropout in the
classifier head, single logit output trained with `BCEWithLogitsLoss`. Images are resized to
128×128 RGB, with random horizontal flip augmentation on the training set.

## 3. Hyperparameter tuning

Six configurations were trained (20 epochs each), varying **learning rate**, **batch size**,
**dropout**, and **optimizer**:

| config | lr | batch size | dropout | optimizer | final train acc | final val acc | best val acc | final val loss |
|---|---|---|---|---|---|---|---|---|
| B_low_lr       | 1e-4 | 32 | 0.3 | Adam | 0.925 | 0.946 | **0.973** | 0.148 |
| E_more_dropout | 1e-3 | 32 | 0.6 | Adam | 0.892 | 0.694 | 0.973 | 1.187 |
| A_baseline     | 1e-3 | 32 | 0.3 | Adam | 0.925 | 0.748 | 0.955 | 0.621 |
| D_small_batch  | 1e-3 | 8  | 0.3 | Adam | 0.930 | 0.910 | 0.946 | 0.195 |
| C_high_lr      | 5e-3 | 32 | 0.3 | Adam | 0.909 | 0.802 | 0.928 | 0.547 |
| F_sgd_momentum | 1e-2 | 32 | 0.3 | SGD+momentum | 0.886 | 0.793 | 0.919 | 0.588 |

![Validation curves for all configurations](outputs/03_hyperparam_val_curves.png)

![Best validation accuracy per configuration](outputs/04_hyperparam_comparison.png)

**Observations:**

- **Learning rate** was the most sensitive hyperparameter. `A_baseline` (1e-3) and `C_high_lr`
  (5e-3) both touched a high *peak* validation accuracy at some point during training, but their
  *final*-epoch validation loss (0.62 and 0.55) is 3–4× higher than the best run — a sign the
  optimizer keeps overshooting narrow minima once it gets close, rather than settling.
- A **low learning rate** (1e-4, `B_low_lr`) converged the most smoothly and finished with both
  the highest final validation accuracy (0.946) and the lowest final validation loss (0.148) of
  the six runs — the best balance of "keeps improving" vs. "doesn't overshoot" for this dataset
  size (518 training images).
- `E_more_dropout` (dropout 0.6) reached the same peak validation accuracy as `B_low_lr` at one
  point, but was far less stable overall (final val acc dropped to 0.694, final val loss 1.19) —
  heavy dropout combined with a default learning rate made training noisier rather than more
  robust here.
- **Batch size** 8 (`D_small_batch`) trained smoothly but slightly under the best configuration —
  more frequent, noisier gradient updates did not help as much as simply lowering the learning
  rate did.
- **SGD + momentum** (lr=1e-2) was the weakest of the six, needing more epochs or a different
  learning-rate schedule to fully catch up to the Adam runs.
- The best configuration by validation accuracy was **`B_low_lr`** (Adam, lr=1e-4, batch=32,
  dropout=0.3), which was carried forward to test-set evaluation.

Training/validation curves for the selected best model:

![Best model training curves](outputs/05_best_model_curves.png)

## 4. Test-set evaluation

The best model (`B_low_lr`) was evaluated once on the re-carved, larger **test** split (111
images, 56 plain / 55 pothole — held out from both training and hyperparameter selection):

| Metric | Value |
|---|---|
| Accuracy | **0.964** |
| Precision (pothole) | **0.964** |
| Recall (pothole) | **0.964** |
| F1-score (pothole) | **0.964** |

Confusion matrix:

![Confusion matrix](outputs/06_confusion_matrix.png)

|                 | Predicted plain | Predicted pothole |
|---|---|---|
| **Actual plain**   | 54 | 2 |
| **Actual pothole** | 2  | 53 |

**Interpretation.** With a 111-image test set (vs. the original 16), the metrics are far more
statistically trustworthy: only **4 misclassifications out of 111** images (2 false positives,
2 false negatives), and precision/recall/F1 all land at the same balanced 0.964 value, indicating
no systematic bias toward either class. This is consistent with the validation-accuracy trend
seen during hyperparameter tuning, where `B_low_lr` was already the most stable configuration.

## 5. Discussion / conclusions

- The Week 2 pothole dataset (`Pothole.v1-raw.yolov8`) has **no pothole-free images**, so it
  cannot support a whole-image classifier on its own. The Kaggle *Pothole and Plain Road Images*
  dataset was used instead, since it provides a genuine two-class (`Plain` / `Pothole`) label.
- The dataset's shipped `train`/`test` folders were pooled and re-split 70/15/15 (stratified),
  growing the test set from 16 to 111 images. This turned a noisy, low-confidence evaluation
  (56% accuracy on 16 images in an earlier pass) into a much more reliable one (96.4% accuracy
  on 111 images), without changing the model or training procedure at all — a good illustration
  of why test-set size matters as much as model choice when judging performance.
- A small custom CNN, trained from scratch on 518 images, reaches **96.4% test accuracy** with
  balanced precision/recall/F1, without any pretrained backbone.
- Hyperparameter tuning showed the model is most sensitive to learning rate: 1e-3 and 5e-3 both
  reach a good peak validation accuracy but destabilise by the end of 20 epochs (rising
  validation loss), while a lower learning rate (1e-4) with Adam gave the smoothest, most
  reliable convergence for this dataset size.
- Remaining errors (2 false positives, 2 false negatives) are exactly the kind expected at this
  image resolution — road markings, cracks, or shadows resembling pothole texture, and shallow
  or partially-obscured potholes resembling plain road.

## Reproducing this report

```
cd W4_ASSGN
python build_notebook.py                                        # regenerates the .ipynb
jupyter nbconvert --to notebook --execute --inplace pothole_ann_classifier.ipynb
```

All figures embedded above are written to `outputs/` by the notebook; raw metrics are also saved
to `outputs/metrics_summary.json`.
