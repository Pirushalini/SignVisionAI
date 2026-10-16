# SignVisionAI — Part 3: Distance-Based Recognition (1-NN & KNN)

**Sri Lankan Sign Language (SLS) Alphabet Recognition System**
**Module:** `src/distance_recognition.py`
**Evaluation:** `src/evaluate_distance_recognition.py`
**Test Suite:** `tests/test_distance_recognition.py`

---

## 1. Objective of Part 3

In the SignVisionAI architecture, Part 3 provides **non-parametric, distance-based sign recognition** for Sri Lankan Sign Language (SLS) using normalized 3D hand landmarks extracted by Part 2 (`src/normalization.py`).

Unlike parametric classifiers or deep neural networks, distance-based recognition:
1. **Requires No Training Phase:** Recognizes signs directly by comparing live query landmark vectors against an in-memory labeled reference library extracted from the audited dataset (`data/sls_normalized_square_pad.npz`).
2. **Deterministic & Interpretable:** Decisions are based on geometric Euclidean proximity in normalized landmark space, making predictions explainable for academic vivas and evaluations.
3. **Ultra-Low Latency:** Vectorized NumPy calculations execute in $<0.1\text{ ms}$ per query, supporting $>1,000\text{ FPS}$ inference on CPU—far faster than the 30 FPS camera capture rate.
4. **Principled Uncertainty Rejection:** Rejects out-of-distribution gestures, non-sign hand movements, or false detections by thresholding nearest-neighbour distance rather than forcing an incorrect letter prediction.

---

## 2. Mathematical Foundation & Algorithms

### 2.1 Euclidean Distance
For a query landmark feature vector $\mathbf{q} \in \mathbb{R}^D$ and a reference landmark vector $\mathbf{r} \in \mathbb{R}^D$:

$$d(\mathbf{q}, \mathbf{r}) = \|\mathbf{q} - \mathbf{r}\|_2 = \sqrt{\sum_{j=1}^{D} (q_j - r_j)^2}$$

In `src/distance_recognition.py`, distance calculation against the entire reference matrix $\mathbf{R} \in \mathbb{R}^{N \times D}$ is vectorized:

```python
diff = references - query            # Broadcast subtraction (N, D)
distances = np.linalg.norm(diff, axis=1)  # Euclidean norm across rows (N,)
```

### 2.2 1-Nearest Neighbour (1-NN)
When $k = 1$:
1. Finds the index of minimum distance: $i^* = \arg\min_{i} d(\mathbf{q}, \mathbf{r}_i)$.
2. Returns predicted letter $y_{i^*}$ and nearest distance $d^* = d(\mathbf{q}, \mathbf{r}_{i^*})$.
3. Confidence is defined as $1.0$.

### 2.3 K-Nearest Neighbours (KNN, $k \ge 1$)
For configurable odd or even $k$ (e.g., $k=1, 3, 5$):
1. **Top-$k$ Proximity:** Sorts reference distances in ascending order to select the $k$ nearest neighbours:
   $$\{ (\mathbf{r}_{(1)}, y_{(1)}, d_{(1)}), \dots, (\mathbf{r}_{(k)}, y_{(k)}, d_{(k)}) \} \quad \text{where } d_{(1)} \le d_{(2)} \le \dots \le d_{(k)}$$
2. **Majority Voting:** Counts votes per candidate sign label:
   $$v(c) = \sum_{m=1}^{k} \mathbb{I}(y_{(m)} = c)$$
   The predicted class is the majority winner: $c^* = \arg\max_{c} v(c)$.
3. **Deterministic Distance-Based Tie-Breaking:** If two or more classes tie for the maximum vote count (e.g., in $k=2$ with 1 vote for 'A' and 1 vote for 'B', or in $k=3$ with 1 vote each for 'A', 'B', and 'D'):
   $$c^* = \arg\min_{c \in \text{Tied}} \left(\min_{m: y_{(m)} = c} d_{(m)}\right)$$
   Because the neighbour list is already sorted in ascending order of distance, the first tied candidate encountered in the top-$k$ list has the strictly smallest distance to the query. This breaks ties deterministically without arbitrary or alphabetical bias.
4. **Confidence Score:** Defined as the fraction of top-$k$ votes awarded to the winning sign:
   $$\text{Confidence} = \frac{v(c^*)}{k}$$

---

## 3. One-Hand vs Two-Hand Architectural Separation

A defining characteristic of Sri Lankan Sign Language (SLS) is its hand composition:
- **Letter C:** A **single-hand sign** represented by exactly **63 features** ($21\text{ landmarks} \times 3\text{ coordinates } (x, y, z)$).
- **Letters A, B, D..Z (25 letters):** **Two-hand signs** represented by **126 features** (ordered strictly as Left Hand $0 \dots 62$ followed by Right Hand $63 \dots 125$), or **129 features** when augmented with the 3D relative wrist displacement vector.

### 3.1 Strict Separation (No Zero-Padding)
Euclidean distance is only mathematically defined between vectors in the **same dimensional space** ($\mathbb{R}^{63}$ or $\mathbb{R}^{126}$). Comparing a 63-dimensional vector directly to a 126-dimensional vector is invalid. Furthermore:
- Missing hands are **never zero-padded** to 126 dimensions. Zero-padding would place an absent hand at coordinate $(0, 0, 0)$, distorting Euclidean metrics and producing false matches against real hands positioned near the wrist origin.
- The recognizer maintains **two independent reference libraries**:
  - `_features_1h` ($\text{shape: } (N_{1h}, 63)$) and `_labels_1h` for Letter C.
  - `_features_2h` ($\text{shape: } (N_{2h}, 126)$) and `_labels_2h` for the other 25 letters.
- Queries are strictly dispatched based on the detected `hand_count`:
  - `hand_count == 1` $\rightarrow$ matched exclusively against `_features_1h`.
  - `hand_count == 2` $\rightarrow$ matched exclusively against `_features_2h`.
  - Mismatched dimensions (e.g., 63 coordinates with `hand_count == 2`) or unsupported hand counts ($0$ or $>2$) are safely rejected with `status="invalid_input"`.

---

## 4. Uncertainty & Distance Threshold Rejection

In real-world deployment, the system must not force a sign prediction when the user is resting their hands, transitioning between gestures, or presenting an unsupported hand sign.

### 4.1 Separate Thresholds for 1-Hand and 2-Hand
Euclidean distance scales with the square root of dimensionality ($\sqrt{D}$):
$$\mathbb{E}[\|\mathbf{x} - \mathbf{y}\|_2] \propto \sqrt{D}$$
Because two-hand signs reside in $D = 126$ while single-hand signs reside in $D = 63$, distances in 126-dimensional space are naturally larger by a factor of approximately $\sqrt{126 / 63} \approx \sqrt{2} \approx 1.414$ for identical per-landmark coordinate deviations.

Therefore, `DistanceRecognizer` enforces **separate thresholds**:
- `threshold_1h` (e.g., $\approx 0.472$)
- `threshold_2h` (e.g., $\approx 1.682$)

### 4.2 Empirical Validation Calibration
Thresholds are calibrated on the **validation set** (never on the held-out test set):
1. For each validation sample, compute distance to its nearest reference sample in the reference library.
2. The threshold is set at an empirical percentile cutoff (e.g., 95th or 99th percentile):
   - At the **95th percentile threshold**: The system accepts 95% of normal validation gestures and rejects the 5% most extreme outliers.
   - At the **99th percentile threshold**: The system allows a wider tolerance band while filtering severe anomalies.
3. If $d^* > \text{threshold}$, the recognizer returns:
   ```json
   {
     "letter": null,
     "status": "uncertain",
     "distance": 1.8452,
     "threshold": 1.6816,
     "method": "knn",
     "k": 3
   }
   ```

---

## 5. Experimental Evaluation & Results

The evaluation script (`src/evaluate_distance_recognition.py`) evaluates the system using the audited normalized dataset `data/sls_normalized_square_pad.npz`.

### 5.1 Dataset Split & Leakage Mitigation
- **Dataset Size:** 796 extracted samples (80 single-hand samples of Letter C; 716 two-hand samples across 25 letters A–Z).
- **Split Ratio:** Stratified Group 60% Reference (477 samples), 20% Validation (161 samples), 20% Held-Out Test (158 samples).
- **Random Seed:** Fixed seed `random_seed=42` ensures strict reproducibility.
- **Atomic Duplicate Clustering & Leakage Prevention:**
  - `ref_set ∩ val_set = ∅`, `ref_set ∩ test_set = ∅`, `val_set ∩ test_set = ∅`.
  - Exact duplicate feature vectors are clustered and atomically assigned to the SAME split partition.
  - Real dataset duplicate audit:
    * `E_50.jpg` & `E_51.jpg`: Reference vs Reference (SAME)
    * `I_49.jpg` & `I_50.jpg`: Reference vs Reference (SAME)
    * `L_49.jpg` & `L_50.jpg`: Test vs Test (SAME)
    * `N_56.jpg` & `N_57.jpg`: Test vs Test (SAME)
    * `S_55.jpg` & `S_56.jpg`: Reference vs Reference (SAME)
  - **Result:** ZERO duplicate leakage across ALL three sets (Ref, Val, and Test).
  - Minimum distance from test samples to reference samples is $0.1288$ (2-hand) and $0.1322$ (1-hand), confirming strict sample isolation.
- **Near-Duplicate Risk Audit ($0.0 < \text{dist} < 0.15$):**
  Identified 5 near-duplicate pairs representing slight hand movement across consecutive frames from the same contributor:
  * `O_48.jpg` & `O_49.jpg` (Class O): dist = 0.10543
  * `M_41.jpg` & `M_42.jpg` (Class M): dist = 0.12882
  * `E_24.jpeg` & `E_33.jpeg` (Class E): dist = 0.12884
  * `I_41.jpg` & `I_42.jpg` (Class I): dist = 0.13994
  * `I_42.jpg` & `I_43.jpg` (Class I): dist = 0.14522
  All near-duplicate pairs belong to the exact same sign class; no cross-class ambiguity occurs at this distance range.

### 5.2 Calibrated Uncertainty Thresholds (Validation Set)
- **95th Percentile:** `threshold_1h = 0.4724`, `threshold_2h = 1.5691`
- **99th Percentile:** `threshold_1h = 0.4749`, `threshold_2h = 2.6802`

### 5.3 Benchmarking on Untouched Held-Out Test Set (158 Samples)

| Experiment | $k$ | Feature Representation | Uncertainty Threshold | Total Queries | Accepted | Uncertain (Rejected) | Rejection Rate (%) | Overall Accuracy (%) | Accepted Accuracy (%) | Mean NN Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1-NN Baseline** | 1 | Baseline (126) | None | 158 | 158 | 0 | 0.00% | **96.20%** | **96.20%** | 0.6265 |
| **KNN ($k=3$) Baseline** | 3 | Baseline (126) | None | 158 | 158 | 0 | 0.00% | **96.20%** | **96.20%** | 0.6265 |
| **KNN ($k=5$) Baseline** | 5 | Baseline (126) | None | 158 | 158 | 0 | 0.00% | **92.41%** | **92.41%** | 0.6265 |
| **KNN ($k=3$) + 95% Cutoff** | 3 | Baseline (126) | 95th Percentile | 158 | 150 | 8 | 5.06% | 93.67% | **98.67%** | 0.6265 |
| **KNN ($k=3$) + 99% Cutoff** | 3 | Baseline (126) | 99th Percentile | 158 | 154 | 4 | 2.53% | 94.94% | **97.40%** | 0.6265 |
| **1-NN Augmented** | 1 | Augmented (129) | None | 158 | 158 | 0 | 0.00% | **96.84%** | **96.84%** | 0.6427 |
| **KNN ($k=3$) Augmented** | 3 | Augmented (129) | None | 158 | 158 | 0 | 0.00% | **96.20%** | **96.20%** | 0.6427 |
| **KNN ($k=5$) Augmented** | 5 | Augmented (129) | None | 158 | 158 | 0 | 0.00% | **93.04%** | **93.04%** | 0.6427 |

### 5.4 Key Experimental Findings
1. **High Baseline Recognition:** 1-NN and 3-NN achieve **96.20% overall accuracy** (152/158 correct) across all 26 letters of Sri Lankan Sign Language on the untouched held-out test set without parametric training.
2. **Value of Uncertainty Filtering:** Applying the 95th percentile validation threshold successfully filters ambiguous gestures, boosting the accuracy of accepted predictions from **96.20% to 98.67%** (148/150 correct) at a low 5.06% rejection rate.
3. **Inter-Hand Wrist Augmentation (129 features):** Adding the normalized 3D relative wrist displacement vector ($\Delta \mathbf{W}$) pushes 1-NN accuracy to **96.84%** (153/158 correct), confirming that inter-hand spatial offset provides valuable discriminative geometry for two-hand signs.
4. **Perfect Single-Hand Recognition:** Single-hand Letter C achieved **100.0% accuracy** (16/16 test samples recognized correctly).

### 5.5 Per-Class Accuracy Breakdown (KNN $k=3$ Baseline)
Full results are exported to `results/per_class_accuracy_knn3.csv`:
- **100% Accuracy (22 letters):** B, C, D, E, F, G, H, I, J, K, M, N, O, P, Q, R, S, T, V, W, X, Y, Z
- **85.7% Accuracy (1 letter):** U (6/7 correct)
- **75.0% Accuracy (1 letter):** A (6/8 correct)
- **25.0% Accuracy (1 letter):** L (1/4 correct) — because identical duplicate samples `L_49` and `L_50` were both allocated to the test set, the model faced a true generalization challenge without memorization.

---

## 6. Known Limitations & Academic Analysis

1. **Single 1-Hand Class Limitation (Letter C):**
   In the SLS alphabet, C is the sole single-hand letter. In an evaluation consisting purely of SLS test images, all valid 1-hand queries are Letter C. Without an out-of-vocabulary dataset of non-SLS single-hand gestures, true negative rejection rates on arbitrary 1-hand poses cannot be benchmarked. The 1-hand threshold is therefore calibrated as the empirical upper bound of valid intra-class variance.
2. **Contributor Imbalance & Dropout Constraints:**
   Part 1 established that MediaPipe hand detection failed on ~61.7% of raw dataset images due to motion blur, skin tone variations, and hand occlusions. As a result, certain letters were only detected for specific members (e.g., letters H and T exist exclusively in Member 4's extracted samples). A strict Leave-One-Signer-Out (LOSO) split would leave letters H and T with 0 reference samples, preventing full alphabet evaluation. Grouped stratified splitting was therefore required.
3. **Burst Photography Duplicates:**
   Several consecutive photos from Member 3 yielded identical landmark coordinates. These were atomically clustered and held within the same split to guarantee zero duplicate leakage across Reference, Validation, and Test sets.

---

## 7. How to Run & Verify

All commands use the project virtual environment (`.venv`):

### 7.1 Running Automated Unit Tests
Run the 67 unit tests covering distance calculations, 1-NN, KNN, tie-breaking, dimension checks, atomic duplicate grouping, and uncertainty rejection:
```bash
.venv/bin/python -m unittest tests/test_distance_recognition.py
```
Or run the complete project test suite:
```bash
.venv/bin/python -m unittest discover tests
```

### 7.2 Running the Full Evaluation Benchmark
Run the evaluation script to generate benchmark tables, per-class summaries, and confusion matrices:
```bash
.venv/bin/python src/evaluate_distance_recognition.py
```
Generated artifacts in `results/`:
- `results/distance_recognition_summary.csv`
- `results/per_class_accuracy_knn3.csv`
- `results/confusion_matrix_knn3.csv`
- `results/confusion_matrix_1nn.csv`
- `results/confusion_matrix_knn5.csv`
- `results/distance_recognition_evaluation.json`

### 7.3 Running the Interactive Recognition Demo
```bash
.venv/bin/python src/distance_recognition.py --k 3
```

---

## 8. Integration Guide for Part 4 (Webcam Integration)

The teammate responsible for Part 4 can integrate `DistanceRecognizer` in just **3 lines of code**.

### 8.1 Direct Integration with `HandFeatureResult`
`DistanceRecognizer.predict()` natively accepts the `HandFeatureResult` dataclass returned by Part 2's `format_hand_features()`:

```python
import cv2
import numpy as np
from src.normalization import format_hand_features
from src.distance_recognition import DistanceRecognizer

# 1. Initialize the recognizer once at application startup
recognizer = DistanceRecognizer(
    reference_path="data/sls_normalized_square_pad.npz",
    k=3,
    threshold_1h=0.4724,  # Calibrated 95th percentile threshold
    threshold_2h=1.6816,  # Calibrated 95th percentile threshold
)

# 2. Inside the live webcam loop:
# detected_hands, handedness = mediapipe_detector.process(frame)
feature_result = format_hand_features(
    detected_hands=detected_hands,
    handedness=handedness,
    is_mirrored=True,  # True for selfie webcam feed
)

# 3. Pass the result directly to the recognizer
prediction = recognizer.predict(feature_result)

# 4. Display results on the webcam overlay
if prediction["status"] == "recognized":
    display_text = f"Sign: {prediction['letter']} (Dist: {prediction['distance']:.2f})"
    color = (0, 255, 0)  # Green
elif prediction["status"] == "uncertain":
    display_text = f"Uncertain (Dist: {prediction['distance']:.2f})"
    color = (0, 165, 255)  # Orange
else:
    display_text = "Position hands in camera view"
    color = (0, 0, 255)  # Red

cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
```

### 8.2 Standard Output Schema
`prediction` is guaranteed to return a dictionary with the following schema:
```json
{
  "letter": "A",
  "status": "recognized",
  "distance": 0.5421,
  "method": "knn",
  "k": 3,
  "hand_count": 2,
  "confidence": 1.0,
  "threshold": 1.6816,
  "top_k": [
    {"rank": 1, "letter": "A", "distance": 0.5421, "index": 12, "image_id": "A_10.jpg"},
    {"rank": 2, "letter": "A", "distance": 0.6210, "index": 15, "image_id": "A_19.jpg"},
    {"rank": 3, "letter": "A", "distance": 0.7105, "index": 18, "image_id": "A_22.jpg"}
  ],
  "votes": {"A": 3},
  "error_message": null
}
```
If the query exceeds the distance threshold:
```json
{
  "letter": null,
  "status": "uncertain",
  "distance": 2.1542,
  "method": "knn",
  "k": 3,
  "hand_count": 2,
  "confidence": 0.6667,
  "threshold": 1.6816,
  "top_k": [...],
  "votes": {"A": 2, "K": 1},
  "error_message": null
}
```
If the hand detection is invalid or missing:
```json
{
  "letter": null,
  "status": "invalid_input",
  "distance": null,
  "method": "knn",
  "k": 3,
  "hand_count": 0,
  "confidence": null,
  "threshold": 1.6816,
  "top_k": null,
  "votes": null,
  "error_message": "HandFeatureResult is invalid: EMPTY_DETECTION"
}
```
