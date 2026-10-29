# Task 4 — Confidence-Aware Gesture Recognition

> **SignVisionAI | Integration & Handover Guide**  
> **Developer:** Nimasha  
> **Branch:** `feature/nimasha-confidence-estimation`  
> **Status:** Implemented and unit-tested; application integration and real-data validation pending

## Overview

This module improves the **reliability of gesture recognition decisions**, rather than artificially increasing a confidence percentage. It evaluates the best class distance, separation from the runner-up class, and consistency across recent frames.

> **Note:** `L` is only an example. The estimator is **not hardcoded to any particular letter**; it works with the labels supplied by the recognizer.

### Recognition outcomes

| Outcome | Example displayed | Meaning |
|:--|:--|:--|
| **Strong match** | `L` | The candidate satisfies the configured strong-match and stability rules. |
| **Likely match** | `L?` | `L` is the leading candidate, but the evidence is not strong enough for acceptance. |
| **Uncertain gesture** | `Uncertain gesture` | The scores do not reliably distinguish the candidates or indicate a poor match. |
| **No hand detected** | `No hand detected` | No usable hand input is present. |

## How confidence is evaluated

1. **Best-class distance:** How close the observed hand features are to the nearest class. Lower is better.
2. **Top-two margin:** `second_best_distance - best_distance`. A larger positive gap separates the best candidate from the next best.
3. **Temporal stability:** The winning label must remain consistent across recent frames and for sufficient time.
4. **Window-level evidence:** The median distance and margin across matching frames prevent one unusually good frame from immediately becoming a strong match.

These are *decision criteria*, not statistically calibrated probability guarantees.

## Included files

| File | Purpose |
|:--|:--|
| `confidence_estimator.py` | Confidence-aware decision logic and probability-to-score conversion helper. |
| `tests/test_confidence_estimator.py` | Automated tests of the decision logic and edge cases. |
| `TASK4_INTEGRATION.md` | Integration notes and handover information. |

## Integration quick start

### 1. Import and initialize

Initialize **one estimator per recognition stream** and reuse it across frames:

```python
from confidence_estimator import ConfidenceEstimator

estimator = ConfidenceEstimator()
```

### 2. Pass per-class distances on each processed frame

The matcher must supply a mapping containing **distances for at least two classes**, not just the winning label:

```python
class_distances = {
    "L": 0.20,
    "T": 0.80,
    "A": 1.10,
}  # Illustrative values only — not measured results.

decision = estimator.update(
    distances=class_distances,
    hands_present=True,
    now=frame_timestamp,  # Monotonic time in seconds, if supplied.
)
```

**Important:** Smaller distance means a better match. The example values do **not** establish recommended thresholds.

### 3. Handle frames without a usable hand

```python
decision = estimator.update(
    distances=None,
    hands_present=False,
    now=frame_timestamp,
)
```

### 4. Connect the result to the UI and existing hold-to-add logic

```python
shown_text = decision.display

if decision.accepted:
    # Eligible for the application's existing hold-to-add checks.
    # Do not insert the letter automatically at this point.
    pass
```

- Use `decision.display` for the displayed recognition state.
- Use `decision.accepted` to **gate eligibility** for letter insertion; retain the app's hold-time and repeat-prevention behavior.
- A `likely` or `uncertain` output must not be accepted as a confirmed letter.
- Reset state on stream stop/restart, a new recognition session, or an intentional discontinuity:

```python
estimator.reset()
```

### 5. If using an ML classifier instead of matcher distances

A classifier may return probabilities rather than distances. **Do not pass raw probabilities directly into the distance-based estimator.** Convert them to score values first:

```python
from confidence_estimator import probabilities_to_scores

scores = probabilities_to_scores(class_probabilities)
ml_decision = ml_estimator.update(scores, hands_present=True)
```

Use a **separate estimator and separately validated thresholds** for probability-derived scores. Converting to `-log(p)` scores is **not probability calibration**. Ensure the mapping keys are the actual class labels.

## Unit tests

Run from the project repository root:

```bash
python -m pytest tests/test_confidence_estimator.py -v
```

**Reported local result:** **69 passed, 1 skipped, 0 failed** (Python 3.13.5).

Tests cover score ordering, margins, all four recognition states, invalid inputs, temporal stability, timer/history resets, and the probability conversion helper. This result verifies the isolated logic; it does **not** establish real-world recognition accuracy or successful GUI integration.



## Integration handover checklist

- [ ] Import `ConfidenceEstimator` into the active recognition pipeline.
- [ ] Supply **per-class distance values** from the actual matcher, or suitable converted scores with separate thresholds for an ML model.
- [ ] Call `update()` for every processed frame, including no-hand cases.
- [ ] Show `decision.display` in the UI.
- [ ] Gate insertion using `decision.accepted` while retaining hold-to-add and repeat-prevention checks.
- [ ] Reset estimator state when starting or restarting recognition.
- [ ] Validate the default thresholds with held-out real data and test temporal behavior with sequences.
- [ ] Confirm compatibility with the final application and recognition model.

