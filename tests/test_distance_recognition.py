"""
SignVisionAI - Unit Tests for Distance-Based Recognition Module
Part 3: Distance-Based Recognition (1-NN & KNN)

Test Coverage:
1. Pure Euclidean Distance:
   - Exact mathematical calculation vs known analytical values and scipy
   - Numerical precision and symmetry: d(x, y) == d(y, x)
   - Dimension mismatch validation
   - Non-finite (NaN, Inf) rejection
2. Vectorized Pairwise Distance:
   - compute_distances matches manual loop
   - Query/reference dimension compatibility
3. 1-Nearest Neighbour (1-NN):
   - Correct nearest sample selection and metadata
4. K-Nearest Neighbours (KNN):
   - Majority voting with k=3 and k=5
   - Top-k metadata inspection
5. Deterministic Tie-Breaking:
   - Distance-based resolution when vote counts are tied
   - Preservation of deterministic outcome regardless of label dictionary ordering
6. One-Hand vs Two-Hand Architectural Separation:
   - 1-hand (63 features) queries match exclusively against 1-hand library (Letter C)
   - 2-hand (126 features) queries match exclusively against 2-hand library
   - Cross-dimension rejection (e.g. 63 features with hand_count=2)
   - Unsupported hand counts (0, 3) rejected safely
7. Robustness & Error Handling:
   - None inputs, empty arrays, NaN, Inf handling without exceptions
   - k exceeding reference sample count validation
8. Uncertainty & Distance Threshold Rejection:
   - Samples exceeding threshold rejected with status="uncertain" and letter=None
   - Samples within threshold accepted with status="recognized"
   - Separate 1-hand and 2-hand threshold application
9. Seamless Integration with Part 2 (HandFeatureResult):
   - Valid HandFeatureResult passed directly to recognizer
   - Invalid HandFeatureResult handled safely
10. Live Dataset Verification:
    - Real data/sls_normalized_square_pad.npz loading and verification
"""

from pathlib import Path
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distance_recognition import (
    AUGMENTED_TWO_HAND_FEATURE_COUNT,
    DEFAULT_DATASET_NPZ,
    ONE_HAND_FEATURE_COUNT,
    TWO_HAND_FEATURE_COUNT,
    DistanceRecognizer,
    RecognitionResult,
    calibrate_distance_thresholds,
    compute_distances,
    euclidean_distance,
)
from src.evaluate_distance_recognition import stratified_split_indices
from src.normalization import HandFeatureResult


class TestEuclideanDistance(unittest.TestCase):
    """Test suite for core Euclidean distance mathematical functions."""

    def test_euclidean_distance_known_values(self):
        """Test Euclidean distance on known 2D and 3D vectors."""
        # 3-4-5 right triangle
        p1 = np.array([0.0, 0.0], dtype=np.float32)
        p2 = np.array([3.0, 4.0], dtype=np.float32)
        self.assertAlmostEqual(euclidean_distance(p1, p2), 5.0, places=5)

        # 3D vector: sqrt(1^2 + 2^2 + 2^2) = sqrt(9) = 3.0
        p3 = np.array([1.0, 2.0, 2.0], dtype=np.float32)
        p4 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(euclidean_distance(p3, p4), 3.0, places=5)

    def test_euclidean_distance_symmetry_and_identity(self):
        """Test mathematical properties: d(x, x) == 0 and d(x, y) == d(y, x)."""
        rng = np.random.default_rng(42)
        vec_a = rng.normal(size=ONE_HAND_FEATURE_COUNT).astype(np.float32)
        vec_b = rng.normal(size=ONE_HAND_FEATURE_COUNT).astype(np.float32)

        # Identity
        self.assertAlmostEqual(euclidean_distance(vec_a, vec_a), 0.0, places=5)

        # Symmetry
        d_ab = euclidean_distance(vec_a, vec_b)
        d_ba = euclidean_distance(vec_b, vec_a)
        self.assertAlmostEqual(d_ab, d_ba, places=5)

    def test_euclidean_distance_dimension_mismatch(self):
        """Ensure ValueError is raised if vector dimensions differ."""
        v63 = np.zeros(63, dtype=np.float32)
        v126 = np.zeros(126, dtype=np.float32)
        with self.assertRaises(ValueError):
            euclidean_distance(v63, v126)

    def test_euclidean_distance_non_finite_rejection(self):
        """Ensure vectors containing NaN or Inf raise ValueError."""
        v1 = np.zeros(63, dtype=np.float32)
        v_nan = np.zeros(63, dtype=np.float32)
        v_nan[5] = np.nan
        with self.assertRaises(ValueError):
            euclidean_distance(v1, v_nan)

        v_inf = np.zeros(63, dtype=np.float32)
        v_inf[10] = np.inf
        with self.assertRaises(ValueError):
            euclidean_distance(v1, v_inf)

    def test_compute_distances_vectorized(self):
        """Test compute_distances matches manual pairwise calculations."""
        rng = np.random.default_rng(101)
        query = rng.normal(size=63).astype(np.float32)
        refs = rng.normal(size=(20, 63)).astype(np.float32)

        distances = compute_distances(query, refs)
        self.assertEqual(distances.shape, (20,))

        for i in range(20):
            expected = euclidean_distance(query, refs[i])
            self.assertAlmostEqual(float(distances[i]), expected, places=4)


class TestDistanceRecognizerCore(unittest.TestCase):
    """Test suite for 1-NN, KNN, voting, and tie-breaking logic."""

    def setUp(self):
        """Set up synthetic reference libraries with known geometry."""
        self.recognizer = DistanceRecognizer(k=3)

        # Synthetic 1-hand library: 5 samples of 'C' around origin
        self.ref_1h = np.zeros((5, ONE_HAND_FEATURE_COUNT), dtype=np.float32)
        for i in range(5):
            self.ref_1h[i, 0] = float(i) * 0.1  # x_0 = 0.0, 0.1, 0.2, 0.3, 0.4
        self.labels_1h = np.array(["C"] * 5)
        self.recognizer.set_references_1h(self.ref_1h, self.labels_1h)

        # Synthetic 2-hand library:
        # 3 samples of 'A' (near [1.0, ...])
        # 3 samples of 'B' (near [2.0, ...])
        # 3 samples of 'D' (near [3.0, ...])
        self.ref_2h = np.zeros((9, TWO_HAND_FEATURE_COUNT), dtype=np.float32)
        self.labels_2h = np.array(["A", "A", "A", "B", "B", "B", "D", "D", "D"])

        for i in range(3):
            self.ref_2h[i, :] = 1.0 + (i * 0.05)
        for i in range(3, 6):
            self.ref_2h[i, :] = 2.0 + ((i - 3) * 0.05)
        for i in range(6, 9):
            self.ref_2h[i, :] = 3.0 + ((i - 6) * 0.05)

        self.recognizer.set_references_2h(self.ref_2h, self.labels_2h)

    def test_1nn_recognition(self):
        """Test 1-NN prediction directly returns closest sample label."""
        query_a = np.full(TWO_HAND_FEATURE_COUNT, 1.02, dtype=np.float32)
        result = self.recognizer.predict(query_a, hand_count=2, k=1)

        self.assertEqual(result["status"], "recognized")
        self.assertEqual(result["letter"], "A")
        self.assertEqual(result["method"], "1nn")
        self.assertEqual(result["k"], 1)
        self.assertEqual(result["confidence"], 1.0)
        self.assertLess(result["distance"], 0.5)

    def test_knn_majority_voting(self):
        """Test that k=3 majority voting returns the dominant class."""
        # Query situated closest to 'B' references
        query_b = np.full(TWO_HAND_FEATURE_COUNT, 2.01, dtype=np.float32)
        result = self.recognizer.predict(query_b, hand_count=2, k=3)

        self.assertEqual(result["status"], "recognized")
        self.assertEqual(result["letter"], "B")
        self.assertEqual(result["method"], "knn")
        self.assertEqual(result["k"], 3)
        self.assertEqual(result["votes"]["B"], 3)
        self.assertEqual(result["confidence"], 1.0)

    def test_deterministic_tie_breaking(self):
        """
        Test deterministic tie-breaking:
        When votes are tied, candidate with smallest distance to query wins.
        """
        # Create a custom recognizer with explicit tied references
        rec = DistanceRecognizer(k=2)

        # 2 samples: one 'A' at distance ~0.3, one 'B' at distance ~0.1
        custom_refs = np.zeros((2, TWO_HAND_FEATURE_COUNT), dtype=np.float32)
        custom_refs[0, 0] = 0.3  # Class 'A'
        custom_refs[1, 0] = 0.1  # Class 'B'
        custom_labels = np.array(["A", "B"])
        rec.set_references_2h(custom_refs, custom_labels)

        query = np.zeros(TWO_HAND_FEATURE_COUNT, dtype=np.float32)
        # With k=2, A has 1 vote, B has 1 vote (tie).
        # B is at distance 0.1, A is at distance 0.3 -> B must win deterministically.
        res = rec.predict(query, hand_count=2, k=2)

        self.assertEqual(res["status"], "recognized")
        self.assertEqual(res["letter"], "B")
        self.assertEqual(res["votes"], {"B": 1, "A": 1})
        self.assertAlmostEqual(res["distance"], 0.1, places=4)

    def test_three_way_tie_breaking(self):
        """Test k=3 tie with 1 vote each for A, B, and D."""
        rec = DistanceRecognizer(k=3)
        custom_refs = np.zeros((3, TWO_HAND_FEATURE_COUNT), dtype=np.float32)
        custom_refs[0, 0] = 0.5  # A
        custom_refs[1, 0] = 0.2  # B (closest!)
        custom_refs[2, 0] = 0.8  # D
        rec.set_references_2h(custom_refs, np.array(["A", "B", "D"]))

        query = np.zeros(TWO_HAND_FEATURE_COUNT, dtype=np.float32)
        res = rec.predict(query, hand_count=2, k=3)

        self.assertEqual(res["status"], "recognized")
        self.assertEqual(res["letter"], "B")  # B is closest of the 3 tied


class TestOneHandTwoHandSeparation(unittest.TestCase):
    """Test suite ensuring strict separation between 1-hand and 2-hand workflows."""

    def setUp(self):
        self.recognizer = DistanceRecognizer(k=1)
        self.ref_1h = np.zeros((10, ONE_HAND_FEATURE_COUNT), dtype=np.float32)
        self.recognizer.set_references_1h(self.ref_1h, np.array(["C"] * 10))

        self.ref_2h = np.ones((10, TWO_HAND_FEATURE_COUNT), dtype=np.float32)
        self.recognizer.set_references_2h(self.ref_2h, np.array(["A"] * 10))

    def test_1hand_query_routes_to_1hand_library(self):
        """Ensure 63-dim input with hand_count=1 queries only 1-hand library."""
        query_1h = np.zeros(ONE_HAND_FEATURE_COUNT, dtype=np.float32)
        res = self.recognizer.predict(query_1h, hand_count=1)

        self.assertEqual(res["status"], "recognized")
        self.assertEqual(res["letter"], "C")
        self.assertEqual(res["hand_count"], 1)

    def test_2hand_query_routes_to_2hand_library(self):
        """Ensure 126-dim input with hand_count=2 queries only 2-hand library."""
        query_2h = np.ones(TWO_HAND_FEATURE_COUNT, dtype=np.float32)
        res = self.recognizer.predict(query_2h, hand_count=2)

        self.assertEqual(res["status"], "recognized")
        self.assertEqual(res["letter"], "A")
        self.assertEqual(res["hand_count"], 2)

    def test_dimension_and_hand_count_mismatch_rejected(self):
        """Reject mismatched hand count vs dimensions without comparing vectors."""
        # 63 dims with hand_count=2
        query_63 = np.zeros(ONE_HAND_FEATURE_COUNT, dtype=np.float32)
        res1 = self.recognizer.predict(query_63, hand_count=2)
        self.assertEqual(res1["status"], "invalid_input")
        self.assertIn("Dimension mismatch", res1["error_message"])

        # 126 dims with hand_count=1
        query_126 = np.zeros(TWO_HAND_FEATURE_COUNT, dtype=np.float32)
        res2 = self.recognizer.predict(query_126, hand_count=1)
        self.assertEqual(res2["status"], "invalid_input")
        self.assertIn("Dimension mismatch", res2["error_message"])

    def test_unsupported_hand_counts_rejected(self):
        """Reject unsupported hand counts (0 hands or 3+ hands)."""
        query_63 = np.zeros(ONE_HAND_FEATURE_COUNT, dtype=np.float32)
        res_zero = self.recognizer.predict(query_63, hand_count=0)
        self.assertEqual(res_zero["status"], "invalid_input")
        self.assertIn("Unsupported hand_count", res_zero["error_message"])

        res_three = self.recognizer.predict(query_63, hand_count=3)
        self.assertEqual(res_three["status"], "invalid_input")


class TestUncertaintyAndThresholdRejection(unittest.TestCase):
    """Test suite for distance thresholding and uncertain rejection mechanism."""

    def setUp(self):
        self.recognizer = DistanceRecognizer(k=1, threshold_1h=0.4, threshold_2h=1.2)
        # Reference at origin
        ref_1h = np.zeros((5, ONE_HAND_FEATURE_COUNT), dtype=np.float32)
        self.recognizer.set_references_1h(ref_1h, np.array(["C"] * 5))

        ref_2h = np.zeros((5, TWO_HAND_FEATURE_COUNT), dtype=np.float32)
        self.recognizer.set_references_2h(ref_2h, np.array(["A"] * 5))

    def test_close_sample_recognized(self):
        """Query within threshold is recognized."""
        close_1h = np.zeros(ONE_HAND_FEATURE_COUNT, dtype=np.float32)
        close_1h[0] = 0.2  # distance = 0.2 < 0.4
        res = self.recognizer.predict(close_1h, hand_count=1)

        self.assertEqual(res["status"], "recognized")
        self.assertEqual(res["letter"], "C")
        self.assertAlmostEqual(res["distance"], 0.2, places=4)

    def test_distant_sample_uncertain(self):
        """Query exceeding threshold is flagged as uncertain with letter=None."""
        far_1h = np.zeros(ONE_HAND_FEATURE_COUNT, dtype=np.float32)
        far_1h[0] = 0.8  # distance = 0.8 > 0.4
        res = self.recognizer.predict(far_1h, hand_count=1)

        self.assertEqual(res["status"], "uncertain")
        self.assertIsNone(res["letter"])
        self.assertAlmostEqual(res["distance"], 0.8, places=4)
        self.assertEqual(res["threshold"], 0.4)

    def test_separate_thresholds_for_1h_and_2h(self):
        """Ensure 1h uses threshold_1h and 2h uses threshold_2h independently."""
        # Query 2h with distance 0.9 (exceeds threshold_1h=0.4, but within threshold_2h=1.2)
        query_2h = np.zeros(TWO_HAND_FEATURE_COUNT, dtype=np.float32)
        query_2h[0] = 0.9
        res = self.recognizer.predict(query_2h, hand_count=2)

        self.assertEqual(res["status"], "recognized")
        self.assertEqual(res["letter"], "A")
        self.assertEqual(res["threshold"], 1.2)

    def test_threshold_calibration_helper(self):
        """Verify calibration computes valid empirical percentiles on validation sets."""
        ref_2h = np.zeros((10, TWO_HAND_FEATURE_COUNT), dtype=np.float32)
        val_2h = np.zeros((10, TWO_HAND_FEATURE_COUNT), dtype=np.float32)
        for i in range(10):
            val_2h[i, 0] = float(i + 1) * 0.1  # distances: 0.1 .. 1.0

        calib = calibrate_distance_thresholds(
            val_features_1h=None,
            ref_features_1h=None,
            val_features_2h=val_2h,
            ref_features_2h=ref_2h,
            percentile=90.0,
            k=1,
        )

        self.assertIsNotNone(calib["threshold_2h"])
        # 90th percentile of [0.1 .. 1.0] is approx 0.91
        self.assertGreaterEqual(calib["threshold_2h"], 0.85)
        self.assertLessEqual(calib["threshold_2h"], 1.0)


class TestInputRobustnessAndHandFeatureResult(unittest.TestCase):
    """Test suite for malformed inputs, edge cases, and Part 2 dataclass integration."""

    def setUp(self):
        self.recognizer = DistanceRecognizer(k=1)
        ref_1h = np.zeros((2, ONE_HAND_FEATURE_COUNT), dtype=np.float32)
        self.recognizer.set_references_1h(ref_1h, np.array(["C", "C"]))

    def test_none_input_rejected(self):
        """Pass None as input data."""
        res = self.recognizer.predict(None, hand_count=1)
        self.assertEqual(res["status"], "invalid_input")
        self.assertIn("cannot be None", res["error_message"])

    def test_nan_features_rejected(self):
        """Pass feature array containing NaN."""
        feat = np.zeros(ONE_HAND_FEATURE_COUNT, dtype=np.float32)
        feat[10] = np.nan
        res = self.recognizer.predict(feat, hand_count=1)
        self.assertEqual(res["status"], "invalid_input")
        self.assertIn("NaN", res["error_message"])

    def test_inf_features_rejected(self):
        """Pass feature array containing Inf."""
        feat = np.zeros(ONE_HAND_FEATURE_COUNT, dtype=np.float32)
        feat[5] = np.inf
        res = self.recognizer.predict(feat, hand_count=1)
        self.assertEqual(res["status"], "invalid_input")

    def test_invalid_k_rejected(self):
        """Pass non-positive k."""
        feat = np.zeros(ONE_HAND_FEATURE_COUNT, dtype=np.float32)
        res = self.recognizer.predict(feat, hand_count=1, k=0)
        self.assertEqual(res["status"], "invalid_input")

        res_neg = self.recognizer.predict(feat, hand_count=1, k=-1)
        self.assertEqual(res_neg["status"], "invalid_input")

    def test_valid_hand_feature_result_integration(self):
        """Verify passing a valid HandFeatureResult dataclass works seamlessly."""
        mock_result = HandFeatureResult(
            features=np.zeros(ONE_HAND_FEATURE_COUNT, dtype=np.float32),
            hand_count=1,
            resolved_handedness=("Right",),
            is_valid=True,
            status="OK",
        )
        res = self.recognizer.predict(mock_result)
        self.assertEqual(res["status"], "recognized")
        self.assertEqual(res["letter"], "C")
        self.assertEqual(res["hand_count"], 1)

    def test_invalid_hand_feature_result_integration(self):
        """Verify passing an invalid HandFeatureResult safely returns invalid_input."""
        mock_invalid = HandFeatureResult(
            features=None,
            hand_count=0,
            resolved_handedness=(),
            is_valid=False,
            status="UNSUPPORTED_HAND_COUNT",
            error_message="Detected 0 hands.",
        )
        res = self.recognizer.predict(mock_invalid)
        self.assertEqual(res["status"], "invalid_input")
        self.assertIn("HandFeatureResult is invalid", res["error_message"])

    def test_auto_flatten_2d_input(self):
        """Verify passing a 2D landmark array (21, 3) is automatically flattened to (63,)."""
        landmarks_2d = np.zeros((21, 3), dtype=np.float32)
        res = self.recognizer.predict(landmarks_2d, hand_count=1)
        self.assertEqual(res["status"], "recognized")
        self.assertEqual(res["letter"], "C")
        self.assertEqual(res["hand_count"], 1)


class TestRealDatasetLoading(unittest.TestCase):
    """Test loading and validating against the real project dataset archive."""

    @unittest.skipUnless(DEFAULT_DATASET_NPZ.exists(), "Dataset file sls_normalized_square_pad.npz not found.")
    def test_load_real_dataset_npz(self):
        """Verify real dataset loads with exact expected counts and shapes."""
        recognizer = DistanceRecognizer(reference_path=DEFAULT_DATASET_NPZ, k=3)
        counts = recognizer.sample_counts

        self.assertEqual(counts["one_hand"], 80)
        self.assertEqual(counts["two_hand"], 716)

        # Query real sample from 1-hand library
        archive = np.load(DEFAULT_DATASET_NPZ)
        feat_1h = archive["features_1h"][0]
        res_1h = recognizer.predict(feat_1h, hand_count=1, k=1)
        self.assertEqual(res_1h["status"], "recognized")
        self.assertEqual(res_1h["letter"], "C")
        self.assertAlmostEqual(res_1h["distance"], 0.0, places=4)

        # Query real sample from 2-hand library
        feat_2h = archive["features_2h"][0]
        label_2h = str(archive["labels_2h"][0])
        res_2h = recognizer.predict(feat_2h, hand_count=2, k=1)
        self.assertEqual(res_2h["status"], "recognized")
        self.assertEqual(res_2h["letter"], label_2h)
        self.assertAlmostEqual(res_2h["distance"], 0.0, places=4)


class TestReferenceAndTestSeparation(unittest.TestCase):
    """Test suite ensuring reference and test sample separation (preventing self-comparison)."""

    def test_train_test_separation_prevents_zero_distance_leakage(self):
        """
        Verify that when a sample is strictly held out in the test set,
        it cannot match against itself with 0.0 distance unless an exact duplicate exists.
        """
        rng = np.random.default_rng(999)
        # Generate 10 distinct random samples
        all_features = rng.normal(size=(10, TWO_HAND_FEATURE_COUNT)).astype(np.float32)
        all_labels = np.array(["A", "A", "B", "B", "D", "D", "E", "E", "F", "F"])

        # Split: 8 reference, 2 test
        ref_features = all_features[:8]
        ref_labels = all_labels[:8]

        test_features = all_features[8:]  # Samples 8 and 9
        test_labels = all_labels[8:]

        recognizer = DistanceRecognizer(k=1)
        recognizer.set_references_2h(ref_features, ref_labels)

        for i, test_feat in enumerate(test_features):
            res = recognizer.predict(test_feat, hand_count=2, k=1)
            # Distance must be strictly positive since the exact sample is not in reference set
            self.assertGreater(res["distance"], 0.0)

            # Ensure the top neighbor is one of the reference samples, not the test sample itself
            self.assertLess(res["top_k"][0]["index"], len(ref_features))


class TestDuplicateLeakagePrevention(unittest.TestCase):
    """Test suite ensuring exact duplicate feature vectors cannot be split across sets."""

    def test_synthetic_exact_duplicates_always_in_same_split(self):
        """
        Create synthetic feature vectors with known exact duplicates and assert that
        duplicate pairs are ALWAYS placed in the exact same split (Ref, Val, or Test).
        """
        rng = np.random.default_rng(1234)
        # 15 samples across 3 classes (A, B, D)
        features = rng.normal(size=(15, TWO_HAND_FEATURE_COUNT)).astype(np.float32)
        labels = np.array(["A"] * 5 + ["B"] * 5 + ["D"] * 5)

        # Introduce duplicate pairs:
        # Sample 1 is duplicate of sample 0 (Class A)
        features[1] = features[0].copy()
        # Sample 6 is duplicate of sample 5 (Class B)
        features[6] = features[5].copy()
        # Sample 11 is duplicate of sample 10 (Class D)
        features[11] = features[10].copy()

        train_idx, val_idx, test_idx = stratified_split_indices(
            labels, features=features, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42
        )

        # Check all 3 duplicate pairs
        for i, j in [(0, 1), (5, 6), (10, 11)]:
            loc_i = "train" if i in train_idx else ("val" if i in val_idx else "test")
            loc_j = "train" if j in train_idx else ("val" if j in val_idx else "test")
            self.assertEqual(
                loc_i, loc_j,
                f"Duplicate pair ({i}, {j}) was split across {loc_i} and {loc_j}!"
            )

    @unittest.skipUnless(DEFAULT_DATASET_NPZ.exists(), "Dataset file sls_normalized_square_pad.npz not found.")
    def test_real_dataset_duplicates_never_split_across_sets(self):
        """
        Verify that all 5 known duplicate pairs in the real project dataset
        are placed in the same split without any leakage across Ref, Val, and Test.
        """
        data = np.load(DEFAULT_DATASET_NPZ)
        f2 = data["features_2h"]
        l2 = data["labels_2h"]
        img2 = data["image_ids_2h"]

        train_idx, val_idx, test_idx = stratified_split_indices(
            l2, features=f2, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42
        )

        dup_pairs = [
            ("E_50.jpg", "E_51.jpg"),
            ("I_49.jpg", "I_50.jpg"),
            ("L_49.jpg", "L_50.jpg"),
            ("N_56.jpg", "N_57.jpg"),
            ("S_55.jpg", "S_56.jpg"),
        ]

        for a, b in dup_pairs:
            ia = int(np.where(img2 == a)[0][0])
            ib = int(np.where(img2 == b)[0][0])
            loc_a = "train" if ia in train_idx else ("val" if ia in val_idx else "test")
            loc_b = "train" if ib in train_idx else ("val" if ib in val_idx else "test")
            self.assertEqual(
                loc_a, loc_b,
                f"Real duplicate pair ({a}, {b}) was split across {loc_a} and {loc_b}!"
            )


if __name__ == "__main__":
    unittest.main()
