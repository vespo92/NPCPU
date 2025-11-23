"""
Confidence Calibration System for ORACLE-Z Metacognitive Agent

Implements uncertainty estimation and confidence calibration that enables
the system to accurately assess its own confidence levels and provide
well-calibrated probability estimates.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from collections import defaultdict


class ConfidenceLevel(Enum):
    """Categorical confidence levels"""
    VERY_LOW = "very_low"      # < 0.2
    LOW = "low"                # 0.2-0.4
    MODERATE = "moderate"      # 0.4-0.6
    HIGH = "high"              # 0.6-0.8
    VERY_HIGH = "very_high"    # > 0.8


class UncertaintyType(Enum):
    """Types of uncertainty"""
    ALEATORIC = "aleatoric"        # Irreducible, inherent randomness
    EPISTEMIC = "epistemic"        # Reducible, due to lack of knowledge
    MODEL = "model"                # Uncertainty in the model itself
    DISTRIBUTIONAL = "distributional"  # Data distribution shift


@dataclass
class ConfidenceEstimate:
    """A confidence estimate with uncertainty bounds"""
    value: float  # Point estimate
    lower_bound: float
    upper_bound: float
    uncertainty_type: UncertaintyType = UncertaintyType.EPISTEMIC
    calibrated: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def uncertainty(self) -> float:
        """Get uncertainty as range width"""
        return self.upper_bound - self.lower_bound

    @property
    def level(self) -> ConfidenceLevel:
        """Get categorical confidence level"""
        if self.value < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif self.value < 0.4:
            return ConfidenceLevel.LOW
        elif self.value < 0.6:
            return ConfidenceLevel.MODERATE
        elif self.value < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH


@dataclass
class CalibrationSample:
    """A sample for calibration tracking"""
    predicted_confidence: float
    actual_outcome: bool  # True if prediction was correct
    domain: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CalibrationBin:
    """Bin for tracking calibration statistics"""
    bin_lower: float
    bin_upper: float
    predictions: List[float] = field(default_factory=list)
    outcomes: List[bool] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.predictions)

    @property
    def avg_confidence(self) -> float:
        if not self.predictions:
            return 0.0
        return np.mean(self.predictions)

    @property
    def accuracy(self) -> float:
        if not self.outcomes:
            return 0.0
        return np.mean(self.outcomes)

    @property
    def calibration_error(self) -> float:
        """Difference between average confidence and actual accuracy"""
        if not self.predictions:
            return 0.0
        return abs(self.avg_confidence - self.accuracy)


class ConfidenceCalibrator:
    """
    Core confidence calibration system for ORACLE-Z.

    Provides:
    - Uncertainty estimation for predictions
    - Confidence calibration using historical data
    - Expected Calibration Error (ECE) computation
    - Domain-specific calibration
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

        # Calibration bins
        self.calibration_bins: List[CalibrationBin] = []
        self._initialize_bins()

        # Domain-specific calibration
        self.domain_calibrators: Dict[str, 'DomainCalibrator'] = {}

        # Calibration history
        self.samples: List[CalibrationSample] = []
        self.max_samples = 10000

        # Calibration parameters (learned)
        self.temperature: float = 1.0  # Temperature scaling
        self.bias: float = 0.0
        self.scale: float = 1.0

        # Uncertainty estimation parameters
        self.ensemble_size: int = 5
        self.dropout_rate: float = 0.1

    def _initialize_bins(self):
        """Initialize calibration bins"""
        self.calibration_bins = []
        bin_width = 1.0 / self.n_bins

        for i in range(self.n_bins):
            lower = i * bin_width
            upper = (i + 1) * bin_width
            self.calibration_bins.append(CalibrationBin(
                bin_lower=lower,
                bin_upper=upper
            ))

    def estimate_confidence(self,
                           raw_score: float,
                           domain: Optional[str] = None,
                           use_calibration: bool = True) -> ConfidenceEstimate:
        """
        Estimate confidence with uncertainty bounds.

        Args:
            raw_score: Raw confidence/probability score (0-1)
            domain: Optional domain for domain-specific calibration
            use_calibration: Whether to apply calibration

        Returns:
            Calibrated confidence estimate with uncertainty bounds
        """
        raw_score = np.clip(raw_score, 0.0, 1.0)

        # Apply calibration if available
        if use_calibration:
            calibrated_score = self._calibrate_score(raw_score, domain)
        else:
            calibrated_score = raw_score

        # Estimate uncertainty
        uncertainty = self._estimate_uncertainty(raw_score, domain)

        # Calculate bounds
        lower = max(0.0, calibrated_score - uncertainty / 2)
        upper = min(1.0, calibrated_score + uncertainty / 2)

        return ConfidenceEstimate(
            value=calibrated_score,
            lower_bound=lower,
            upper_bound=upper,
            uncertainty_type=self._determine_uncertainty_type(raw_score, domain),
            calibrated=use_calibration
        )

    def _calibrate_score(self, raw_score: float, domain: Optional[str] = None) -> float:
        """Apply calibration to raw score"""
        # Domain-specific calibration if available
        if domain and domain in self.domain_calibrators:
            return self.domain_calibrators[domain].calibrate(raw_score)

        # Global temperature scaling
        logit = np.log(raw_score / (1 - raw_score + 1e-10) + 1e-10)
        scaled_logit = logit / self.temperature
        calibrated = 1 / (1 + np.exp(-scaled_logit))

        # Apply bias and scale
        calibrated = self.scale * calibrated + self.bias

        return np.clip(calibrated, 0.0, 1.0)

    def _estimate_uncertainty(self, raw_score: float, domain: Optional[str] = None) -> float:
        """Estimate uncertainty for a prediction"""
        # Base uncertainty from score extremity
        # Scores near 0.5 have higher aleatoric uncertainty
        aleatoric = 0.2 * (1 - 4 * (raw_score - 0.5) ** 2)

        # Epistemic uncertainty from calibration history
        bin_idx = self._get_bin_index(raw_score)
        bin_data = self.calibration_bins[bin_idx]

        if bin_data.count > 10:
            epistemic = bin_data.calibration_error
        else:
            epistemic = 0.2  # High uncertainty when few samples

        # Domain-specific uncertainty
        if domain and domain in self.domain_calibrators:
            domain_uncertainty = self.domain_calibrators[domain].get_uncertainty()
        else:
            domain_uncertainty = 0.1

        # Combine uncertainties
        total_uncertainty = np.sqrt(
            aleatoric ** 2 +
            epistemic ** 2 +
            domain_uncertainty ** 2
        )

        return np.clip(total_uncertainty, 0.05, 0.5)

    def _determine_uncertainty_type(self, raw_score: float, domain: Optional[str]) -> UncertaintyType:
        """Determine the dominant type of uncertainty"""
        # Check for epistemic (lack of data)
        bin_idx = self._get_bin_index(raw_score)
        if self.calibration_bins[bin_idx].count < 10:
            return UncertaintyType.EPISTEMIC

        # Check for distributional shift
        if domain and domain in self.domain_calibrators:
            if self.domain_calibrators[domain].detect_shift():
                return UncertaintyType.DISTRIBUTIONAL

        # Scores near 0.5 suggest aleatoric uncertainty
        if 0.4 < raw_score < 0.6:
            return UncertaintyType.ALEATORIC

        return UncertaintyType.MODEL

    def _get_bin_index(self, score: float) -> int:
        """Get bin index for a score"""
        idx = int(score * self.n_bins)
        return min(idx, self.n_bins - 1)

    def record_outcome(self,
                      predicted_confidence: float,
                      actual_outcome: bool,
                      domain: Optional[str] = None):
        """Record outcome for calibration learning"""
        sample = CalibrationSample(
            predicted_confidence=predicted_confidence,
            actual_outcome=actual_outcome,
            domain=domain or "general"
        )

        self.samples.append(sample)
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples:]

        # Update bin
        bin_idx = self._get_bin_index(predicted_confidence)
        self.calibration_bins[bin_idx].predictions.append(predicted_confidence)
        self.calibration_bins[bin_idx].outcomes.append(actual_outcome)

        # Update domain calibrator
        if domain:
            if domain not in self.domain_calibrators:
                self.domain_calibrators[domain] = DomainCalibrator(domain)
            self.domain_calibrators[domain].add_sample(predicted_confidence, actual_outcome)

    def compute_ece(self) -> float:
        """Compute Expected Calibration Error"""
        total_samples = sum(b.count for b in self.calibration_bins)
        if total_samples == 0:
            return 0.0

        ece = 0.0
        for bin_data in self.calibration_bins:
            if bin_data.count > 0:
                weight = bin_data.count / total_samples
                ece += weight * bin_data.calibration_error

        return ece

    def compute_mce(self) -> float:
        """Compute Maximum Calibration Error"""
        errors = [b.calibration_error for b in self.calibration_bins if b.count > 0]
        return max(errors) if errors else 0.0

    def fit_temperature(self, patience: int = 100):
        """Fit temperature scaling parameter"""
        if len(self.samples) < 100:
            return

        # Simple grid search for temperature
        best_ece = float('inf')
        best_temp = 1.0

        confidences = [s.predicted_confidence for s in self.samples]
        outcomes = [s.actual_outcome for s in self.samples]

        for temp in np.linspace(0.5, 2.0, 30):
            # Apply temperature scaling
            scaled = []
            for conf in confidences:
                logit = np.log(conf / (1 - conf + 1e-10) + 1e-10)
                scaled_logit = logit / temp
                scaled.append(1 / (1 + np.exp(-scaled_logit)))

            # Compute ECE with this temperature
            ece = self._compute_ece_for_predictions(scaled, outcomes)

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self.temperature = best_temp

    def _compute_ece_for_predictions(self, predictions: List[float], outcomes: List[bool]) -> float:
        """Compute ECE for a list of predictions"""
        bins = defaultdict(lambda: {"preds": [], "outcomes": []})

        for pred, out in zip(predictions, outcomes):
            bin_idx = self._get_bin_index(pred)
            bins[bin_idx]["preds"].append(pred)
            bins[bin_idx]["outcomes"].append(out)

        total = len(predictions)
        ece = 0.0

        for bin_data in bins.values():
            if bin_data["preds"]:
                weight = len(bin_data["preds"]) / total
                avg_conf = np.mean(bin_data["preds"])
                accuracy = np.mean(bin_data["outcomes"])
                ece += weight * abs(avg_conf - accuracy)

        return ece

    def get_reliability_diagram_data(self) -> Dict[str, List[float]]:
        """Get data for plotting reliability diagram"""
        return {
            "bin_centers": [(b.bin_lower + b.bin_upper) / 2 for b in self.calibration_bins],
            "avg_confidence": [b.avg_confidence for b in self.calibration_bins],
            "accuracy": [b.accuracy for b in self.calibration_bins],
            "counts": [b.count for b in self.calibration_bins]
        }

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration state"""
        return {
            "ece": self.compute_ece(),
            "mce": self.compute_mce(),
            "temperature": self.temperature,
            "total_samples": len(self.samples),
            "domain_count": len(self.domain_calibrators),
            "bin_stats": [
                {
                    "range": f"{b.bin_lower:.1f}-{b.bin_upper:.1f}",
                    "count": b.count,
                    "avg_confidence": b.avg_confidence,
                    "accuracy": b.accuracy
                }
                for b in self.calibration_bins
            ]
        }

    def is_well_calibrated(self, threshold: float = 0.1) -> bool:
        """Check if the system is well-calibrated"""
        return self.compute_ece() < threshold

    def export_state(self, filepath: str):
        """Export calibration state"""
        state = {
            "temperature": self.temperature,
            "bias": self.bias,
            "scale": self.scale,
            "ece": self.compute_ece(),
            "mce": self.compute_mce(),
            "bins": [
                {
                    "lower": b.bin_lower,
                    "upper": b.bin_upper,
                    "count": b.count,
                    "avg_confidence": b.avg_confidence,
                    "accuracy": b.accuracy
                }
                for b in self.calibration_bins
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)


class DomainCalibrator:
    """Domain-specific calibration"""

    def __init__(self, domain: str):
        self.domain = domain
        self.samples: List[Tuple[float, bool]] = []
        self.temperature: float = 1.0
        self.bias: float = 0.0
        self.recent_accuracy: float = 0.5

    def add_sample(self, confidence: float, outcome: bool):
        """Add a calibration sample"""
        self.samples.append((confidence, outcome))
        if len(self.samples) > 1000:
            self.samples = self.samples[-1000:]

        # Update recent accuracy
        recent = self.samples[-50:]
        self.recent_accuracy = np.mean([s[1] for s in recent])

    def calibrate(self, raw_score: float) -> float:
        """Apply domain-specific calibration"""
        logit = np.log(raw_score / (1 - raw_score + 1e-10) + 1e-10)
        scaled = logit / self.temperature
        return np.clip(1 / (1 + np.exp(-scaled)) + self.bias, 0, 1)

    def get_uncertainty(self) -> float:
        """Get domain-specific uncertainty"""
        if len(self.samples) < 10:
            return 0.3

        confidences = [s[0] for s in self.samples[-100:]]
        outcomes = [s[1] for s in self.samples[-100:]]

        return abs(np.mean(confidences) - np.mean(outcomes))

    def detect_shift(self) -> bool:
        """Detect distribution shift in recent data"""
        if len(self.samples) < 100:
            return False

        old_accuracy = np.mean([s[1] for s in self.samples[:50]])
        new_accuracy = np.mean([s[1] for s in self.samples[-50:]])

        return abs(old_accuracy - new_accuracy) > 0.2


class UncertaintyEstimator:
    """
    Advanced uncertainty estimation using multiple methods.
    """

    def __init__(self):
        self.ensemble_predictions: List[List[float]] = []
        self.dropout_samples: int = 10

    def estimate_from_ensemble(self, predictions: List[float]) -> ConfidenceEstimate:
        """Estimate uncertainty from ensemble predictions"""
        if not predictions:
            return ConfidenceEstimate(
                value=0.5,
                lower_bound=0.0,
                upper_bound=1.0,
                uncertainty_type=UncertaintyType.EPISTEMIC
            )

        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        return ConfidenceEstimate(
            value=mean_pred,
            lower_bound=max(0, mean_pred - 2 * std_pred),
            upper_bound=min(1, mean_pred + 2 * std_pred),
            uncertainty_type=UncertaintyType.EPISTEMIC
        )

    def decompose_uncertainty(self,
                             predictions: List[List[float]]) -> Dict[str, float]:
        """
        Decompose total uncertainty into aleatoric and epistemic components.

        Args:
            predictions: List of probability distributions from ensemble members

        Returns:
            Dictionary with aleatoric and epistemic uncertainty values
        """
        if not predictions:
            return {"aleatoric": 0.5, "epistemic": 0.5, "total": 0.7}

        predictions = np.array(predictions)

        # Expected entropy (aleatoric)
        expected_entropy = np.mean([
            -np.sum(p * np.log(p + 1e-10)) for p in predictions
        ])

        # Entropy of expected (total)
        mean_pred = np.mean(predictions, axis=0)
        entropy_of_expected = -np.sum(mean_pred * np.log(mean_pred + 1e-10))

        # Epistemic = total - aleatoric
        aleatoric = expected_entropy
        epistemic = max(0, entropy_of_expected - expected_entropy)
        total = np.sqrt(aleatoric ** 2 + epistemic ** 2)

        return {
            "aleatoric": float(aleatoric),
            "epistemic": float(epistemic),
            "total": float(total)
        }
