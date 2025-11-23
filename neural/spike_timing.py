"""
DENDRITE-X: Spike-Timing-Dependent Plasticity (STDP)

Implements STDP learning rules where synaptic changes depend on
the precise timing between pre and post-synaptic spikes.

Part of AGENT DENDRITE-X (Agent #3) - Advanced Neural Architecture
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# STDP Types
# ============================================================================

class STDPType(Enum):
    """Types of STDP learning rules"""
    CLASSIC = "classic"  # Asymmetric Hebbian
    SYMMETRIC = "symmetric"  # Symmetric around zero
    ANTI_HEBBIAN = "anti_hebbian"  # Reversed classic
    TRIPLET = "triplet"  # Triplet-based rule
    VOLTAGE_BASED = "voltage_based"  # Voltage-dependent


class SpikeEventType(Enum):
    """Types of spike events"""
    PRESYNAPTIC = "pre"
    POSTSYNAPTIC = "post"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class STDPConfig:
    """Configuration for STDP learning"""
    # Time constants (in steps)
    tau_plus: float = 20.0  # LTP time constant
    tau_minus: float = 20.0  # LTD time constant

    # Amplitude
    A_plus: float = 0.01  # LTP amplitude
    A_minus: float = 0.01  # LTD amplitude

    # Weight bounds
    w_max: float = 1.0
    w_min: float = 0.0

    # Learning rate
    learning_rate: float = 1.0

    # Triplet rule parameters
    tau_x: float = 15.0  # Fast presynaptic trace
    tau_y: float = 15.0  # Fast postsynaptic trace
    tau_slow: float = 100.0  # Slow trace

    # Weight dependence
    weight_dependence: str = "soft"  # "soft", "hard", or "none"
    mu: float = 0.5  # Weight dependence exponent


@dataclass
class SpikeEvent:
    """A spike event with timing information"""
    neuron_idx: int
    timestamp: float
    event_type: SpikeEventType


# ============================================================================
# STDP Engine
# ============================================================================

class STDPEngine:
    """
    Spike-Timing-Dependent Plasticity engine.

    Implements multiple STDP rules:
    1. Classic asymmetric Hebbian STDP
    2. Symmetric STDP
    3. Triplet STDP (more biologically accurate)
    4. Voltage-based STDP

    Temporal dynamics:
    - Pre-before-post: Potentiation (LTP)
    - Post-before-pre: Depression (LTD)

    Example:
        stdp = STDPEngine(num_neurons=100)

        # Record spikes
        stdp.record_spike(neuron_idx=5, is_presynaptic=True)
        stdp.record_spike(neuron_idx=10, is_presynaptic=False)

        # Apply STDP
        stdp.apply_stdp()
    """

    def __init__(
        self,
        num_neurons: int,
        weights: Optional[np.ndarray] = None,
        config: Optional[STDPConfig] = None
    ):
        self.num_neurons = num_neurons
        self.config = config or STDPConfig()

        # Weight matrix
        if weights is not None:
            self.weights = weights.copy()
        else:
            self.weights = np.random.rand(num_neurons, num_neurons) * 0.5
            np.fill_diagonal(self.weights, 0)

        # Spike traces (exponentially decaying)
        self.pre_trace = np.zeros(num_neurons)  # Presynaptic trace
        self.post_trace = np.zeros(num_neurons)  # Postsynaptic trace

        # For triplet rule
        self.pre_trace_slow = np.zeros(num_neurons)
        self.post_trace_slow = np.zeros(num_neurons)

        # Spike history
        self.spike_history: List[SpikeEvent] = []
        self.max_history = 1000

        # Current time
        self.current_time = 0.0
        self.dt = 1.0  # Time step

        # Statistics
        self.total_ltp = 0.0
        self.total_ltd = 0.0
        self.update_count = 0

    def record_spike(
        self,
        neuron_idx: int,
        is_presynaptic: bool = True
    ):
        """
        Record a spike event.

        Args:
            neuron_idx: Index of spiking neuron
            is_presynaptic: True if presynaptic spike
        """
        event_type = SpikeEventType.PRESYNAPTIC if is_presynaptic else SpikeEventType.POSTSYNAPTIC

        event = SpikeEvent(
            neuron_idx=neuron_idx,
            timestamp=self.current_time,
            event_type=event_type
        )

        self.spike_history.append(event)

        # Update traces
        if is_presynaptic:
            self.pre_trace[neuron_idx] += 1.0
            self.pre_trace_slow[neuron_idx] += 1.0
        else:
            self.post_trace[neuron_idx] += 1.0
            self.post_trace_slow[neuron_idx] += 1.0

        # Trim history
        if len(self.spike_history) > self.max_history:
            self.spike_history = self.spike_history[-self.max_history//2:]

    def record_spikes_batch(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray
    ):
        """
        Record batch of spikes from spike indicators.

        Args:
            pre_spikes: Boolean array of presynaptic spikes
            post_spikes: Boolean array of postsynaptic spikes
        """
        for i in np.where(pre_spikes)[0]:
            self.record_spike(i, is_presynaptic=True)

        for i in np.where(post_spikes)[0]:
            self.record_spike(i, is_presynaptic=False)

    def decay_traces(self):
        """Apply exponential decay to spike traces"""
        # Fast traces
        self.pre_trace *= np.exp(-self.dt / self.config.tau_plus)
        self.post_trace *= np.exp(-self.dt / self.config.tau_minus)

        # Slow traces (for triplet)
        self.pre_trace_slow *= np.exp(-self.dt / self.config.tau_slow)
        self.post_trace_slow *= np.exp(-self.dt / self.config.tau_slow)

    def _weight_dependence(self, w: float, is_ltp: bool) -> float:
        """Compute weight dependence factor"""
        if self.config.weight_dependence == "none":
            return 1.0

        elif self.config.weight_dependence == "soft":
            if is_ltp:
                return (1 - w / self.config.w_max) ** self.config.mu
            else:
                return (w / self.config.w_max) ** self.config.mu

        elif self.config.weight_dependence == "hard":
            if is_ltp:
                return self.config.w_max - w
            else:
                return w - self.config.w_min

        return 1.0

    def apply_classic_stdp(self) -> np.ndarray:
        """
        Apply classic asymmetric STDP rule.

        ΔW = A+ * exp(-Δt/τ+) for Δt > 0 (pre before post)
        ΔW = -A- * exp(Δt/τ-) for Δt < 0 (post before pre)
        """
        delta_w = np.zeros((self.num_neurons, self.num_neurons))

        # LTP: pre-before-post
        # For each presynaptic spike, check postsynaptic trace
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j:
                    continue

                # LTP contribution (pre->post)
                ltp = self.config.A_plus * self.pre_trace[i] * self.post_trace[j]
                ltp *= self._weight_dependence(self.weights[i, j], is_ltp=True)

                # LTD contribution (post->pre)
                ltd = self.config.A_minus * self.post_trace[i] * self.pre_trace[j]
                ltd *= self._weight_dependence(self.weights[i, j], is_ltp=False)

                delta_w[i, j] = self.config.learning_rate * (ltp - ltd)

        return delta_w

    def apply_symmetric_stdp(self) -> np.ndarray:
        """Apply symmetric STDP rule (both directions can potentiate)"""
        delta_w = np.zeros((self.num_neurons, self.num_neurons))

        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j:
                    continue

                # Symmetric: both directions contribute to LTP
                correlation = self.pre_trace[i] * self.post_trace[j]
                correlation += self.post_trace[i] * self.pre_trace[j]

                delta_w[i, j] = self.config.learning_rate * self.config.A_plus * correlation

        return delta_w

    def apply_triplet_stdp(self) -> np.ndarray:
        """
        Apply triplet STDP rule.

        More biologically accurate, considers triplets of spikes.
        Based on Pfister & Gerstner (2006).
        """
        delta_w = np.zeros((self.num_neurons, self.num_neurons))

        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j:
                    continue

                # Pair term (classic STDP)
                ltp_pair = self.config.A_plus * self.pre_trace[i] * self.post_trace[j]

                # Triplet term (requires slow trace)
                ltp_triplet = (self.config.A_plus * 2) * self.pre_trace[i] * \
                              self.post_trace[j] * self.post_trace_slow[j]

                # LTD pair
                ltd_pair = self.config.A_minus * self.post_trace[i] * self.pre_trace[j]

                # LTD triplet
                ltd_triplet = (self.config.A_minus * 2) * self.post_trace[i] * \
                              self.pre_trace[j] * self.pre_trace_slow[i]

                delta_w[i, j] = self.config.learning_rate * (
                    ltp_pair + ltp_triplet - ltd_pair - ltd_triplet
                )

        return delta_w

    def apply_stdp(
        self,
        stdp_type: STDPType = STDPType.CLASSIC
    ) -> np.ndarray:
        """
        Apply STDP updates to weights.

        Args:
            stdp_type: Type of STDP rule to use

        Returns:
            Weight change matrix
        """
        if stdp_type == STDPType.CLASSIC:
            delta_w = self.apply_classic_stdp()
        elif stdp_type == STDPType.SYMMETRIC:
            delta_w = self.apply_symmetric_stdp()
        elif stdp_type == STDPType.TRIPLET:
            delta_w = self.apply_triplet_stdp()
        elif stdp_type == STDPType.ANTI_HEBBIAN:
            delta_w = -self.apply_classic_stdp()
        else:
            delta_w = self.apply_classic_stdp()

        # Apply weight changes
        self.weights += delta_w

        # Clip weights
        self.weights = np.clip(
            self.weights,
            self.config.w_min,
            self.config.w_max
        )

        # Update statistics
        self.total_ltp += np.sum(np.maximum(delta_w, 0))
        self.total_ltd += np.sum(np.maximum(-delta_w, 0))
        self.update_count += 1

        return delta_w

    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """
        Perform one simulation step.

        Args:
            pre_spikes: Boolean array of presynaptic spikes
            post_spikes: Boolean array of postsynaptic spikes
        """
        # Record spikes
        self.record_spikes_batch(pre_spikes, post_spikes)

        # Apply STDP
        self.apply_stdp()

        # Decay traces
        self.decay_traces()

        # Advance time
        self.current_time += self.dt

    def get_timing_window(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get STDP timing window curve.

        Returns:
            Tuple of (delta_t, delta_w) for plotting
        """
        delta_t = np.linspace(-50, 50, num_points)
        delta_w = np.zeros(num_points)

        for i, dt in enumerate(delta_t):
            if dt > 0:  # Pre before post: LTP
                delta_w[i] = self.config.A_plus * np.exp(-dt / self.config.tau_plus)
            else:  # Post before pre: LTD
                delta_w[i] = -self.config.A_minus * np.exp(dt / self.config.tau_minus)

        return delta_t, delta_w

    def get_statistics(self) -> Dict[str, Any]:
        """Get STDP statistics"""
        return {
            "current_time": self.current_time,
            "update_count": self.update_count,
            "total_ltp": self.total_ltp,
            "total_ltd": self.total_ltd,
            "ltp_ltd_ratio": self.total_ltp / (self.total_ltd + 1e-10),
            "mean_weight": float(np.mean(self.weights)),
            "std_weight": float(np.std(self.weights)),
            "mean_pre_trace": float(np.mean(self.pre_trace)),
            "mean_post_trace": float(np.mean(self.post_trace))
        }

    def get_weight_matrix(self) -> np.ndarray:
        """Get current weight matrix"""
        return self.weights.copy()


# ============================================================================
# Reward-Modulated STDP
# ============================================================================

class RewardModulatedSTDP:
    """
    STDP modulated by reward signals (dopamine-like).

    Eligibility traces allow credit assignment over time.
    """

    def __init__(
        self,
        stdp_engine: STDPEngine,
        eligibility_decay: float = 0.95,
        reward_scale: float = 1.0
    ):
        self.engine = stdp_engine
        self.eligibility_decay = eligibility_decay
        self.reward_scale = reward_scale

        # Eligibility traces
        self.eligibility = np.zeros((stdp_engine.num_neurons, stdp_engine.num_neurons))

    def update_eligibility(self, delta_w: np.ndarray):
        """Update eligibility traces with STDP-computed changes"""
        self.eligibility = self.eligibility_decay * self.eligibility + delta_w

    def apply_reward(self, reward: float) -> np.ndarray:
        """
        Apply reward signal to convert eligibility to actual weight changes.

        Args:
            reward: Reward signal (-1 to 1)

        Returns:
            Actual weight changes applied
        """
        # Convert eligibility to weight changes based on reward
        delta_w = reward * self.reward_scale * self.eligibility

        # Apply to weights
        self.engine.weights += delta_w
        self.engine.weights = np.clip(
            self.engine.weights,
            self.engine.config.w_min,
            self.engine.config.w_max
        )

        return delta_w

    def step(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        reward: Optional[float] = None
    ):
        """
        Perform one step with optional reward.

        Args:
            pre_spikes: Presynaptic spikes
            post_spikes: Postsynaptic spikes
            reward: Optional reward signal
        """
        # Record spikes
        self.engine.record_spikes_batch(pre_spikes, post_spikes)

        # Compute STDP (but don't apply directly)
        delta_w = self.engine.apply_classic_stdp()

        # Update eligibility
        self.update_eligibility(delta_w)

        # Apply reward if provided
        if reward is not None:
            self.apply_reward(reward)

        # Decay traces
        self.engine.decay_traces()
        self.engine.current_time += self.engine.dt


# ============================================================================
# Homeostatic STDP
# ============================================================================

class HomeostaticSTDP:
    """
    STDP with homeostatic regulation to prevent runaway potentiation.
    """

    def __init__(
        self,
        stdp_engine: STDPEngine,
        target_rate: float = 0.1,
        homeostatic_tau: float = 1000.0
    ):
        self.engine = stdp_engine
        self.target_rate = target_rate
        self.homeostatic_tau = homeostatic_tau

        # Running rate estimate
        self.rate_estimate = np.ones(stdp_engine.num_neurons) * target_rate

    def update_rate_estimate(self, spikes: np.ndarray):
        """Update running estimate of firing rate"""
        alpha = 1.0 / self.homeostatic_tau
        self.rate_estimate = (1 - alpha) * self.rate_estimate + alpha * spikes.astype(float)

    def get_homeostatic_factor(self) -> np.ndarray:
        """Compute homeostatic scaling factor"""
        # Neurons above target rate have reduced LTP
        # Neurons below target rate have enhanced LTP
        return self.target_rate / (self.rate_estimate + 1e-10)

    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """Perform homeostatic STDP step"""
        # Update rate estimates
        self.update_rate_estimate(pre_spikes)
        self.update_rate_estimate(post_spikes)

        # Get homeostatic factor
        h_factor = self.get_homeostatic_factor()

        # Temporarily modify A_plus based on homeostasis
        original_A_plus = self.engine.config.A_plus
        self.engine.config.A_plus *= np.mean(h_factor)

        # Run STDP
        self.engine.step(pre_spikes, post_spikes)

        # Restore
        self.engine.config.A_plus = original_A_plus


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("DENDRITE-X: Spike-Timing-Dependent Plasticity Demo")
    print("=" * 50)

    # Create STDP engine
    stdp = STDPEngine(num_neurons=50)

    print(f"\n1. Initial State:")
    print(f"   Mean weight: {np.mean(stdp.weights):.4f}")

    # Simulate with correlated spiking
    print("\n2. Simulating Correlated Spiking:")
    for step in range(100):
        # Create correlated pre-post pattern
        pre = (np.random.rand(50) > 0.9).astype(float)
        post = np.roll(pre, 1) * (np.random.rand(50) > 0.5)  # Post follows pre

        stdp.step(pre, post)

    stats = stdp.get_statistics()
    print(f"   Mean weight: {stats['mean_weight']:.4f}")
    print(f"   LTP/LTD ratio: {stats['ltp_ltd_ratio']:.4f}")

    # Get timing window
    print("\n3. STDP Timing Window:")
    delta_t, delta_w = stdp.get_timing_window()
    print(f"   Max LTP (at Δt=1): {delta_w[len(delta_w)//2+1]:.4f}")
    print(f"   Max LTD (at Δt=-1): {delta_w[len(delta_w)//2-1]:.4f}")

    # Triplet STDP
    print("\n4. Triplet STDP:")
    stdp_triplet = STDPEngine(num_neurons=50)
    for step in range(100):
        pre = (np.random.rand(50) > 0.9).astype(float)
        post = (np.random.rand(50) > 0.9).astype(float)
        stdp_triplet.record_spikes_batch(pre, post)
        stdp_triplet.apply_stdp(STDPType.TRIPLET)
        stdp_triplet.decay_traces()
        stdp_triplet.current_time += 1

    stats_triplet = stdp_triplet.get_statistics()
    print(f"   Mean weight: {stats_triplet['mean_weight']:.4f}")

    # Reward-modulated STDP
    print("\n5. Reward-Modulated STDP:")
    rm_stdp = RewardModulatedSTDP(STDPEngine(num_neurons=50))

    for step in range(100):
        pre = (np.random.rand(50) > 0.9).astype(float)
        post = (np.random.rand(50) > 0.9).astype(float)
        reward = np.random.randn() * 0.5  # Random reward
        rm_stdp.step(pre, post, reward)

    print(f"   Mean weight: {np.mean(rm_stdp.engine.weights):.4f}")

    # Homeostatic STDP
    print("\n6. Homeostatic STDP:")
    h_stdp = HomeostaticSTDP(STDPEngine(num_neurons=50), target_rate=0.1)

    for step in range(100):
        pre = (np.random.rand(50) > 0.9).astype(float)
        post = (np.random.rand(50) > 0.9).astype(float)
        h_stdp.step(pre, post)

    print(f"   Mean weight: {np.mean(h_stdp.engine.weights):.4f}")
    print(f"   Mean rate estimate: {np.mean(h_stdp.rate_estimate):.4f}")

    print("\n" + "=" * 50)
    print("STDP Engine ready!")
