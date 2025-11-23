"""
DENDRITE-X: Neural Oscillations

Brain wave patterns (alpha, beta, theta, gamma) for consciousness states.
Implements oscillatory dynamics and their effects on cognitive processing.

Part of AGENT DENDRITE-X (Agent #3) - Advanced Neural Architecture
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Brain Wave Types
# ============================================================================

class BrainWave(Enum):
    """Types of brain waves with frequency ranges"""
    DELTA = "delta"  # 0.5-4 Hz - Deep sleep
    THETA = "theta"  # 4-8 Hz - Drowsy, meditation, memory
    ALPHA = "alpha"  # 8-13 Hz - Relaxed awareness
    BETA = "beta"  # 13-30 Hz - Active thinking
    GAMMA = "gamma"  # 30-100 Hz - Higher consciousness, binding


class ConsciousnessMode(Enum):
    """Modes of consciousness associated with oscillation patterns"""
    DEEP_SLEEP = "deep_sleep"  # Dominant delta
    REM_SLEEP = "rem_sleep"  # Theta + beta bursts
    DROWSY = "drowsy"  # Theta dominant
    RELAXED = "relaxed"  # Alpha dominant
    ALERT = "alert"  # Beta dominant
    FOCUSED = "focused"  # High beta + gamma
    FLOW = "flow"  # Balanced with strong gamma
    MEDITATIVE = "meditative"  # Alpha-theta border


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OscillationConfig:
    """Configuration for neural oscillations"""
    # Frequency ranges (Hz)
    delta_range: Tuple[float, float] = (0.5, 4.0)
    theta_range: Tuple[float, float] = (4.0, 8.0)
    alpha_range: Tuple[float, float] = (8.0, 13.0)
    beta_range: Tuple[float, float] = (13.0, 30.0)
    gamma_range: Tuple[float, float] = (30.0, 100.0)

    # Simulation parameters
    sample_rate: float = 1000.0  # Hz
    dt: float = 0.001  # seconds

    # Coupling parameters
    cross_frequency_coupling: float = 0.3
    phase_amplitude_coupling: float = 0.2


@dataclass
class OscillatorState:
    """State of a single oscillator"""
    frequency: float
    amplitude: float
    phase: float
    wave_type: BrainWave


# ============================================================================
# Oscillation Manager
# ============================================================================

class OscillationManager:
    """
    Manages neural oscillations and brain wave patterns.

    Implements:
    1. Multiple frequency band oscillators
    2. Cross-frequency coupling
    3. Phase-amplitude coupling
    4. Consciousness mode transitions

    Inspired by:
    - EEG/MEG research
    - Oscillatory theories of consciousness
    - Binding by synchrony hypothesis

    Example:
        manager = OscillationManager()

        # Set consciousness mode
        manager.set_mode(ConsciousnessMode.FOCUSED)

        # Generate oscillation signal
        signal, spectrum = manager.generate(duration=1.0)

        # Get current dominant frequency
        dominant = manager.get_dominant_frequency()
    """

    def __init__(self, config: Optional[OscillationConfig] = None):
        self.config = config or OscillationConfig()

        # Initialize oscillators for each band
        self.oscillators: Dict[BrainWave, OscillatorState] = {}
        self._init_oscillators()

        # Current mode
        self.mode = ConsciousnessMode.RELAXED

        # Mode-specific amplitude profiles
        self.mode_profiles = {
            ConsciousnessMode.DEEP_SLEEP: {
                BrainWave.DELTA: 1.0, BrainWave.THETA: 0.2,
                BrainWave.ALPHA: 0.1, BrainWave.BETA: 0.05, BrainWave.GAMMA: 0.01
            },
            ConsciousnessMode.REM_SLEEP: {
                BrainWave.DELTA: 0.3, BrainWave.THETA: 0.7,
                BrainWave.ALPHA: 0.2, BrainWave.BETA: 0.4, BrainWave.GAMMA: 0.2
            },
            ConsciousnessMode.DROWSY: {
                BrainWave.DELTA: 0.2, BrainWave.THETA: 0.8,
                BrainWave.ALPHA: 0.4, BrainWave.BETA: 0.2, BrainWave.GAMMA: 0.1
            },
            ConsciousnessMode.RELAXED: {
                BrainWave.DELTA: 0.1, BrainWave.THETA: 0.3,
                BrainWave.ALPHA: 1.0, BrainWave.BETA: 0.3, BrainWave.GAMMA: 0.1
            },
            ConsciousnessMode.ALERT: {
                BrainWave.DELTA: 0.05, BrainWave.THETA: 0.2,
                BrainWave.ALPHA: 0.4, BrainWave.BETA: 0.8, BrainWave.GAMMA: 0.3
            },
            ConsciousnessMode.FOCUSED: {
                BrainWave.DELTA: 0.02, BrainWave.THETA: 0.1,
                BrainWave.ALPHA: 0.2, BrainWave.BETA: 1.0, BrainWave.GAMMA: 0.6
            },
            ConsciousnessMode.FLOW: {
                BrainWave.DELTA: 0.05, BrainWave.THETA: 0.4,
                BrainWave.ALPHA: 0.6, BrainWave.BETA: 0.7, BrainWave.GAMMA: 0.8
            },
            ConsciousnessMode.MEDITATIVE: {
                BrainWave.DELTA: 0.1, BrainWave.THETA: 0.9,
                BrainWave.ALPHA: 0.9, BrainWave.BETA: 0.2, BrainWave.GAMMA: 0.5
            }
        }

        # Simulation state
        self.current_time = 0.0

        # History for analysis
        self.amplitude_history: List[Dict[BrainWave, float]] = []
        self.phase_history: List[Dict[BrainWave, float]] = []

    def _init_oscillators(self):
        """Initialize oscillators for each frequency band"""
        band_centers = {
            BrainWave.DELTA: 2.0,
            BrainWave.THETA: 6.0,
            BrainWave.ALPHA: 10.0,
            BrainWave.BETA: 20.0,
            BrainWave.GAMMA: 40.0
        }

        for wave_type, freq in band_centers.items():
            self.oscillators[wave_type] = OscillatorState(
                frequency=freq,
                amplitude=0.5,
                phase=np.random.uniform(0, 2 * np.pi),
                wave_type=wave_type
            )

    def set_mode(self, mode: ConsciousnessMode, transition_steps: int = 100):
        """
        Set consciousness mode with smooth transition.

        Args:
            mode: Target consciousness mode
            transition_steps: Steps for smooth transition (not implemented in simple version)
        """
        self.mode = mode

        # Update amplitudes based on mode profile
        profile = self.mode_profiles[mode]
        for wave_type, amplitude in profile.items():
            self.oscillators[wave_type].amplitude = amplitude

    def step(self):
        """Advance oscillation by one time step"""
        dt = self.config.dt

        # Update phases
        for osc in self.oscillators.values():
            osc.phase += 2 * np.pi * osc.frequency * dt
            osc.phase = osc.phase % (2 * np.pi)

        # Apply cross-frequency coupling (gamma coupled to theta phase)
        theta_phase = self.oscillators[BrainWave.THETA].phase
        gamma = self.oscillators[BrainWave.GAMMA]

        # Phase-amplitude coupling: gamma amplitude modulated by theta phase
        pac_modulation = 1 + self.config.phase_amplitude_coupling * np.cos(theta_phase)
        gamma.amplitude = self.mode_profiles[self.mode][BrainWave.GAMMA] * pac_modulation

        self.current_time += dt

        # Store history
        self.amplitude_history.append({
            w: osc.amplitude for w, osc in self.oscillators.items()
        })
        self.phase_history.append({
            w: osc.phase for w, osc in self.oscillators.items()
        })

        # Trim history
        if len(self.amplitude_history) > 10000:
            self.amplitude_history = self.amplitude_history[-5000:]
            self.phase_history = self.phase_history[-5000:]

    def get_signal(self) -> float:
        """Get combined oscillation signal at current time"""
        signal = 0.0
        for osc in self.oscillators.values():
            signal += osc.amplitude * np.sin(osc.phase)
        return signal

    def get_band_signals(self) -> Dict[BrainWave, float]:
        """Get individual band signals"""
        return {
            wave: osc.amplitude * np.sin(osc.phase)
            for wave, osc in self.oscillators.items()
        }

    def generate(self, duration: float) -> Tuple[np.ndarray, Dict[BrainWave, np.ndarray]]:
        """
        Generate oscillation signals for specified duration.

        Args:
            duration: Duration in seconds

        Returns:
            Tuple of (combined_signal, band_signals_dict)
        """
        num_samples = int(duration * self.config.sample_rate)
        combined = np.zeros(num_samples)
        band_signals = {wave: np.zeros(num_samples) for wave in BrainWave}

        for i in range(num_samples):
            self.step()
            combined[i] = self.get_signal()
            for wave, signal in self.get_band_signals().items():
                band_signals[wave][i] = signal

        return combined, band_signals

    def get_dominant_frequency(self) -> Tuple[BrainWave, float]:
        """Get currently dominant frequency band"""
        max_amp = 0.0
        dominant = BrainWave.ALPHA

        for wave, osc in self.oscillators.items():
            if osc.amplitude > max_amp:
                max_amp = osc.amplitude
                dominant = wave

        return dominant, max_amp

    def get_power_spectrum(self) -> Dict[BrainWave, float]:
        """Get power in each frequency band (amplitude squared)"""
        return {
            wave: osc.amplitude ** 2
            for wave, osc in self.oscillators.items()
        }

    def compute_coherence(self, band1: BrainWave, band2: BrainWave) -> float:
        """Compute phase coherence between two frequency bands"""
        phase1 = self.oscillators[band1].phase
        phase2 = self.oscillators[band2].phase

        # Simple phase coherence (would need history for proper computation)
        phase_diff = phase1 - phase2
        return np.abs(np.cos(phase_diff))

    def get_consciousness_metrics(self) -> Dict[str, float]:
        """Get consciousness-related metrics from oscillations"""
        power = self.get_power_spectrum()

        # Compute ratios and indices
        alpha_theta_ratio = power[BrainWave.ALPHA] / (power[BrainWave.THETA] + 1e-10)
        beta_alpha_ratio = power[BrainWave.BETA] / (power[BrainWave.ALPHA] + 1e-10)

        # Approximate alertness index
        alertness = (power[BrainWave.BETA] + power[BrainWave.GAMMA]) / (
            power[BrainWave.DELTA] + power[BrainWave.THETA] + 1e-10
        )

        # Consciousness index (simplified)
        consciousness_index = (
            0.2 * power[BrainWave.ALPHA] +
            0.3 * power[BrainWave.BETA] +
            0.5 * power[BrainWave.GAMMA]
        ) / (
            0.4 * power[BrainWave.DELTA] +
            0.2 * power[BrainWave.THETA] + 1e-10
        )

        return {
            "alpha_theta_ratio": alpha_theta_ratio,
            "beta_alpha_ratio": beta_alpha_ratio,
            "alertness_index": alertness,
            "consciousness_index": consciousness_index,
            "dominant_band": self.get_dominant_frequency()[0].value,
            "gamma_coherence": self.compute_coherence(BrainWave.GAMMA, BrainWave.THETA)
        }

    def get_state(self) -> Dict[str, Any]:
        """Get complete oscillation state"""
        return {
            "mode": self.mode.value,
            "current_time": self.current_time,
            "oscillators": {
                wave.value: {
                    "frequency": osc.frequency,
                    "amplitude": osc.amplitude,
                    "phase": osc.phase
                }
                for wave, osc in self.oscillators.items()
            },
            "power_spectrum": {
                w.value: p for w, p in self.get_power_spectrum().items()
            },
            "consciousness_metrics": self.get_consciousness_metrics()
        }


# ============================================================================
# Neural Synchrony
# ============================================================================

class NeuralSynchrony:
    """
    Manages synchronization between neural populations.

    Implements binding-by-synchrony hypothesis.
    """

    def __init__(
        self,
        num_populations: int,
        oscillation_manager: OscillationManager
    ):
        self.num_populations = num_populations
        self.osc_manager = oscillation_manager

        # Phase of each population
        self.population_phases = np.random.uniform(0, 2 * np.pi, num_populations)

        # Coupling matrix
        self.coupling = np.random.rand(num_populations, num_populations) * 0.1
        np.fill_diagonal(self.coupling, 0)

        # Natural frequencies (slight variations around dominant)
        base_freq = 10.0  # Hz
        self.natural_frequencies = base_freq + np.random.randn(num_populations) * 0.5

    def kuramoto_step(self, dt: float = 0.001):
        """
        Update phases using Kuramoto model of coupled oscillators.

        Kuramoto model: dθ_i/dt = ω_i + (K/N) * Σ sin(θ_j - θ_i)
        """
        phase_diffs = self.population_phases[:, np.newaxis] - self.population_phases[np.newaxis, :]
        coupling_term = np.sum(self.coupling * np.sin(-phase_diffs), axis=1) / self.num_populations

        # Update phases
        self.population_phases += dt * (self.natural_frequencies + coupling_term)
        self.population_phases = self.population_phases % (2 * np.pi)

    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter (synchrony measure).

        Returns:
            Tuple of (synchrony, mean_phase)
        """
        complex_phases = np.exp(1j * self.population_phases)
        order = np.mean(complex_phases)

        synchrony = np.abs(order)
        mean_phase = np.angle(order)

        return synchrony, mean_phase

    def get_synchronized_groups(self, threshold: float = 0.3) -> List[List[int]]:
        """
        Find groups of synchronized populations.

        Args:
            threshold: Phase difference threshold for grouping

        Returns:
            List of synchronized population groups
        """
        groups = []
        assigned = set()

        for i in range(self.num_populations):
            if i in assigned:
                continue

            group = [i]
            assigned.add(i)

            for j in range(i + 1, self.num_populations):
                if j in assigned:
                    continue

                phase_diff = np.abs(self.population_phases[i] - self.population_phases[j])
                phase_diff = min(phase_diff, 2 * np.pi - phase_diff)

                if phase_diff < threshold:
                    group.append(j)
                    assigned.add(j)

            if len(group) > 1:
                groups.append(group)

        return groups


# ============================================================================
# Sleep-Wake Dynamics
# ============================================================================

class SleepWakeDynamics:
    """
    Models sleep-wake transitions through oscillation changes.
    """

    def __init__(self, oscillation_manager: OscillationManager):
        self.osc_manager = oscillation_manager

        # Sleep state
        self.awake = True
        self.sleep_pressure = 0.0  # Adenosine-like accumulation
        self.circadian_phase = 0.0

        # Parameters
        self.sleep_pressure_rate = 0.001  # Accumulation when awake
        self.sleep_clearance_rate = 0.002  # Clearance during sleep
        self.sleep_threshold = 0.8
        self.wake_threshold = 0.2

    def update(self, dt: float = 1.0):
        """Update sleep-wake dynamics"""
        # Update circadian rhythm (24-hour cycle)
        self.circadian_phase += dt / (24 * 3600)  # Assuming dt in seconds
        self.circadian_phase = self.circadian_phase % 1.0

        # Circadian drive for wakefulness
        circadian_wake = np.cos(2 * np.pi * self.circadian_phase)

        if self.awake:
            # Accumulate sleep pressure
            self.sleep_pressure += self.sleep_pressure_rate * dt

            # Check for sleep onset
            if self.sleep_pressure > self.sleep_threshold and circadian_wake < 0:
                self.enter_sleep()
        else:
            # Clear sleep pressure
            self.sleep_pressure -= self.sleep_clearance_rate * dt
            self.sleep_pressure = max(0, self.sleep_pressure)

            # Check for wake
            if self.sleep_pressure < self.wake_threshold and circadian_wake > 0:
                self.wake_up()

    def enter_sleep(self):
        """Transition to sleep state"""
        self.awake = False
        self.osc_manager.set_mode(ConsciousnessMode.DEEP_SLEEP)

    def wake_up(self):
        """Transition to wake state"""
        self.awake = True
        self.osc_manager.set_mode(ConsciousnessMode.ALERT)

    def get_state(self) -> Dict[str, Any]:
        """Get sleep-wake state"""
        return {
            "awake": self.awake,
            "sleep_pressure": self.sleep_pressure,
            "circadian_phase": self.circadian_phase,
            "consciousness_mode": self.osc_manager.mode.value
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("DENDRITE-X: Neural Oscillations Demo")
    print("=" * 50)

    # Create oscillation manager
    manager = OscillationManager()

    # Test different consciousness modes
    print("\n1. Consciousness Modes:")
    for mode in [ConsciousnessMode.RELAXED, ConsciousnessMode.FOCUSED, ConsciousnessMode.MEDITATIVE]:
        manager.set_mode(mode)
        dominant, amp = manager.get_dominant_frequency()
        metrics = manager.get_consciousness_metrics()
        print(f"   {mode.value}: dominant={dominant.value}, consciousness_idx={metrics['consciousness_index']:.2f}")

    # Generate signal
    print("\n2. Signal Generation:")
    manager.set_mode(ConsciousnessMode.FOCUSED)
    signal, bands = manager.generate(duration=1.0)
    print(f"   Signal length: {len(signal)} samples")
    print(f"   Signal range: [{signal.min():.2f}, {signal.max():.2f}]")

    # Power spectrum
    print("\n3. Power Spectrum:")
    power = manager.get_power_spectrum()
    for wave, p in power.items():
        print(f"   {wave.value}: {p:.4f}")

    # Neural synchrony
    print("\n4. Neural Synchrony:")
    synchrony = NeuralSynchrony(num_populations=10, oscillation_manager=manager)

    for _ in range(100):
        synchrony.kuramoto_step()

    sync_level, mean_phase = synchrony.compute_order_parameter()
    groups = synchrony.get_synchronized_groups()
    print(f"   Global synchrony: {sync_level:.4f}")
    print(f"   Synchronized groups: {len(groups)}")

    # Sleep-wake dynamics
    print("\n5. Sleep-Wake Dynamics:")
    sleep_wake = SleepWakeDynamics(manager)

    # Simulate hours
    for hour in range(24):
        for _ in range(3600):  # Seconds per hour
            sleep_wake.update(dt=1.0)

        if hour % 6 == 0:
            state = sleep_wake.get_state()
            status = "Awake" if state["awake"] else "Asleep"
            print(f"   Hour {hour}: {status}, pressure={state['sleep_pressure']:.2f}")

    # Final state
    print("\n6. Final State:")
    state = manager.get_state()
    print(f"   Mode: {state['mode']}")
    print(f"   Dominant: {state['consciousness_metrics']['dominant_band']}")

    print("\n" + "=" * 50)
    print("Neural Oscillations ready!")
