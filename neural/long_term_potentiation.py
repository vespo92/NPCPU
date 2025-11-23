"""
DENDRITE-X: Long-Term Potentiation (LTP)

Memory consolidation through LTP mechanisms.
Implements biological memory formation processes.

Part of AGENT DENDRITE-X (Agent #3) - Advanced Neural Architecture
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# LTP Phases
# ============================================================================

class LTPPhase(Enum):
    """Phases of long-term potentiation"""
    EARLY_LTP = "early_ltp"  # Protein-independent, < 1-3 hours
    LATE_LTP = "late_ltp"  # Protein-dependent, > 3 hours
    CONSOLIDATED = "consolidated"  # Stable memory
    DECAYED = "decayed"  # Memory has decayed


class LTDPhase(Enum):
    """Phases of long-term depression"""
    EARLY_LTD = "early_ltd"
    LATE_LTD = "late_ltd"
    CONSOLIDATED = "consolidated"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LTPConfig:
    """Configuration for LTP mechanisms"""
    induction_threshold: float = 0.7  # Threshold for LTP induction
    depression_threshold: float = 0.3  # Threshold for LTD
    early_duration: int = 100  # Steps for early phase
    consolidation_threshold: float = 0.8  # Strength needed for late LTP
    decay_rate: float = 0.01
    potentiation_rate: float = 0.1
    depression_rate: float = 0.05
    protein_synthesis_delay: int = 50  # Steps before late LTP
    tagging_threshold: float = 0.6  # Threshold for synaptic tagging


@dataclass
class SynapticTag:
    """Tag marking synapse for potential consolidation"""
    synapse_id: Tuple[int, int]
    tag_strength: float
    created_step: int
    is_potentiation: bool  # True for LTP tag, False for LTD tag
    captured_proteins: float = 0.0


# ============================================================================
# LTP Engine
# ============================================================================

class LTPEngine:
    """
    Long-Term Potentiation engine for memory consolidation.

    Implements:
    1. Early LTP: Rapid but transient potentiation
    2. Synaptic tagging: Marks active synapses
    3. Late LTP: Protein synthesis-dependent consolidation
    4. Synaptic tagging and capture (STC) theory

    Based on:
    - Frey & Morris (1997) Synaptic tagging and LTP
    - Redondo & Morris (2011) Making memories last

    Example:
        ltp = LTPEngine(num_neurons=100)

        # Induce LTP at specific synapses
        pre_activity = np.array([0.9, 0.8, 0.1, ...])
        post_activity = np.array([0.8, 0.9, 0.2, ...])
        ltp.induce(pre_activity, post_activity)

        # Simulate time passing
        for step in range(200):
            ltp.tick()

        # Check consolidated memories
        consolidated = ltp.get_consolidated_synapses()
    """

    def __init__(
        self,
        num_neurons: int,
        weights: Optional[np.ndarray] = None,
        config: Optional[LTPConfig] = None
    ):
        self.num_neurons = num_neurons
        self.config = config or LTPConfig()

        # Weight matrix
        if weights is not None:
            self.weights = weights.copy()
        else:
            self.weights = np.random.randn(num_neurons, num_neurons) * 0.1
            np.fill_diagonal(self.weights, 0)

        # Baseline weights (for computing potentiation)
        self.baseline_weights = self.weights.copy()

        # LTP state tracking
        self.ltp_state: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.ltd_state: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Synaptic tags
        self.tags: Dict[Tuple[int, int], SynapticTag] = {}

        # Protein availability (for late LTP)
        self.protein_pool = np.zeros(num_neurons)
        self.protein_synthesis_rate = 0.01

        # Step counter
        self.current_step = 0

        # Statistics
        self.total_ltp_events = 0
        self.total_ltd_events = 0
        self.total_consolidated = 0

    def compute_coincidence(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray
    ) -> np.ndarray:
        """
        Compute coincidence detector output.

        LTP requires pre-before-post timing (simplified as co-activity here).
        """
        # Outer product gives pairwise coincidence
        coincidence = np.outer(pre_activity, post_activity)
        np.fill_diagonal(coincidence, 0)

        return coincidence

    def induce(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray
    ) -> Dict[str, int]:
        """
        Induce LTP/LTD based on activity patterns.

        Args:
            pre_activity: Presynaptic activity (num_neurons,)
            post_activity: Postsynaptic activity (num_neurons,)

        Returns:
            Dict with counts of LTP and LTD inductions
        """
        coincidence = self.compute_coincidence(pre_activity, post_activity)

        ltp_count = 0
        ltd_count = 0

        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j:
                    continue

                c = coincidence[i, j]

                # LTP induction
                if c > self.config.induction_threshold:
                    self._induce_ltp(i, j, c)
                    ltp_count += 1

                # LTD induction
                elif c < self.config.depression_threshold and c > 0.01:
                    self._induce_ltd(i, j, c)
                    ltd_count += 1

        return {"ltp": ltp_count, "ltd": ltd_count}

    def _induce_ltp(self, pre_idx: int, post_idx: int, strength: float):
        """Induce early LTP at a synapse"""
        synapse_id = (pre_idx, post_idx)

        # Check if already in LTP
        if synapse_id in self.ltp_state:
            # Reinforce existing LTP
            self.ltp_state[synapse_id]["strength"] += strength * 0.5
            self.ltp_state[synapse_id]["strength"] = min(
                self.ltp_state[synapse_id]["strength"], 1.0
            )
        else:
            # New LTP induction
            self.ltp_state[synapse_id] = {
                "phase": LTPPhase.EARLY_LTP,
                "strength": strength,
                "induced_step": self.current_step,
                "baseline": self.weights[pre_idx, post_idx]
            }
            self.total_ltp_events += 1

        # Apply early potentiation
        delta = strength * self.config.potentiation_rate
        self.weights[pre_idx, post_idx] += delta

        # Create synaptic tag
        self._create_tag(synapse_id, strength, is_ltp=True)

    def _induce_ltd(self, pre_idx: int, post_idx: int, strength: float):
        """Induce LTD at a synapse"""
        synapse_id = (pre_idx, post_idx)

        if synapse_id in self.ltd_state:
            self.ltd_state[synapse_id]["strength"] += strength * 0.5
        else:
            self.ltd_state[synapse_id] = {
                "phase": LTDPhase.EARLY_LTD,
                "strength": strength,
                "induced_step": self.current_step
            }
            self.total_ltd_events += 1

        # Apply depression
        delta = -strength * self.config.depression_rate
        self.weights[pre_idx, post_idx] += delta
        self.weights[pre_idx, post_idx] = max(self.weights[pre_idx, post_idx], -1.0)

        # Create LTD tag
        self._create_tag(synapse_id, strength, is_ltp=False)

    def _create_tag(self, synapse_id: Tuple[int, int], strength: float, is_ltp: bool):
        """Create or update synaptic tag"""
        if synapse_id in self.tags:
            # Refresh tag
            self.tags[synapse_id].tag_strength = max(
                self.tags[synapse_id].tag_strength, strength
            )
            self.tags[synapse_id].created_step = self.current_step
        else:
            self.tags[synapse_id] = SynapticTag(
                synapse_id=synapse_id,
                tag_strength=strength,
                created_step=self.current_step,
                is_potentiation=is_ltp
            )

    def synthesize_proteins(self, neuron_activity: np.ndarray):
        """
        Synthesize proteins in active neurons.

        Strong activity triggers protein synthesis needed for late LTP.
        """
        # Neurons with high activity synthesize proteins
        synthesis = (neuron_activity > 0.8).astype(float) * self.protein_synthesis_rate
        self.protein_pool += synthesis

        # Protein decay
        self.protein_pool *= 0.99

    def capture_proteins(self):
        """
        Tagged synapses capture available proteins.

        Implements synaptic tagging and capture (STC).
        """
        for synapse_id, tag in list(self.tags.items()):
            pre_idx, post_idx = synapse_id

            # Check if tag is still valid (not too old)
            tag_age = self.current_step - tag.created_step
            if tag_age > self.config.early_duration:
                # Tag expired without capturing proteins
                del self.tags[synapse_id]
                continue

            # Capture proteins from post-synaptic neuron
            available = self.protein_pool[post_idx]
            if available > 0.1 and tag.tag_strength > self.config.tagging_threshold:
                # Capture proportional to tag strength
                captured = min(available * tag.tag_strength, available * 0.5)
                self.protein_pool[post_idx] -= captured
                tag.captured_proteins += captured

                # If enough proteins captured, transition to late LTP/LTD
                if tag.captured_proteins > 0.3:
                    if tag.is_potentiation and synapse_id in self.ltp_state:
                        self.ltp_state[synapse_id]["phase"] = LTPPhase.LATE_LTP
                    elif not tag.is_potentiation and synapse_id in self.ltd_state:
                        self.ltd_state[synapse_id]["phase"] = LTDPhase.LATE_LTD

    def tick(self):
        """Advance simulation by one step"""
        self.current_step += 1

        # Decay early LTP
        for synapse_id, state in list(self.ltp_state.items()):
            if state["phase"] == LTPPhase.EARLY_LTP:
                age = self.current_step - state["induced_step"]

                if age > self.config.early_duration:
                    # Check if transitioned to late LTP
                    if synapse_id in self.tags and self.tags[synapse_id].captured_proteins > 0.3:
                        state["phase"] = LTPPhase.LATE_LTP
                    else:
                        # Decay back toward baseline
                        pre, post = synapse_id
                        self.weights[pre, post] *= (1 - self.config.decay_rate)
                        state["strength"] *= (1 - self.config.decay_rate)

                        if state["strength"] < 0.1:
                            state["phase"] = LTPPhase.DECAYED
                            del self.ltp_state[synapse_id]

            elif state["phase"] == LTPPhase.LATE_LTP:
                # Consolidate if strong enough
                if state["strength"] > self.config.consolidation_threshold:
                    state["phase"] = LTPPhase.CONSOLIDATED
                    self.total_consolidated += 1

        # Decay early LTD similarly
        for synapse_id, state in list(self.ltd_state.items()):
            if state["phase"] == LTDPhase.EARLY_LTD:
                age = self.current_step - state["induced_step"]
                if age > self.config.early_duration:
                    if synapse_id in self.tags and self.tags[synapse_id].captured_proteins > 0.3:
                        state["phase"] = LTDPhase.LATE_LTD
                    else:
                        del self.ltd_state[synapse_id]

        # Process protein capture
        self.capture_proteins()

    def consolidate_memory(self, synapse_id: Tuple[int, int]) -> bool:
        """
        Force consolidation of a specific synapse.

        Returns True if consolidation successful.
        """
        if synapse_id not in self.ltp_state:
            return False

        state = self.ltp_state[synapse_id]

        # Need to be in late LTP
        if state["phase"] not in [LTPPhase.LATE_LTP, LTPPhase.CONSOLIDATED]:
            return False

        state["phase"] = LTPPhase.CONSOLIDATED
        self.total_consolidated += 1

        return True

    def get_consolidated_synapses(self) -> List[Tuple[int, int]]:
        """Get list of consolidated (stable memory) synapses"""
        consolidated = []

        for synapse_id, state in self.ltp_state.items():
            if state["phase"] == LTPPhase.CONSOLIDATED:
                consolidated.append(synapse_id)

        return consolidated

    def get_potentiation_level(self, synapse_id: Tuple[int, int]) -> float:
        """
        Get current potentiation level of a synapse.

        Returns change from baseline as percentage.
        """
        pre, post = synapse_id
        current = self.weights[pre, post]
        baseline = self.baseline_weights[pre, post]

        if abs(baseline) < 0.001:
            return 0.0

        return (current - baseline) / abs(baseline)

    def get_statistics(self) -> Dict[str, Any]:
        """Get LTP statistics"""
        ltp_phases = {"early": 0, "late": 0, "consolidated": 0}
        for state in self.ltp_state.values():
            if state["phase"] == LTPPhase.EARLY_LTP:
                ltp_phases["early"] += 1
            elif state["phase"] == LTPPhase.LATE_LTP:
                ltp_phases["late"] += 1
            elif state["phase"] == LTPPhase.CONSOLIDATED:
                ltp_phases["consolidated"] += 1

        return {
            "current_step": self.current_step,
            "total_ltp_events": self.total_ltp_events,
            "total_ltd_events": self.total_ltd_events,
            "total_consolidated": self.total_consolidated,
            "active_ltp": len(self.ltp_state),
            "active_ltd": len(self.ltd_state),
            "active_tags": len(self.tags),
            "ltp_phases": ltp_phases,
            "mean_protein_pool": float(np.mean(self.protein_pool))
        }

    def get_weight_matrix(self) -> np.ndarray:
        """Get current weight matrix"""
        return self.weights.copy()


# ============================================================================
# Memory Reconsolidation
# ============================================================================

class MemoryReconsolidation:
    """
    Implements memory reconsolidation - reactivated memories
    become labile and can be modified.
    """

    def __init__(self, ltp_engine: LTPEngine):
        self.engine = ltp_engine
        self.reactivated: Dict[Tuple[int, int], int] = {}  # synapse -> reactivation step

    def reactivate_memory(self, synapse_id: Tuple[int, int]) -> bool:
        """
        Reactivate a consolidated memory, making it labile.

        Returns True if memory was reactivated.
        """
        if synapse_id not in self.engine.ltp_state:
            return False

        state = self.engine.ltp_state[synapse_id]

        if state["phase"] != LTPPhase.CONSOLIDATED:
            return False

        # Transition back to late LTP (labile)
        state["phase"] = LTPPhase.LATE_LTP
        self.reactivated[synapse_id] = self.engine.current_step

        return True

    def modify_reactivated(
        self,
        synapse_id: Tuple[int, int],
        modification: float
    ) -> bool:
        """
        Modify a reactivated (labile) memory.

        Args:
            synapse_id: Synapse to modify
            modification: Change to apply (-1 to 1)

        Returns True if modification applied.
        """
        if synapse_id not in self.reactivated:
            return False

        pre, post = synapse_id
        self.engine.weights[pre, post] += modification
        self.engine.weights[pre, post] = np.clip(
            self.engine.weights[pre, post], -1.0, 1.0
        )

        # Update LTP state
        if synapse_id in self.engine.ltp_state:
            self.engine.ltp_state[synapse_id]["strength"] += modification

        return True

    def reconsolidate(self, synapse_id: Tuple[int, int]) -> bool:
        """
        Reconsolidate a reactivated memory.

        Returns True if reconsolidation successful.
        """
        if synapse_id not in self.reactivated:
            return False

        # Check if enough time has passed
        reactivation_age = self.engine.current_step - self.reactivated[synapse_id]
        if reactivation_age < self.engine.config.protein_synthesis_delay:
            return False

        # Reconsolidate
        if synapse_id in self.engine.ltp_state:
            self.engine.ltp_state[synapse_id]["phase"] = LTPPhase.CONSOLIDATED

        del self.reactivated[synapse_id]
        return True


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("DENDRITE-X: Long-Term Potentiation Demo")
    print("=" * 50)

    # Create LTP engine
    ltp = LTPEngine(num_neurons=50)

    print(f"\n1. Initial State:")
    stats = ltp.get_statistics()
    print(f"   LTP events: {stats['total_ltp_events']}")

    # Induce LTP with strong co-activity
    print("\n2. Inducing LTP/LTD:")
    for i in range(10):
        pre = np.random.rand(50)
        post = np.random.rand(50)
        result = ltp.induce(pre, post)
        if i == 0:
            print(f"   First induction: LTP={result['ltp']}, LTD={result['ltd']}")

    stats = ltp.get_statistics()
    print(f"   Total LTP events: {stats['total_ltp_events']}")
    print(f"   Total LTD events: {stats['total_ltd_events']}")

    # Simulate protein synthesis
    print("\n3. Protein Synthesis:")
    for i in range(50):
        activity = np.random.rand(50)
        ltp.synthesize_proteins(activity)
        ltp.tick()

    stats = ltp.get_statistics()
    print(f"   Mean protein pool: {stats['mean_protein_pool']:.4f}")
    print(f"   Active tags: {stats['active_tags']}")

    # Continue for consolidation
    print("\n4. Consolidation:")
    for i in range(100):
        ltp.tick()

    stats = ltp.get_statistics()
    print(f"   LTP phases: {stats['ltp_phases']}")
    print(f"   Consolidated: {stats['total_consolidated']}")

    # Get consolidated synapses
    consolidated = ltp.get_consolidated_synapses()
    print(f"\n5. Consolidated Synapses: {len(consolidated)}")

    # Memory reconsolidation
    print("\n6. Memory Reconsolidation:")
    recon = MemoryReconsolidation(ltp)

    if consolidated:
        synapse = consolidated[0]
        before = ltp.get_potentiation_level(synapse)

        recon.reactivate_memory(synapse)
        recon.modify_reactivated(synapse, 0.2)

        for i in range(60):
            ltp.tick()

        recon.reconsolidate(synapse)
        after = ltp.get_potentiation_level(synapse)

        print(f"   Synapse {synapse}:")
        print(f"   Before modification: {before:.4f}")
        print(f"   After modification: {after:.4f}")

    # Final statistics
    print("\n7. Final Statistics:")
    stats = ltp.get_statistics()
    print(f"   Total LTP events: {stats['total_ltp_events']}")
    print(f"   Total consolidated: {stats['total_consolidated']}")
    print(f"   Active LTP: {stats['active_ltp']}")

    print("\n" + "=" * 50)
    print("LTP Engine ready for memory consolidation!")
