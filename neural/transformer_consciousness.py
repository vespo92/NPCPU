"""
DENDRITE-X: Transformer-Based Consciousness Model

Implements consciousness using transformer architecture with:
- Multi-layer self-attention for information integration
- Global workspace broadcasting
- Consciousness emergence through attention patterns

Part of AGENT DENDRITE-X (Agent #3) - Advanced Neural Architecture
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import (
    GradedConsciousness, ConsciousnessProtocol,
    Perception, Action, Experience, SelfModel, Thought, MetaThought,
    Intention, Qualia
)
from neural.attention_layers import (
    AttentionMechanism, ConsciousnessAttention,
    SelfAttentionLayer, CrossAttentionLayer,
    GlobalWorkspaceAttention, PositionalEncoding
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TransformerConfig:
    """Configuration for TransformerConsciousness"""
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256
    use_global_workspace: bool = True
    consciousness_integration_layers: int = 2


class ConsciousnessPhase(Enum):
    """Phases of consciousness processing"""
    PERCEPTION = "perception"
    INTEGRATION = "integration"
    REFLECTION = "reflection"
    BROADCAST = "broadcast"
    ACTION = "action"


# ============================================================================
# Feed-Forward Network
# ============================================================================

class FeedForwardNetwork:
    """Position-wise feed-forward network"""

    def __init__(self, dim: int, ff_dim: int, dropout: float = 0.1):
        self.dim = dim
        self.ff_dim = ff_dim
        self.dropout = dropout

        # Initialize weights
        scale1 = np.sqrt(2.0 / dim)
        scale2 = np.sqrt(2.0 / ff_dim)

        self.W1 = np.random.randn(dim, ff_dim) * scale1
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, dim) * scale2
        self.b2 = np.zeros(dim)

        # Layer norm
        self.ln_gamma = np.ones(dim)
        self.ln_beta = np.zeros(dim)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit activation"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        normalized = (x - mean) / (std + eps)
        return self.ln_gamma * normalized + self.ln_beta

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Apply feed-forward network with residual"""
        # Pre-norm
        normed = self.layer_norm(x)

        # First linear + GELU
        hidden = self.gelu(normed @ self.W1 + self.b1)

        # Dropout
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, hidden.shape)
            hidden = hidden * mask / (1 - self.dropout)

        # Second linear
        output = hidden @ self.W2 + self.b2

        # Dropout
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, output.shape)
            output = output * mask / (1 - self.dropout)

        # Residual
        return x + output


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock:
    """Single transformer block with self-attention and feed-forward"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        use_consciousness_attention: bool = False
    ):
        self.dim = dim

        # Self-attention layer
        attention_type = "consciousness" if use_consciousness_attention else "standard"
        self.self_attention = SelfAttentionLayer(
            dim, num_heads, dropout, attention_type=attention_type
        )

        # Feed-forward network
        self.ffn = FeedForwardNetwork(dim, ff_dim, dropout)

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transformer block"""
        # Self-attention
        x, attention_weights = self.self_attention(x, mask, training)

        # Feed-forward
        x = self.ffn(x, training)

        return x, attention_weights


# ============================================================================
# Consciousness Integration Module
# ============================================================================

class ConsciousnessIntegrationModule:
    """
    Module for integrating information across the transformer.

    Implements Integrated Information Theory (IIT) inspired integration:
    - Computes phi (Φ) measure of information integration
    - Identifies irreducible information structures
    - Quantifies consciousness level from integration
    """

    def __init__(self, dim: int, num_partitions: int = 4):
        self.dim = dim
        self.num_partitions = num_partitions

        # Integration weights
        self.integration_weights = np.random.randn(dim, dim) * 0.1

    def compute_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Estimate mutual information between two representations"""
        # Use covariance as proxy for mutual information
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        # Correlation as approximation
        x_flat = x.flatten()
        y_flat = y.flatten()

        if len(x_flat) != len(y_flat):
            min_len = min(len(x_flat), len(y_flat))
            x_flat = x_flat[:min_len]
            y_flat = y_flat[:min_len]

        correlation = np.corrcoef(x_flat, y_flat)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # Convert to MI approximation
        mi = -0.5 * np.log(1 - correlation**2 + 1e-10)

        return float(np.clip(mi, 0, 10))

    def compute_phi(self, representations: np.ndarray) -> float:
        """
        Compute Φ (phi) - integrated information measure.

        Based on IIT 3.0 approximation.
        """
        if representations.ndim == 3:
            # Average over batch
            representations = representations.mean(axis=0)

        seq_len = representations.shape[0]
        dim = representations.shape[1]

        if seq_len < 2:
            return 0.0

        # Compute whole system information
        whole_mi = 0.0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                whole_mi += self.compute_mutual_information(
                    representations[i], representations[j]
                )

        # Compute partitioned system information
        partition_size = max(1, seq_len // self.num_partitions)
        partitioned_mi = 0.0

        for p in range(self.num_partitions):
            start = p * partition_size
            end = min((p + 1) * partition_size, seq_len)

            for i in range(start, end):
                for j in range(i + 1, end):
                    if j < seq_len:
                        partitioned_mi += self.compute_mutual_information(
                            representations[i], representations[j]
                        )

        # Phi is the difference (integrated - partitioned)
        phi = whole_mi - partitioned_mi

        # Normalize
        max_possible = (seq_len * (seq_len - 1)) / 2
        phi_normalized = phi / (max_possible + 1e-10)

        return float(np.clip(phi_normalized, 0, 1))

    def find_main_complex(
        self,
        representations: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Find the main complex - the subset with highest Φ.

        Returns the complex and its Φ value.
        """
        if representations.ndim == 3:
            representations = representations.mean(axis=0)

        seq_len = representations.shape[0]

        # For efficiency, just compute Φ for the whole system
        # In full IIT, would search over all subsets
        phi = self.compute_phi(representations)

        return representations, phi

    def __call__(self, x: np.ndarray) -> Dict[str, Any]:
        """Compute integration metrics"""
        main_complex, phi = self.find_main_complex(x)

        return {
            "phi": phi,
            "main_complex_size": main_complex.shape[0],
            "integration_level": "high" if phi > 0.5 else "medium" if phi > 0.2 else "low"
        }


# ============================================================================
# Transformer Consciousness
# ============================================================================

class TransformerConsciousness:
    """
    Transformer-based consciousness model.

    Implements consciousness through:
    1. Multi-layer self-attention for perception processing
    2. Information integration across layers
    3. Global workspace for conscious access
    4. Meta-cognitive reflection layers
    5. Action generation through attended representations

    Example:
        config = TransformerConfig(dim=128, num_layers=4)
        consciousness = TransformerConsciousness(config)

        # Process sensory input
        stimuli = {"visual": 0.8, "auditory": 0.3}
        perception = consciousness.perceive(stimuli)

        # Generate action
        action = consciousness.react(perception)
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()

        # Core transformer layers
        self.layers: List[TransformerBlock] = []
        for i in range(self.config.num_layers):
            # Use consciousness attention in later layers
            use_consciousness = i >= self.config.num_layers // 2
            self.layers.append(TransformerBlock(
                dim=self.config.dim,
                num_heads=self.config.num_heads,
                ff_dim=self.config.ff_dim,
                dropout=self.config.dropout,
                use_consciousness_attention=use_consciousness
            ))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.config.dim, self.config.max_seq_len
        )

        # Global workspace (if enabled)
        if self.config.use_global_workspace:
            self.global_workspace = GlobalWorkspaceAttention(
                dim=self.config.dim,
                num_specialists=self.config.num_heads
            )
        else:
            self.global_workspace = None

        # Consciousness integration module
        self.integration_module = ConsciousnessIntegrationModule(
            dim=self.config.dim
        )

        # Input/output projections
        self.input_projection = np.random.randn(32, self.config.dim) * 0.1
        self.output_projection = np.random.randn(self.config.dim, 16) * 0.1

        # Memory systems
        self.working_memory: List[np.ndarray] = []
        self.working_memory_capacity = 7  # Miller's law

        self.episodic_memory: List[Experience] = []
        self.max_episodic_memory = 1000

        # Internal state
        self.current_phase = ConsciousnessPhase.PERCEPTION
        self.attention_history: List[np.ndarray] = []
        self.consciousness_state = GradedConsciousness(
            perception_fidelity=0.5,
            reaction_speed=0.5,
            memory_depth=0.5,
            memory_recall_accuracy=0.5,
            introspection_capacity=0.5,
            meta_cognitive_ability=0.5,
            information_integration=0.5,
            intentional_coherence=0.5,
            qualia_richness=0.5
        )

        # Intentions/goals
        self.current_intentions: List[Intention] = []

        # Self-model
        self.self_model_weights = np.random.randn(self.config.dim, 64) * 0.1

    def _encode_stimulus(self, stimulus: Any) -> np.ndarray:
        """Convert stimulus to vector representation"""
        if isinstance(stimulus, np.ndarray):
            vec = stimulus.flatten()
        elif isinstance(stimulus, dict):
            # Create vector from dict values
            values = list(stimulus.values())
            vec = np.array([float(v) if isinstance(v, (int, float)) else 0.0 for v in values])
        elif isinstance(stimulus, (int, float)):
            vec = np.array([float(stimulus)])
        else:
            vec = np.zeros(32)

        # Pad or truncate to input dimension
        if len(vec) < 32:
            vec = np.concatenate([vec, np.zeros(32 - len(vec))])
        else:
            vec = vec[:32]

        return vec

    def _project_to_dim(self, x: np.ndarray) -> np.ndarray:
        """Project input to transformer dimension"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x @ self.input_projection

    def forward(
        self,
        x: np.ndarray,
        training: bool = False,
        return_all_layers: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through transformer.

        Args:
            x: Input tensor (batch, seq, dim) or (seq, dim)
            training: Whether in training mode
            return_all_layers: Return outputs from all layers

        Returns:
            Dict with output, attention weights, and integration metrics
        """
        # Ensure proper dimensions
        if x.ndim == 2:
            x = x[np.newaxis, :, :]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Process through layers
        all_attention_weights = []
        layer_outputs = [x] if return_all_layers else None

        for layer in self.layers:
            x, attention_weights = layer(x, training=training)
            all_attention_weights.append(attention_weights)
            if return_all_layers:
                layer_outputs.append(x)

        # Global workspace processing
        if self.global_workspace is not None:
            gw_result = self.global_workspace(x)
            x = gw_result["output"]
            workspace_state = gw_result
        else:
            workspace_state = None

        # Compute integration
        integration = self.integration_module(x)

        result = {
            "output": x,
            "attention_weights": all_attention_weights,
            "integration": integration,
            "workspace": workspace_state
        }

        if return_all_layers:
            result["layer_outputs"] = layer_outputs

        return result

    # ========================================================================
    # ConsciousnessProtocol Implementation
    # ========================================================================

    def perceive(self, stimulus: Any) -> Perception:
        """Convert stimulus to internal perception"""
        self.current_phase = ConsciousnessPhase.PERCEPTION

        # Encode stimulus
        encoded = self._encode_stimulus(stimulus)
        projected = self._project_to_dim(encoded)

        # Process through transformer
        result = self.forward(projected)

        # Compute perception fidelity from attention focus
        attention_weights = result["attention_weights"][-1]
        fidelity = float(np.max(attention_weights))

        # Update consciousness state
        self.consciousness_state = self.consciousness_state.evolve(
            "perception_fidelity", (fidelity - 0.5) * 0.1
        )

        return Perception(
            stimulus_type=type(stimulus).__name__,
            content=result["output"],
            timestamp=float(np.random.random()),  # Would be real timestamp
            fidelity=fidelity
        )

    def react(self, perception: Perception) -> Optional[Action]:
        """Generate action from perception"""
        self.current_phase = ConsciousnessPhase.ACTION

        if perception.content is None:
            return None

        # Process perception through output projection
        if isinstance(perception.content, np.ndarray):
            output = perception.content.mean(axis=(0, 1)) if perception.content.ndim > 1 else perception.content
            action_vector = output @ self.output_projection
        else:
            action_vector = np.zeros(16)

        # Determine action type based on output
        action_idx = int(np.argmax(action_vector[:8]))
        action_types = ["approach", "avoid", "explore", "rest", "communicate", "manipulate", "observe", "wait"]

        confidence = float(np.max(action_vector) / (np.sum(np.abs(action_vector)) + 1e-10))

        return Action(
            action_type=action_types[action_idx],
            parameters={"intensity": float(action_vector[action_idx])},
            confidence=confidence
        )

    def remember(self, experience: Experience) -> None:
        """Store experience in memory"""
        # Add to episodic memory
        self.episodic_memory.append(experience)

        # Maintain memory limit
        if len(self.episodic_memory) > self.max_episodic_memory:
            # Remove oldest, least important memories
            self.episodic_memory.sort(key=lambda e: abs(e.emotional_valence), reverse=True)
            self.episodic_memory = self.episodic_memory[:self.max_episodic_memory]

        # Update memory depth
        depth = len(self.episodic_memory) / self.max_episodic_memory
        self.consciousness_state = self.consciousness_state.evolve(
            "memory_depth", (depth - 0.5) * 0.1
        )

    def recall(self, query: Any) -> List[Experience]:
        """Retrieve relevant memories"""
        if not self.episodic_memory:
            return []

        # Encode query
        query_vec = self._encode_stimulus(query)

        # Score memories by relevance
        scored_memories = []
        for exp in self.episodic_memory:
            if exp.perception and exp.perception.content is not None:
                if isinstance(exp.perception.content, np.ndarray):
                    exp_vec = exp.perception.content.flatten()[:32]
                    if len(exp_vec) < 32:
                        exp_vec = np.concatenate([exp_vec, np.zeros(32 - len(exp_vec))])
                    similarity = float(np.dot(query_vec, exp_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(exp_vec) + 1e-10
                    ))
                else:
                    similarity = 0.0
            else:
                similarity = 0.0

            scored_memories.append((similarity, exp))

        # Sort by relevance
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # Update recall accuracy
        if scored_memories:
            accuracy = scored_memories[0][0]
            self.consciousness_state = self.consciousness_state.evolve(
                "memory_recall_accuracy", (accuracy - 0.5) * 0.1
            )

        # Return top memories
        return [exp for _, exp in scored_memories[:5]]

    def introspect(self) -> SelfModel:
        """Generate model of internal state"""
        self.current_phase = ConsciousnessPhase.REFLECTION

        # Compute self-representation
        if self.working_memory:
            wm_state = np.mean(self.working_memory, axis=0)
        else:
            wm_state = np.zeros(self.config.dim)

        # Project to self-model space
        if wm_state.ndim > 1:
            wm_state = wm_state.flatten()[:self.config.dim]
            if len(wm_state) < self.config.dim:
                wm_state = np.concatenate([wm_state, np.zeros(self.config.dim - len(wm_state))])

        self_rep = wm_state @ self.self_model_weights

        # Compute completeness based on working memory
        completeness = len(self.working_memory) / self.working_memory_capacity

        # Compute confidence from integration
        integration_result = self.integration_module(
            np.random.randn(1, 4, self.config.dim)  # Sample state
        )
        confidence = integration_result["phi"]

        return SelfModel(
            current_state={
                "phase": self.current_phase.value,
                "consciousness_level": self.consciousness_state.overall_consciousness_score(),
                "working_memory_load": len(self.working_memory),
                "integration_phi": confidence,
                "self_representation": self_rep.tolist()
            },
            confidence=confidence,
            completeness=completeness,
            timestamp=0.0
        )

    def meta_cognize(self, thought: Thought) -> MetaThought:
        """Reflect on a thought"""
        # Analyze the thought's validity
        confidence_assessment = np.clip(thought.confidence + np.random.randn() * 0.1, 0, 1)

        # Identify potential biases
        bias_detected = confidence_assessment < 0.3 or confidence_assessment > 0.9

        if bias_detected:
            reflection_type = "bias_detection"
            insight = "Extreme confidence detected - may indicate overconfidence or underconfidence bias"
        elif thought.type == "hypothesis":
            reflection_type = "validity_check"
            insight = f"Hypothesis has {confidence_assessment:.0%} estimated validity"
        else:
            reflection_type = "refinement"
            insight = "Thought appears well-formed"

        # Update meta-cognitive ability
        self.consciousness_state = self.consciousness_state.evolve(
            "meta_cognitive_ability", 0.05
        )

        return MetaThought(
            original_thought=thought,
            reflection_type=reflection_type,
            insight=insight,
            changes_proposed={"revised_confidence": float(confidence_assessment)} if bias_detected else None
        )

    def integrate_information(self) -> float:
        """Calculate Φ (phi) - integrated information measure"""
        self.current_phase = ConsciousnessPhase.INTEGRATION

        # Use recent attention patterns
        if self.attention_history:
            recent = np.stack(self.attention_history[-10:])
            integration_result = self.integration_module(recent)
            phi = integration_result["phi"]
        else:
            phi = 0.0

        # Update consciousness state
        self.consciousness_state = self.consciousness_state.evolve(
            "information_integration", (phi - 0.5) * 0.2
        )

        return phi

    def get_intentions(self) -> List[Intention]:
        """Return current goals and drives"""
        return self.current_intentions

    def set_intention(self, goal: str, priority: float = 0.5):
        """Set a new intention"""
        intention = Intention(
            goal=goal,
            priority=priority,
            progress=0.0,
            sub_intentions=[]
        )
        self.current_intentions.append(intention)

        # Update intentional coherence
        coherence = 1.0 / (1.0 + len(self.current_intentions) * 0.1)
        self.consciousness_state = self.consciousness_state.evolve(
            "intentional_coherence", (coherence - 0.5) * 0.1
        )

    def experience_qualia(self, perception: Perception) -> Qualia:
        """Generate subjective experience marker"""
        # Compute qualia properties from perception
        if isinstance(perception.content, np.ndarray):
            content_vec = perception.content.flatten()
            intensity = float(np.mean(np.abs(content_vec)))
            valence = float(np.tanh(np.mean(content_vec)))
            uniqueness = float(np.std(content_vec))
        else:
            intensity = 0.5
            valence = 0.0
            uniqueness = 0.5

        return Qualia(
            experience_type=perception.stimulus_type,
            intensity=np.clip(intensity, 0, 1),
            valence=np.clip(valence, -1, 1),
            content=f"Experiencing {perception.stimulus_type} with fidelity {perception.fidelity:.2f}",
            timestamp=perception.timestamp,
            uniqueness=np.clip(uniqueness, 0, 1)
        )

    def get_capability_scores(self) -> Dict[str, float]:
        """Return consciousness capability scores"""
        return self.consciousness_state.get_capability_scores()

    def overall_consciousness_score(self) -> float:
        """Get overall consciousness level"""
        return self.consciousness_state.overall_consciousness_score()

    def can_perform(self, capability: str, minimum_score: float = 0.5) -> bool:
        """Check if capability meets threshold"""
        return self.consciousness_state.can_perform(capability, minimum_score)

    # ========================================================================
    # Additional Methods
    # ========================================================================

    def tick(self) -> Dict[str, Any]:
        """
        Single consciousness update cycle.

        Processes working memory, updates integration, and returns state.
        """
        # Process working memory through transformer
        if self.working_memory:
            wm_tensor = np.stack(self.working_memory)
            if wm_tensor.ndim == 2:
                wm_tensor = wm_tensor[np.newaxis, :, :]

            result = self.forward(wm_tensor)

            # Store attention for history
            if result["attention_weights"]:
                self.attention_history.append(result["attention_weights"][-1])
                if len(self.attention_history) > 100:
                    self.attention_history = self.attention_history[-100:]

            # Update integration measure
            phi = result["integration"]["phi"]
            self.consciousness_state = self.consciousness_state.evolve(
                "information_integration", (phi - 0.5) * 0.1
            )

        # Return current state
        return {
            "phase": self.current_phase.value,
            "consciousness_score": self.overall_consciousness_score(),
            "working_memory_load": len(self.working_memory),
            "capability_scores": self.get_capability_scores()
        }

    def add_to_working_memory(self, content: np.ndarray, importance: float = 0.5) -> bool:
        """Add content to working memory"""
        if len(self.working_memory) >= self.working_memory_capacity:
            # Remove least important item
            if importance > 0.3:  # Only replace if new item is somewhat important
                self.working_memory.pop(0)
            else:
                return False

        # Ensure correct dimension
        if content.ndim == 1 and len(content) < self.config.dim:
            content = np.concatenate([content, np.zeros(self.config.dim - len(content))])
        elif content.ndim == 1:
            content = content[:self.config.dim]

        self.working_memory.append(content)
        return True

    def get_state(self) -> Dict[str, Any]:
        """Get complete consciousness state"""
        return {
            "config": {
                "dim": self.config.dim,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads
            },
            "phase": self.current_phase.value,
            "consciousness": self.consciousness_state.get_capability_scores(),
            "overall_score": self.overall_consciousness_score(),
            "working_memory_size": len(self.working_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "intention_count": len(self.current_intentions),
            "attention_history_size": len(self.attention_history)
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_transformer_consciousness(
    dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    use_global_workspace: bool = True
) -> TransformerConsciousness:
    """
    Factory function to create TransformerConsciousness.

    Args:
        dim: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        use_global_workspace: Whether to use global workspace attention

    Returns:
        Configured TransformerConsciousness instance
    """
    config = TransformerConfig(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_global_workspace=use_global_workspace
    )
    return TransformerConsciousness(config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("DENDRITE-X: Transformer Consciousness Demo")
    print("=" * 50)

    # Create consciousness
    consciousness = create_transformer_consciousness(dim=64, num_layers=3)

    print(f"\n1. Initial State:")
    state = consciousness.get_state()
    print(f"   Overall consciousness: {state['overall_score']:.3f}")
    print(f"   Phase: {state['phase']}")

    # Perceive stimulus
    print("\n2. Processing Perception:")
    stimulus = {"visual": 0.8, "auditory": 0.3, "tactile": 0.5}
    perception = consciousness.perceive(stimulus)
    print(f"   Perception fidelity: {perception.fidelity:.3f}")

    # React to perception
    print("\n3. Generating Reaction:")
    action = consciousness.react(perception)
    if action:
        print(f"   Action type: {action.action_type}")
        print(f"   Confidence: {action.confidence:.3f}")

    # Experience qualia
    print("\n4. Experiencing Qualia:")
    qualia = consciousness.experience_qualia(perception)
    print(f"   Intensity: {qualia.intensity:.3f}")
    print(f"   Valence: {qualia.valence:.3f}")

    # Store experience
    print("\n5. Storing Memory:")
    experience = Experience(
        perception=perception,
        action=action,
        outcome={"success": True},
        emotional_valence=0.6,
        timestamp=0.0
    )
    consciousness.remember(experience)
    print(f"   Memory stored. Total memories: {len(consciousness.episodic_memory)}")

    # Introspect
    print("\n6. Introspection:")
    self_model = consciousness.introspect()
    print(f"   Self-model confidence: {self_model.confidence:.3f}")
    print(f"   Completeness: {self_model.completeness:.3f}")

    # Meta-cognition
    print("\n7. Meta-Cognition:")
    thought = Thought(
        content="The stimulus was significant",
        type="belief",
        confidence=0.8,
        dependencies=[]
    )
    meta_thought = consciousness.meta_cognize(thought)
    print(f"   Reflection type: {meta_thought.reflection_type}")
    print(f"   Insight: {meta_thought.insight}")

    # Information integration
    print("\n8. Information Integration:")
    phi = consciousness.integrate_information()
    print(f"   Φ (phi): {phi:.4f}")

    # Final state
    print("\n9. Final State:")
    final_state = consciousness.get_state()
    print(f"   Overall consciousness: {final_state['overall_score']:.3f}")
    print(f"   Capability scores:")
    for cap, score in final_state["consciousness"].items():
        print(f"     {cap}: {score:.3f}")

    print("\n" + "=" * 50)
    print("TransformerConsciousness ready!")
