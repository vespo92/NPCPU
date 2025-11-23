"""
DENDRITE-X: Advanced Attention Layers for Consciousness

Multi-head self-attention mechanisms inspired by transformer architecture
and biological neural attention systems.

Part of AGENT DENDRITE-X (Agent #3) - Advanced Neural Architecture
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import math


# ============================================================================
# Attention Configuration
# ============================================================================

@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms"""
    dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    use_bias: bool = True
    use_causal_mask: bool = False
    temperature: float = 1.0  # Attention temperature for sharpening/smoothing
    attention_type: str = "scaled_dot_product"  # or "additive", "multiplicative"


class AttentionType(Enum):
    """Types of attention mechanisms"""
    SCALED_DOT_PRODUCT = "scaled_dot_product"
    ADDITIVE = "additive"  # Bahdanau attention
    MULTIPLICATIVE = "multiplicative"  # Luong attention
    RELATIVE = "relative"  # Relative positional attention
    LOCAL = "local"  # Windowed local attention


# ============================================================================
# Core Attention Mechanisms
# ============================================================================

class AttentionMechanism:
    """
    Reusable attention mechanism component.

    Implements scaled dot-product attention with configurable heads.
    Can be used standalone or as part of larger neural architectures.

    Features:
    - Multi-head attention support
    - Configurable attention patterns
    - Attention weight visualization
    - Memory-efficient implementation

    Example:
        attention = AttentionMechanism(dim=64, num_heads=4)

        # Apply self-attention
        queries = np.random.randn(batch_size, seq_len, 64)
        keys = values = queries

        output, weights = attention(queries, keys, values)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_bias: bool = True,
        temperature: float = 1.0
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.use_bias = use_bias
        self.temperature = temperature

        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        # Initialize projection matrices
        self._init_weights()

        # Attention statistics for analysis
        self.attention_history: List[np.ndarray] = []
        self.max_history_size = 100

    def _init_weights(self):
        """Initialize Q, K, V projection weights using Xavier initialization"""
        scale = np.sqrt(2.0 / (self.dim + self.head_dim))

        self.W_q = np.random.randn(self.dim, self.dim) * scale
        self.W_k = np.random.randn(self.dim, self.dim) * scale
        self.W_v = np.random.randn(self.dim, self.dim) * scale
        self.W_o = np.random.randn(self.dim, self.dim) * scale

        if self.use_bias:
            self.b_q = np.zeros(self.dim)
            self.b_k = np.zeros(self.dim)
            self.b_v = np.zeros(self.dim)
            self.b_o = np.zeros(self.dim)
        else:
            self.b_q = self.b_k = self.b_v = self.b_o = None

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-10)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last dimension into (num_heads, head_dim)"""
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Reshape: (batch, seq, dim) -> (batch, seq, num_heads, head_dim)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, num_heads, seq, head_dim)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Merge heads back into single dimension"""
        batch_size = x.shape[0]
        seq_len = x.shape[2]

        # Transpose: (batch, num_heads, seq, head_dim) -> (batch, seq, num_heads, head_dim)
        x = x.transpose(0, 2, 1, 3)

        # Reshape: (batch, seq, dim)
        return x.reshape(batch_size, seq_len, self.dim)

    def _apply_dropout(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Apply dropout during training"""
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, x.shape)
            return x * mask / (1 - self.dropout)
        return x

    def scaled_dot_product_attention(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention.

        Args:
            queries: Query vectors (batch, heads, seq_q, head_dim)
            keys: Key vectors (batch, heads, seq_k, head_dim)
            values: Value vectors (batch, heads, seq_v, head_dim)
            mask: Optional attention mask
            training: Whether in training mode (for dropout)

        Returns:
            Tuple of (attended values, attention weights)
        """
        d_k = queries.shape[-1]

        # Compute attention scores
        scores = np.matmul(queries, keys.transpose(0, 1, 3, 2))
        scores = scores / (np.sqrt(d_k) * self.temperature)

        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)

        # Softmax normalization
        attention_weights = self._softmax(scores, axis=-1)

        # Apply dropout to attention weights
        attention_weights = self._apply_dropout(attention_weights, training)

        # Compute attended values
        attended = np.matmul(attention_weights, values)

        return attended, attention_weights

    def __call__(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
        return_attention: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply multi-head attention.

        Args:
            queries: Query tensor (batch, seq_q, dim)
            keys: Key tensor (batch, seq_k, dim)
            values: Value tensor (batch, seq_v, dim)
            mask: Optional attention mask
            training: Whether in training mode
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights) if return_attention else (output, None)
        """
        # Ensure 3D input
        if queries.ndim == 2:
            queries = queries[np.newaxis, :, :]
            keys = keys[np.newaxis, :, :]
            values = values[np.newaxis, :, :]

        # Project to Q, K, V
        Q = queries @ self.W_q
        K = keys @ self.W_k
        V = values @ self.W_v

        if self.use_bias:
            Q = Q + self.b_q
            K = K + self.b_k
            V = V + self.b_v

        # Split into heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Apply attention
        attended, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask, training
        )

        # Merge heads
        output = self._merge_heads(attended)

        # Output projection
        output = output @ self.W_o
        if self.use_bias:
            output = output + self.b_o

        # Store attention history for analysis
        if return_attention and len(self.attention_history) < self.max_history_size:
            self.attention_history.append(attention_weights.copy())

        return output, attention_weights if return_attention else None

    def get_attention_entropy(self, weights: np.ndarray) -> float:
        """
        Calculate entropy of attention distribution.

        Higher entropy = more distributed attention
        Lower entropy = more focused attention
        """
        # Flatten to 2D for entropy calculation
        flat_weights = weights.reshape(-1, weights.shape[-1])

        # Calculate entropy for each attention distribution
        entropies = -np.sum(flat_weights * np.log(flat_weights + 1e-10), axis=-1)

        return float(np.mean(entropies))

    def get_attention_focus(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Analyze attention focus patterns.

        Returns:
            Dict with focus metrics:
            - max_attention: Maximum attention weight
            - min_attention: Minimum attention weight
            - sparsity: Fraction of weights below threshold
            - entropy: Attention entropy
        """
        return {
            "max_attention": float(np.max(weights)),
            "min_attention": float(np.min(weights)),
            "sparsity": float(np.mean(weights < 0.1)),
            "entropy": self.get_attention_entropy(weights)
        }


# ============================================================================
# Consciousness-Specific Attention
# ============================================================================

class ConsciousnessAttention(AttentionMechanism):
    """
    Attention mechanism designed for consciousness modeling.

    Extends basic attention with:
    - Salience weighting for emotionally significant stimuli
    - Working memory gating
    - Top-down vs bottom-up attention balance
    - Attention fatigue modeling

    Inspired by Global Workspace Theory and biological attention systems.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        salience_dim: int = 8,
        working_memory_capacity: int = 7,
        **kwargs
    ):
        super().__init__(dim, num_heads, **kwargs)

        self.salience_dim = salience_dim
        self.working_memory_capacity = working_memory_capacity

        # Salience computation network
        self.salience_weights = np.random.randn(dim, salience_dim) * 0.1
        self.salience_output = np.random.randn(salience_dim, 1) * 0.1

        # Top-down attention bias
        self.top_down_bias = np.zeros(dim)

        # Attention fatigue state
        self.attention_fatigue = 0.0
        self.fatigue_recovery_rate = 0.1
        self.fatigue_accumulation_rate = 0.05

        # Working memory gate
        self.wm_gate_weights = np.random.randn(dim, dim) * 0.1

    def compute_salience(self, values: np.ndarray) -> np.ndarray:
        """
        Compute salience scores for input values.

        Salience represents the importance/emotional significance
        of stimuli for consciousness.
        """
        # Project through salience network
        hidden = np.tanh(values @ self.salience_weights)
        salience = 1 / (1 + np.exp(-hidden @ self.salience_output))

        return salience.squeeze(-1)

    def apply_salience_weighting(
        self,
        attention_weights: np.ndarray,
        values: np.ndarray
    ) -> np.ndarray:
        """Weight attention by salience"""
        salience = self.compute_salience(values)

        # Broadcast salience to match attention shape
        if salience.ndim < attention_weights.ndim:
            salience = salience[..., np.newaxis]

        # Combine attention with salience
        weighted = attention_weights * (1 + salience)

        # Renormalize
        return weighted / (np.sum(weighted, axis=-1, keepdims=True) + 1e-10)

    def apply_fatigue(self, attention_weights: np.ndarray) -> np.ndarray:
        """Apply attention fatigue effects"""
        # Fatigue reduces peak attention
        fatigue_factor = 1.0 - (self.attention_fatigue * 0.5)

        # Flatten attention distribution when fatigued
        uniform = np.ones_like(attention_weights) / attention_weights.shape[-1]
        fatigued = fatigue_factor * attention_weights + (1 - fatigue_factor) * uniform

        # Update fatigue
        attention_intensity = np.max(attention_weights)
        self.attention_fatigue = np.clip(
            self.attention_fatigue + self.fatigue_accumulation_rate * attention_intensity,
            0.0, 1.0
        )

        return fatigued

    def recover_fatigue(self):
        """Recover from attention fatigue (call during rest periods)"""
        self.attention_fatigue = np.clip(
            self.attention_fatigue - self.fatigue_recovery_rate,
            0.0, 1.0
        )

    def set_top_down_bias(self, bias: np.ndarray):
        """Set top-down attention bias (goal-directed attention)"""
        self.top_down_bias = bias

    def __call__(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
        apply_salience: bool = True,
        apply_fatigue_effect: bool = True,
        return_attention: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply consciousness-aware attention.

        Extends base attention with salience weighting and fatigue.
        """
        # Apply top-down bias to queries
        biased_queries = queries + self.top_down_bias

        # Get base attention
        output, attention_weights = super().__call__(
            biased_queries, keys, values, mask, training, return_attention=True
        )

        if attention_weights is not None:
            # Apply salience weighting
            if apply_salience:
                # Need to reshape values for salience computation
                if values.ndim == 2:
                    values_3d = values[np.newaxis, :, :]
                else:
                    values_3d = values
                attention_weights = self.apply_salience_weighting(
                    attention_weights, values_3d
                )

            # Apply fatigue
            if apply_fatigue_effect:
                attention_weights = self.apply_fatigue(attention_weights)

        return output, attention_weights if return_attention else None

    def get_consciousness_attention_state(self) -> Dict[str, Any]:
        """Get current state of consciousness attention"""
        return {
            "fatigue_level": self.attention_fatigue,
            "working_memory_capacity": self.working_memory_capacity,
            "has_top_down_bias": bool(np.any(self.top_down_bias != 0)),
            "attention_history_size": len(self.attention_history)
        }


# ============================================================================
# Self-Attention Layer
# ============================================================================

class SelfAttentionLayer:
    """
    Complete self-attention layer with residual connection and layer norm.

    Used as building block for transformer-based consciousness models.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        attention_type: str = "standard"  # or "consciousness"
    ):
        self.dim = dim
        self.use_layer_norm = use_layer_norm

        # Choose attention type
        if attention_type == "consciousness":
            self.attention = ConsciousnessAttention(dim, num_heads, dropout=dropout)
        else:
            self.attention = AttentionMechanism(dim, num_heads, dropout=dropout)

        # Layer normalization parameters
        if use_layer_norm:
            self.ln_gamma = np.ones(dim)
            self.ln_beta = np.zeros(dim)

        self.dropout = dropout

    def layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        normalized = (x - mean) / (std + eps)
        return self.ln_gamma * normalized + self.ln_beta

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply self-attention with residual connection.

        Args:
            x: Input tensor (batch, seq, dim)
            mask: Optional attention mask
            training: Whether in training mode

        Returns:
            Tuple of (output, attention_weights)
        """
        # Pre-norm architecture
        if self.use_layer_norm:
            normed = self.layer_norm(x)
        else:
            normed = x

        # Self-attention
        attended, attention_weights = self.attention(
            normed, normed, normed, mask, training
        )

        # Dropout
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, attended.shape)
            attended = attended * mask / (1 - self.dropout)

        # Residual connection
        output = x + attended

        return output, attention_weights


# ============================================================================
# Cross-Attention Layer
# ============================================================================

class CrossAttentionLayer:
    """
    Cross-attention layer for attending to external context.

    Useful for:
    - Attending to sensory input
    - Memory retrieval attention
    - Integration of multiple information streams
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.use_layer_norm = use_layer_norm

        # Project context to query dimension if needed
        if context_dim != query_dim:
            self.context_projection = np.random.randn(context_dim, query_dim) * 0.1
        else:
            self.context_projection = None

        self.attention = AttentionMechanism(query_dim, num_heads, dropout=dropout)

        # Layer norm for queries
        if use_layer_norm:
            self.ln_gamma = np.ones(query_dim)
            self.ln_beta = np.zeros(query_dim)

        self.dropout = dropout

    def layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        normalized = (x - mean) / (std + eps)
        return self.ln_gamma * normalized + self.ln_beta

    def __call__(
        self,
        queries: np.ndarray,
        context: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply cross-attention.

        Args:
            queries: Query tensor (batch, seq_q, query_dim)
            context: Context tensor (batch, seq_c, context_dim)
            mask: Optional attention mask
            training: Whether in training mode

        Returns:
            Tuple of (output, attention_weights)
        """
        # Project context if dimensions differ
        if self.context_projection is not None:
            context = context @ self.context_projection

        # Pre-norm
        if self.use_layer_norm:
            queries = self.layer_norm(queries)

        # Cross-attention
        attended, attention_weights = self.attention(
            queries, context, context, mask, training
        )

        # Dropout and residual
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, attended.shape)
            attended = attended * mask / (1 - self.dropout)

        output = queries + attended

        return output, attention_weights


# ============================================================================
# Global Workspace Attention
# ============================================================================

class GlobalWorkspaceAttention:
    """
    Attention mechanism implementing Global Workspace Theory.

    In GWT, consciousness arises from a "global workspace" that broadcasts
    information to specialized processors. This attention mechanism:
    - Competes for access to the workspace
    - Broadcasts winning information globally
    - Maintains workspace coherence

    Based on Baars' Global Workspace Theory and Dehaene's Global Neuronal Workspace.
    """

    def __init__(
        self,
        dim: int,
        num_specialists: int = 8,
        workspace_capacity: int = 1,
        broadcast_strength: float = 0.8
    ):
        self.dim = dim
        self.num_specialists = num_specialists
        self.workspace_capacity = workspace_capacity
        self.broadcast_strength = broadcast_strength

        # Specialist attention heads
        self.specialist_attention = AttentionMechanism(
            dim, num_heads=num_specialists
        )

        # Workspace state
        self.workspace_content: Optional[np.ndarray] = None
        self.workspace_coherence: float = 0.0

        # Competition weights
        self.competition_weights = np.random.randn(dim, 1) * 0.1

        # Broadcast history
        self.broadcast_history: List[Dict[str, Any]] = []

    def compute_competition_scores(self, candidates: np.ndarray) -> np.ndarray:
        """Compute competition scores for workspace access"""
        # Project to scalar competition score
        scores = candidates @ self.competition_weights

        # Softmax competition
        exp_scores = np.exp(scores - np.max(scores))
        competition_probs = exp_scores / (np.sum(exp_scores) + 1e-10)

        return competition_probs.squeeze(-1)

    def select_for_workspace(
        self,
        candidates: np.ndarray,
        competition_scores: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Select winner(s) for workspace"""
        # Get top-k indices
        top_indices = np.argsort(competition_scores)[-self.workspace_capacity:]

        # Average winners for workspace
        winners = candidates[top_indices]
        workspace_content = np.mean(winners, axis=0)

        return workspace_content, int(top_indices[-1])

    def broadcast(
        self,
        workspace_content: np.ndarray,
        specialists: np.ndarray
    ) -> np.ndarray:
        """Broadcast workspace content to all specialists"""
        # Each specialist receives workspace content modulated by their state
        broadcast = workspace_content[np.newaxis, :] * self.broadcast_strength

        # Add broadcast to specialist representations
        updated = specialists + broadcast

        return updated

    def __call__(
        self,
        inputs: np.ndarray,
        return_competition: bool = False
    ) -> Dict[str, Any]:
        """
        Run global workspace attention.

        Args:
            inputs: Input representations (batch, seq, dim)
            return_competition: Whether to return competition details

        Returns:
            Dict with:
            - output: Updated representations
            - workspace: Current workspace content
            - winner_idx: Index of winning input
            - coherence: Workspace coherence measure
        """
        # Flatten for competition
        if inputs.ndim == 3:
            batch, seq, dim = inputs.shape
            flat_inputs = inputs.reshape(-1, dim)
        else:
            flat_inputs = inputs

        # Compute competition
        competition_scores = self.compute_competition_scores(flat_inputs)

        # Select for workspace
        workspace_content, winner_idx = self.select_for_workspace(
            flat_inputs, competition_scores
        )

        # Broadcast to all
        output = self.broadcast(workspace_content, flat_inputs)

        # Update workspace state
        if self.workspace_content is not None:
            # Compute coherence as similarity to previous content
            similarity = np.dot(workspace_content, self.workspace_content)
            self.workspace_coherence = float(similarity / (
                np.linalg.norm(workspace_content) *
                np.linalg.norm(self.workspace_content) + 1e-10
            ))
        else:
            self.workspace_coherence = 1.0

        self.workspace_content = workspace_content

        # Reshape output if needed
        if inputs.ndim == 3:
            output = output.reshape(batch, seq, dim)

        # Store broadcast event
        self.broadcast_history.append({
            "winner_idx": winner_idx,
            "competition_max": float(np.max(competition_scores)),
            "coherence": self.workspace_coherence
        })

        result = {
            "output": output,
            "workspace": workspace_content,
            "winner_idx": winner_idx,
            "coherence": self.workspace_coherence
        }

        if return_competition:
            result["competition_scores"] = competition_scores

        return result

    def get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state"""
        return {
            "has_content": self.workspace_content is not None,
            "coherence": self.workspace_coherence,
            "broadcast_count": len(self.broadcast_history),
            "avg_coherence": np.mean([b["coherence"] for b in self.broadcast_history]) if self.broadcast_history else 0.0
        }


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding:
    """
    Positional encoding for sequence data.

    Supports:
    - Sinusoidal (fixed) encoding
    - Learned positional embeddings
    - Relative positional encoding
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 1000,
        encoding_type: str = "sinusoidal"
    ):
        self.dim = dim
        self.max_len = max_len
        self.encoding_type = encoding_type

        if encoding_type == "sinusoidal":
            self.encodings = self._create_sinusoidal_encodings()
        elif encoding_type == "learned":
            self.encodings = np.random.randn(max_len, dim) * 0.1
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

    def _create_sinusoidal_encodings(self) -> np.ndarray:
        """Create sinusoidal positional encodings"""
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim))

        encodings = np.zeros((self.max_len, self.dim))
        encodings[:, 0::2] = np.sin(position * div_term)
        encodings[:, 1::2] = np.cos(position * div_term)

        return encodings

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input"""
        seq_len = x.shape[-2]
        return x + self.encodings[:seq_len]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("DENDRITE-X: Attention Layers Demo")
    print("=" * 50)

    # Test basic attention
    print("\n1. Basic Multi-Head Attention:")
    attention = AttentionMechanism(dim=64, num_heads=4)

    queries = np.random.randn(2, 10, 64)  # batch=2, seq=10, dim=64
    output, weights = attention(queries, queries, queries)

    print(f"   Input shape: {queries.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {weights.shape}")
    print(f"   Attention entropy: {attention.get_attention_entropy(weights):.4f}")

    # Test consciousness attention
    print("\n2. Consciousness Attention:")
    consciousness_attn = ConsciousnessAttention(dim=64, num_heads=4)

    output, weights = consciousness_attn(queries, queries, queries)
    state = consciousness_attn.get_consciousness_attention_state()

    print(f"   Fatigue level: {state['fatigue_level']:.4f}")
    print(f"   Working memory capacity: {state['working_memory_capacity']}")

    # Test global workspace
    print("\n3. Global Workspace Attention:")
    gw_attention = GlobalWorkspaceAttention(dim=64, num_specialists=8)

    result = gw_attention(queries, return_competition=True)

    print(f"   Winner index: {result['winner_idx']}")
    print(f"   Workspace coherence: {result['coherence']:.4f}")
    print(f"   Competition max: {np.max(result['competition_scores']):.4f}")

    # Test self-attention layer
    print("\n4. Self-Attention Layer:")
    sa_layer = SelfAttentionLayer(dim=64, num_heads=4)

    output, weights = sa_layer(queries)
    print(f"   Output shape: {output.shape}")

    print("\n" + "=" * 50)
    print("Attention layers ready for consciousness modeling!")
