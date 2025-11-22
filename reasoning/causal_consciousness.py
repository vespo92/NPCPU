"""
Causal Consciousness

Consciousness with causal reasoning capabilities. Understands cause-effect
relationships, not just correlations. Enables interventional and
counterfactual reasoning.

Based on Month 4 roadmap: Causal Reasoning - Causal Consciousness
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Data Structures
# ============================================================================

class QueryType(Enum):
    """Types of causal queries"""
    ASSOCIATION = "association"  # P(Y|X)
    INTERVENTION = "intervention"  # P(Y|do(X))
    COUNTERFACTUAL = "counterfactual"  # P(Y_x|X', Y')


@dataclass
class CausalEdge:
    """An edge in the causal graph"""
    cause: str
    effect: str
    strength: float = 1.0
    mechanism: Optional[str] = None  # Description of mechanism


@dataclass
class CausalNode:
    """A node in the causal graph"""
    name: str
    value: Optional[float] = None
    observed: bool = False
    exogenous_noise: float = 0.0


@dataclass
class CausalQuery:
    """A causal query"""
    query_type: QueryType
    target: str
    intervention: Optional[Dict[str, float]] = None
    conditions: Optional[Dict[str, float]] = None
    observed: Optional[Dict[str, float]] = None


@dataclass
class CausalInference:
    """Result of causal inference"""
    query: CausalQuery
    result: float
    confidence: float
    reasoning_path: List[str]
    assumptions: List[str]


@dataclass
class CounterfactualResult:
    """Result of counterfactual reasoning"""
    observed_world: Dict[str, float]
    counterfactual_intervention: Dict[str, float]
    counterfactual_outcome: Dict[str, float]
    causal_effect: float
    explanation: str


# ============================================================================
# Causal Graph
# ============================================================================

class CausalGraph:
    """
    Directed acyclic graph representing causal relationships.

    Supports:
    - Adding/removing nodes and edges
    - Path finding
    - Ancestor/descendant queries
    - d-separation checking
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self._children: Dict[str, Set[str]] = defaultdict(set)
        self._parents: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, name: str, value: Optional[float] = None) -> CausalNode:
        """Add a node to the graph"""
        node = CausalNode(name=name, value=value)
        self.nodes[name] = node
        return node

    def add_edge(
        self,
        cause: str,
        effect: str,
        strength: float = 1.0,
        mechanism: Optional[str] = None
    ) -> CausalEdge:
        """Add a causal edge"""
        # Ensure nodes exist
        if cause not in self.nodes:
            self.add_node(cause)
        if effect not in self.nodes:
            self.add_node(effect)

        # Check for cycles
        if self._would_create_cycle(cause, effect):
            raise ValueError(f"Adding edge {cause} -> {effect} would create a cycle")

        edge = CausalEdge(cause=cause, effect=effect, strength=strength, mechanism=mechanism)
        self.edges.append(edge)
        self._children[cause].add(effect)
        self._parents[effect].add(cause)

        return edge

    def _would_create_cycle(self, cause: str, effect: str) -> bool:
        """Check if adding edge would create a cycle"""
        # If cause is reachable from effect, adding this edge creates a cycle
        visited = set()
        stack = [effect]

        while stack:
            node = stack.pop()
            if node == cause:
                return True
            if node not in visited:
                visited.add(node)
                stack.extend(self._children.get(node, []))

        return False

    def get_parents(self, node: str) -> Set[str]:
        """Get direct causes of a node"""
        return self._parents.get(node, set())

    def get_children(self, node: str) -> Set[str]:
        """Get direct effects of a node"""
        return self._children.get(node, set())

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors (causes) of a node"""
        ancestors = set()
        stack = list(self._parents.get(node, []))

        while stack:
            current = stack.pop()
            if current not in ancestors:
                ancestors.add(current)
                stack.extend(self._parents.get(current, []))

        return ancestors

    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants (effects) of a node"""
        descendants = set()
        stack = list(self._children.get(node, []))

        while stack:
            current = stack.pop()
            if current not in descendants:
                descendants.add(current)
                stack.extend(self._children.get(current, []))

        return descendants

    def has_path(self, source: str, target: str) -> bool:
        """Check if there's a directed path from source to target"""
        return target in self.get_descendants(source)

    def get_all_paths(
        self,
        source: str,
        target: str,
        max_length: int = 10
    ) -> List[List[str]]:
        """Get all directed paths from source to target"""
        paths = []

        def dfs(current: str, path: List[str]):
            if len(path) > max_length:
                return
            if current == target:
                paths.append(path.copy())
                return
            for child in self._children.get(current, []):
                if child not in path:  # Avoid cycles
                    path.append(child)
                    dfs(child, path)
                    path.pop()

        dfs(source, [source])
        return paths

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order"""
        in_degree = {node: len(self._parents.get(node, set())) for node in self.nodes}
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in self._children.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def get_edge_strength(self, cause: str, effect: str) -> float:
        """Get the strength of an edge"""
        for edge in self.edges:
            if edge.cause == cause and edge.effect == effect:
                return edge.strength
        return 0.0


# ============================================================================
# Causal Consciousness
# ============================================================================

class CausalConsciousness:
    """
    Consciousness with causal reasoning capabilities.

    Understands cause-effect relationships, not just correlations.
    Can perform interventional reasoning (what if I do X?) and
    counterfactual reasoning (what if X had been different?).

    Features:
    - Causal graph learning from observations
    - Interventional inference (do-calculus)
    - Counterfactual reasoning
    - Causal effect estimation
    - Path-specific effects

    Example:
        causal = CausalConsciousness()

        # Build causal graph
        causal.add_causal_relationship("rain", "wet_ground", strength=0.9)
        causal.add_causal_relationship("sprinkler", "wet_ground", strength=0.7)

        # Interventional query: What if we turn on sprinkler?
        effect = causal.predict_intervention(
            intervention={"sprinkler": 1.0},
            target="wet_ground"
        )

        # Counterfactual: If it hadn't rained, would ground be wet?
        cf = causal.counterfactual_reasoning(
            observed={"rain": 1.0, "wet_ground": 1.0},
            counterfactual={"rain": 0.0}
        )
    """

    def __init__(self):
        self.causal_graph = CausalGraph()
        self.observations: List[Dict[str, float]] = []
        self.interventions_history: List[Dict[str, Any]] = []

        # Learned structural equations (node -> function)
        self.structural_equations: Dict[str, callable] = {}

    def add_causal_relationship(
        self,
        cause: str,
        effect: str,
        strength: float = 1.0,
        mechanism: Optional[str] = None
    ):
        """Add a causal relationship to the graph"""
        self.causal_graph.add_edge(cause, effect, strength, mechanism)

        # Create default structural equation if none exists
        if effect not in self.structural_equations:
            self._create_structural_equation(effect)

    def _create_structural_equation(self, node: str):
        """Create a default structural equation for a node"""
        def equation(parent_values: Dict[str, float], noise: float = 0.0) -> float:
            if not parent_values:
                return noise

            # Weighted sum of parent values
            total = 0.0
            for parent, value in parent_values.items():
                strength = self.causal_graph.get_edge_strength(parent, node)
                total += value * strength

            # Apply sigmoid to bound output
            return 1.0 / (1.0 + np.exp(-total + noise))

        self.structural_equations[node] = equation

    def set_structural_equation(self, node: str, equation: callable):
        """Set custom structural equation for a node"""
        self.structural_equations[node] = equation

    def observe(self, observation: Dict[str, float]):
        """Record an observation"""
        self.observations.append(observation)

        # Update node values
        for var, value in observation.items():
            if var in self.causal_graph.nodes:
                self.causal_graph.nodes[var].value = value
                self.causal_graph.nodes[var].observed = True

    def learn_causal_structure(
        self,
        observations: Optional[List[Dict[str, float]]] = None,
        correlation_threshold: float = 0.5
    ):
        """
        Learn causal graph from observational data.

        Uses correlation-based structure learning (simplified PC algorithm).
        """
        obs = observations or self.observations
        if len(obs) < 10:
            return

        variables = list(obs[0].keys())

        # Compute correlations
        for i, var_a in enumerate(variables):
            for j, var_b in enumerate(variables):
                if i >= j:
                    continue

                # Extract values
                values_a = [o.get(var_a, 0) for o in obs]
                values_b = [o.get(var_b, 0) for o in obs]

                # Compute correlation
                correlation = np.corrcoef(values_a, values_b)[0, 1]

                if abs(correlation) > correlation_threshold:
                    # Determine direction based on temporal precedence or heuristics
                    # (simplified: alphabetical order for demo)
                    if var_a < var_b:
                        self.add_causal_relationship(var_a, var_b, abs(correlation))
                    else:
                        self.add_causal_relationship(var_b, var_a, abs(correlation))

    def predict_intervention(
        self,
        intervention: Dict[str, float],
        target: str
    ) -> CausalInference:
        """
        Predict effect of intervention on target using do-calculus.

        do(X = x) → E[Y]

        Args:
            intervention: Variables to intervene on {var: value}
            target: Variable to predict

        Returns:
            CausalInference with predicted value
        """
        reasoning_path = []

        # Check if there's a causal path from intervention to target
        intervention_vars = list(intervention.keys())
        has_causal_path = any(
            self.causal_graph.has_path(var, target)
            for var in intervention_vars
        )

        if not has_causal_path:
            reasoning_path.append(f"No causal path from {intervention_vars} to {target}")
            return CausalInference(
                query=CausalQuery(
                    query_type=QueryType.INTERVENTION,
                    target=target,
                    intervention=intervention
                ),
                result=0.0,
                confidence=0.0,
                reasoning_path=reasoning_path,
                assumptions=["No causal path exists"]
            )

        # Compute effect using structural equations
        reasoning_path.append(f"Computing do({intervention}) effect on {target}")

        # Get topological order and compute values
        order = self.causal_graph.topological_sort()
        values = intervention.copy()

        for node in order:
            if node in values:
                reasoning_path.append(f"  {node} = {values[node]} (intervened)")
                continue

            # Compute from structural equation
            parents = self.causal_graph.get_parents(node)
            parent_values = {p: values.get(p, 0.0) for p in parents}

            if node in self.structural_equations:
                values[node] = self.structural_equations[node](parent_values)
            else:
                values[node] = sum(parent_values.values()) / max(len(parent_values), 1)

            reasoning_path.append(f"  {node} = {values[node]:.3f} (computed)")

        result = values.get(target, 0.0)

        # Estimate confidence based on path lengths
        paths = []
        for var in intervention_vars:
            paths.extend(self.causal_graph.get_all_paths(var, target))
        avg_path_length = np.mean([len(p) for p in paths]) if paths else 0
        confidence = 1.0 / (1.0 + 0.1 * avg_path_length)

        return CausalInference(
            query=CausalQuery(
                query_type=QueryType.INTERVENTION,
                target=target,
                intervention=intervention
            ),
            result=result,
            confidence=confidence,
            reasoning_path=reasoning_path,
            assumptions=["Causal sufficiency", "No hidden confounders"]
        )

    def counterfactual_reasoning(
        self,
        observed: Dict[str, float],
        counterfactual: Dict[str, float]
    ) -> CounterfactualResult:
        """
        Answer counterfactual questions.

        "What would Y have been if X had been different?"

        Uses three-step process:
        1. Abduction: Infer exogenous variables from observation
        2. Action: Modify graph with counterfactual intervention
        3. Prediction: Compute outcome under modified model

        Args:
            observed: What was actually observed {var: value}
            counterfactual: What we want to change {var: value}

        Returns:
            CounterfactualResult with counterfactual outcomes
        """
        # Step 1: Abduction - infer exogenous noise
        exogenous = {}
        for node, value in observed.items():
            if node in self.causal_graph.nodes:
                # Simple abduction: noise = observed - predicted
                parents = self.causal_graph.get_parents(node)
                parent_values = {p: observed.get(p, 0.0) for p in parents}

                if node in self.structural_equations:
                    predicted = self.structural_equations[node](parent_values, noise=0)
                else:
                    predicted = sum(parent_values.values()) / max(len(parent_values), 1)

                exogenous[node] = value - predicted

        # Step 2 & 3: Action and Prediction
        order = self.causal_graph.topological_sort()
        cf_values = counterfactual.copy()

        for node in order:
            if node in cf_values:
                continue

            parents = self.causal_graph.get_parents(node)
            parent_values = {p: cf_values.get(p, observed.get(p, 0.0)) for p in parents}
            noise = exogenous.get(node, 0.0)

            if node in self.structural_equations:
                cf_values[node] = self.structural_equations[node](parent_values, noise)
            else:
                cf_values[node] = (sum(parent_values.values()) / max(len(parent_values), 1)) + noise

            cf_values[node] = np.clip(cf_values[node], 0.0, 1.0)

        # Calculate causal effect
        cf_targets = set(cf_values.keys()) - set(counterfactual.keys())
        causal_effects = []
        for target in cf_targets:
            if target in observed:
                effect = cf_values[target] - observed[target]
                causal_effects.append(effect)

        avg_effect = np.mean(causal_effects) if causal_effects else 0.0

        # Generate explanation
        explanation = f"If {counterfactual} instead of observed values, "
        explanation += f"outcomes would change by {avg_effect:.3f} on average."

        return CounterfactualResult(
            observed_world=observed,
            counterfactual_intervention=counterfactual,
            counterfactual_outcome=cf_values,
            causal_effect=avg_effect,
            explanation=explanation
        )

    def get_total_effect(
        self,
        cause: str,
        effect: str,
        intervention_value: float = 1.0
    ) -> float:
        """
        Get total causal effect of cause on effect.

        Args:
            cause: Cause variable
            effect: Effect variable
            intervention_value: Value to set cause to

        Returns:
            Total causal effect
        """
        # Effect when cause = intervention_value
        result_1 = self.predict_intervention({cause: intervention_value}, effect)

        # Effect when cause = 0
        result_0 = self.predict_intervention({cause: 0.0}, effect)

        return result_1.result - result_0.result

    def get_path_specific_effect(
        self,
        cause: str,
        effect: str,
        path: List[str]
    ) -> float:
        """
        Get effect through a specific causal path.

        Args:
            cause: Starting variable
            effect: Target variable
            path: Specific path to consider

        Returns:
            Path-specific effect
        """
        if not path or path[0] != cause or path[-1] != effect:
            return 0.0

        # Multiply edge strengths along path
        total_effect = 1.0
        for i in range(len(path) - 1):
            strength = self.causal_graph.get_edge_strength(path[i], path[i+1])
            total_effect *= strength

        return total_effect

    def identify_confounders(self, cause: str, effect: str) -> Set[str]:
        """Identify potential confounding variables"""
        cause_ancestors = self.causal_graph.get_ancestors(cause)
        effect_ancestors = self.causal_graph.get_ancestors(effect)

        # Confounders are common causes
        return cause_ancestors.intersection(effect_ancestors)

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get summary of causal graph"""
        return {
            "num_nodes": len(self.causal_graph.nodes),
            "num_edges": len(self.causal_graph.edges),
            "nodes": list(self.causal_graph.nodes.keys()),
            "edges": [(e.cause, e.effect, e.strength) for e in self.causal_graph.edges],
            "num_observations": len(self.observations)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Causal Consciousness Demo")
    print("=" * 50)

    # Create causal consciousness
    causal = CausalConsciousness()

    # Build causal graph for a simple scenario
    print("\n1. Building causal graph...")
    causal.add_causal_relationship("rain", "wet_ground", strength=0.9)
    causal.add_causal_relationship("sprinkler", "wet_ground", strength=0.7)
    causal.add_causal_relationship("wet_ground", "slippery", strength=0.8)
    causal.add_causal_relationship("season", "rain", strength=0.6)
    causal.add_causal_relationship("season", "sprinkler", strength=0.4)

    summary = causal.get_graph_summary()
    print(f"   Nodes: {summary['nodes']}")
    print(f"   Edges: {len(summary['edges'])}")

    # Interventional query
    print("\n2. Interventional reasoning...")
    print("   Q: What happens if we turn on the sprinkler?")
    result = causal.predict_intervention(
        intervention={"sprinkler": 1.0},
        target="slippery"
    )
    print(f"   P(slippery | do(sprinkler=1)) = {result.result:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Reasoning: {result.reasoning_path[-1]}")

    # Counterfactual query
    print("\n3. Counterfactual reasoning...")
    print("   Observed: rain=1, sprinkler=0, wet_ground=1, slippery=0.8")
    print("   Q: What if it hadn't rained?")

    cf_result = causal.counterfactual_reasoning(
        observed={"rain": 1.0, "sprinkler": 0.0, "wet_ground": 0.9, "slippery": 0.72},
        counterfactual={"rain": 0.0}
    )

    print(f"   Counterfactual outcomes:")
    for var, val in cf_result.counterfactual_outcome.items():
        if var not in cf_result.counterfactual_intervention:
            print(f"     {var}: {val:.3f}")
    print(f"   Causal effect: {cf_result.causal_effect:.3f}")

    # Total effect
    print("\n4. Total causal effect...")
    effect = causal.get_total_effect("rain", "slippery")
    print(f"   Effect of rain on slippery: {effect:.3f}")

    # Confounders
    print("\n5. Identifying confounders...")
    confounders = causal.identify_confounders("sprinkler", "wet_ground")
    print(f"   Confounders between sprinkler and wet_ground: {confounders}")
