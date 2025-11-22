"""
Swarm Intelligence Patterns

Multiple swarm intelligence coordination patterns with different trade-offs.

Based on Month 5 roadmap: Collective Intelligence - Swarm Intelligence Patterns
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import random

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# Swarm Intelligence Patterns
# ============================================================================

class SwarmPattern(Enum):
    """Available swarm intelligence patterns"""
    STIGMERGY = "stigmergy"           # Ant pheromone trails
    QUORUM_SENSING = "quorum_sensing"  # Bacterial coordination
    FLOCKING = "flocking"              # Bird/fish swarms (Boids)
    CONSENSUS = "consensus"            # Democratic voting
    DIVISION_OF_LABOR = "division_of_labor"  # Bee hive roles
    PARTICLE_SWARM = "particle_swarm"  # PSO optimization


# ============================================================================
# Agent Classes
# ============================================================================

@dataclass
class SwarmAgent:
    """Base agent in a swarm"""
    agent_id: str
    position: np.ndarray
    velocity: np.ndarray = None
    role: Optional[str] = None
    energy: float = 1.0
    memory: Dict[str, Any] = field(default_factory=dict)
    vote: Optional[str] = None

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.position)


@dataclass
class SharedEnvironment:
    """Shared environment for stigmergy"""
    dimensions: Tuple[int, ...]
    pheromone_map: np.ndarray = None
    resource_map: np.ndarray = None
    evaporation_rate: float = 0.1

    def __post_init__(self):
        if self.pheromone_map is None:
            self.pheromone_map = np.zeros(self.dimensions)
        if self.resource_map is None:
            self.resource_map = np.random.rand(*self.dimensions)

    def deposit(self, position: np.ndarray, strength: float = 1.0):
        """Deposit pheromone at position"""
        idx = tuple(np.clip(position.astype(int), 0, np.array(self.dimensions) - 1))
        self.pheromone_map[idx] += strength

    def evaporate(self):
        """Evaporate pheromones"""
        self.pheromone_map *= (1 - self.evaporation_rate)

    def get_local(self, position: np.ndarray, radius: int = 1) -> Dict[str, np.ndarray]:
        """Get local environment around position"""
        idx = position.astype(int)
        local_pheromone = {}
        local_resources = {}

        for d in range(len(self.dimensions)):
            for offset in [-radius, 0, radius]:
                new_idx = idx.copy()
                new_idx[d] = np.clip(idx[d] + offset, 0, self.dimensions[d] - 1)
                key = f"d{d}_o{offset}"
                local_pheromone[key] = self.pheromone_map[tuple(new_idx)]
                local_resources[key] = self.resource_map[tuple(new_idx)]

        return {
            "pheromone": local_pheromone,
            "resources": local_resources
        }


# ============================================================================
# Swarm Intelligence Engine
# ============================================================================

class SwarmIntelligenceEngine:
    """
    Implements various swarm intelligence patterns.

    Each pattern has different trade-offs:
    - Stigmergy: Decentralized, robust, slow convergence
    - Quorum sensing: Threshold-based decisions, democratic
    - Flocking: Emergent coordination, beautiful dynamics
    - Consensus: Explicit voting, guaranteed convergence
    - Division of labor: Specialized roles, efficient
    - Particle swarm: Optimization-focused, fast convergence

    Example:
        engine = SwarmIntelligenceEngine(SwarmPattern.FLOCKING)
        engine.initialize_agents(num_agents=50, dimensions=2)

        for _ in range(100):
            engine.step()
    """

    def __init__(self, pattern: SwarmPattern):
        self.pattern = pattern
        self.agents: List[SwarmAgent] = []
        self.environment: Optional[SharedEnvironment] = None
        self.step_count = 0
        self.history: List[Dict[str, Any]] = []

        # Pattern-specific parameters
        self.params = self._default_params()

    def _default_params(self) -> Dict[str, Any]:
        """Get default parameters for pattern"""
        defaults = {
            SwarmPattern.STIGMERGY: {
                "pheromone_strength": 1.0,
                "evaporation_rate": 0.1,
                "follow_probability": 0.7
            },
            SwarmPattern.QUORUM_SENSING: {
                "decision_threshold": 0.6,
                "signal_radius": 2.0,
                "response_threshold": 0.5
            },
            SwarmPattern.FLOCKING: {
                "separation_weight": 1.5,
                "alignment_weight": 1.0,
                "cohesion_weight": 1.0,
                "perception_radius": 5.0,
                "max_speed": 2.0,
                "max_force": 0.5
            },
            SwarmPattern.CONSENSUS: {
                "influence_radius": 3.0,
                "stubbornness": 0.3,
                "convergence_threshold": 0.9
            },
            SwarmPattern.DIVISION_OF_LABOR: {
                "roles": ["forager", "guard", "nurse", "scout"],
                "role_threshold": 0.5,
                "specialization_rate": 0.1
            },
            SwarmPattern.PARTICLE_SWARM: {
                "inertia_weight": 0.7,
                "cognitive_weight": 1.5,
                "social_weight": 1.5,
                "max_velocity": 1.0
            }
        }
        return defaults.get(self.pattern, {})

    def set_params(self, **kwargs):
        """Set pattern parameters"""
        self.params.update(kwargs)

    def initialize_agents(
        self,
        num_agents: int,
        dimensions: int = 2,
        space_size: float = 100.0
    ):
        """Initialize swarm agents"""
        self.agents = []

        for i in range(num_agents):
            position = np.random.rand(dimensions) * space_size
            velocity = (np.random.rand(dimensions) - 0.5) * 2

            agent = SwarmAgent(
                agent_id=f"agent_{i}",
                position=position,
                velocity=velocity
            )
            self.agents.append(agent)

        # Initialize environment for stigmergy
        if self.pattern == SwarmPattern.STIGMERGY:
            env_dims = tuple([int(space_size)] * dimensions)
            self.environment = SharedEnvironment(
                dimensions=env_dims,
                evaporation_rate=self.params.get("evaporation_rate", 0.1)
            )

    def step(self) -> Dict[str, Any]:
        """Execute one step of swarm behavior"""
        step_result = {"step": self.step_count}

        if self.pattern == SwarmPattern.STIGMERGY:
            step_result.update(self._stigmergy_step())
        elif self.pattern == SwarmPattern.QUORUM_SENSING:
            step_result.update(self._quorum_sensing_step())
        elif self.pattern == SwarmPattern.FLOCKING:
            step_result.update(self._flocking_step())
        elif self.pattern == SwarmPattern.CONSENSUS:
            step_result.update(self._consensus_step())
        elif self.pattern == SwarmPattern.DIVISION_OF_LABOR:
            step_result.update(self._division_of_labor_step())
        elif self.pattern == SwarmPattern.PARTICLE_SWARM:
            step_result.update(self._particle_swarm_step())

        self.step_count += 1
        self.history.append(step_result)

        return step_result

    # ========================================================================
    # Stigmergy (Ant Colony)
    # ========================================================================

    def _stigmergy_step(self) -> Dict[str, Any]:
        """
        Stigmergy: Coordinate through environment modification.
        Ants leave pheromones, others follow strongest trails.
        """
        pheromone_strength = self.params["pheromone_strength"]
        follow_prob = self.params["follow_probability"]

        moved_count = 0

        for agent in self.agents:
            # Get local pheromone gradient
            local = self.environment.get_local(agent.position)
            pheromone_gradient = local["pheromone"]

            # Decide movement
            if max(pheromone_gradient.values()) > 0.1 and random.random() < follow_prob:
                # Follow pheromone gradient
                best_direction = max(pheromone_gradient.items(), key=lambda x: x[1])[0]
                # Parse direction from key
                d_idx = int(best_direction.split('_')[0][1])
                offset = int(best_direction.split('_')[1][1:])
                move = np.zeros(len(agent.position))
                move[d_idx] = offset
            else:
                # Random walk
                move = (np.random.rand(len(agent.position)) - 0.5) * 2

            # Update position
            agent.position = np.clip(
                agent.position + move,
                0,
                np.array(self.environment.dimensions) - 1
            )
            moved_count += 1

            # Deposit pheromone
            self.environment.deposit(agent.position, pheromone_strength)

        # Evaporate
        self.environment.evaporate()

        return {
            "moved_agents": moved_count,
            "total_pheromone": np.sum(self.environment.pheromone_map)
        }

    # ========================================================================
    # Quorum Sensing
    # ========================================================================

    def _quorum_sensing_step(self) -> Dict[str, Any]:
        """
        Quorum sensing: Make group decision when threshold reached.
        Bacteria coordinate biofilm formation when density > threshold.
        """
        decision_threshold = self.params["decision_threshold"]
        signal_radius = self.params["signal_radius"]

        # Count local density for each agent
        for agent in self.agents:
            neighbors = 0
            for other in self.agents:
                if other.agent_id != agent.agent_id:
                    dist = np.linalg.norm(agent.position - other.position)
                    if dist < signal_radius:
                        neighbors += 1

            # Store local density
            local_density = neighbors / max(len(self.agents) - 1, 1)
            agent.memory["local_density"] = local_density

            # Vote based on local density
            if local_density > self.params["response_threshold"]:
                agent.vote = "yes"
            else:
                agent.vote = "no"

        # Check for quorum
        yes_votes = sum(1 for a in self.agents if a.vote == "yes")
        proportion = yes_votes / len(self.agents)
        decision_made = proportion >= decision_threshold

        return {
            "yes_votes": yes_votes,
            "proportion": proportion,
            "decision_made": decision_made,
            "avg_local_density": np.mean([a.memory.get("local_density", 0) for a in self.agents])
        }

    # ========================================================================
    # Flocking (Boids)
    # ========================================================================

    def _flocking_step(self) -> Dict[str, Any]:
        """
        Flocking: Three simple rules create complex behavior.
        1. Separation: Avoid crowding neighbors
        2. Alignment: Steer toward average heading
        3. Cohesion: Steer toward average position
        """
        separation_weight = self.params["separation_weight"]
        alignment_weight = self.params["alignment_weight"]
        cohesion_weight = self.params["cohesion_weight"]
        perception_radius = self.params["perception_radius"]
        max_speed = self.params["max_speed"]
        max_force = self.params["max_force"]

        for agent in self.agents:
            neighbors = self._get_neighbors(agent, perception_radius)

            if not neighbors:
                continue

            # Rule 1: Separation
            separation = self._compute_separation(agent, neighbors)

            # Rule 2: Alignment
            alignment = self._compute_alignment(agent, neighbors)

            # Rule 3: Cohesion
            cohesion = self._compute_cohesion(agent, neighbors)

            # Combine forces
            force = (
                separation * separation_weight +
                alignment * alignment_weight +
                cohesion * cohesion_weight
            )

            # Limit force
            force_mag = np.linalg.norm(force)
            if force_mag > max_force:
                force = force / force_mag * max_force

            # Update velocity and position
            agent.velocity += force
            vel_mag = np.linalg.norm(agent.velocity)
            if vel_mag > max_speed:
                agent.velocity = agent.velocity / vel_mag * max_speed

            agent.position += agent.velocity

        # Compute swarm metrics
        positions = np.array([a.position for a in self.agents])
        centroid = np.mean(positions, axis=0)
        dispersion = np.mean([np.linalg.norm(a.position - centroid) for a in self.agents])

        velocities = np.array([a.velocity for a in self.agents])
        avg_velocity = np.mean(velocities, axis=0)
        alignment_score = np.mean([
            np.dot(a.velocity, avg_velocity) / (np.linalg.norm(a.velocity) * np.linalg.norm(avg_velocity) + 1e-8)
            for a in self.agents
        ])

        return {
            "centroid": centroid.tolist(),
            "dispersion": dispersion,
            "alignment_score": alignment_score
        }

    def _get_neighbors(self, agent: SwarmAgent, radius: float) -> List[SwarmAgent]:
        """Get neighboring agents within radius"""
        neighbors = []
        for other in self.agents:
            if other.agent_id != agent.agent_id:
                dist = np.linalg.norm(agent.position - other.position)
                if dist < radius:
                    neighbors.append(other)
        return neighbors

    def _compute_separation(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> np.ndarray:
        """Separation: steer away from neighbors"""
        if not neighbors:
            return np.zeros_like(agent.position)

        steer = np.zeros_like(agent.position)
        for neighbor in neighbors:
            diff = agent.position - neighbor.position
            dist = np.linalg.norm(diff) + 1e-8
            steer += diff / (dist * dist)  # Weight by inverse square

        return steer / len(neighbors)

    def _compute_alignment(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> np.ndarray:
        """Alignment: steer toward average velocity"""
        if not neighbors:
            return np.zeros_like(agent.velocity)

        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        return avg_velocity - agent.velocity

    def _compute_cohesion(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> np.ndarray:
        """Cohesion: steer toward center of mass"""
        if not neighbors:
            return np.zeros_like(agent.position)

        center = np.mean([n.position for n in neighbors], axis=0)
        return center - agent.position

    # ========================================================================
    # Consensus
    # ========================================================================

    def _consensus_step(self) -> Dict[str, Any]:
        """
        Consensus: Agents influence neighbors' opinions.
        Converges to agreement through local interactions.
        """
        influence_radius = self.params["influence_radius"]
        stubbornness = self.params["stubbornness"]

        # Initialize opinions if not set
        for agent in self.agents:
            if "opinion" not in agent.memory:
                agent.memory["opinion"] = random.random()

        # Update opinions
        new_opinions = {}
        for agent in self.agents:
            neighbors = self._get_neighbors(agent, influence_radius)
            if neighbors:
                neighbor_opinions = [n.memory.get("opinion", 0.5) for n in neighbors]
                avg_neighbor = np.mean(neighbor_opinions)

                # Weighted update
                new_opinion = (
                    stubbornness * agent.memory["opinion"] +
                    (1 - stubbornness) * avg_neighbor
                )
                new_opinions[agent.agent_id] = new_opinion
            else:
                new_opinions[agent.agent_id] = agent.memory["opinion"]

        # Apply updates
        for agent in self.agents:
            agent.memory["opinion"] = new_opinions[agent.agent_id]

        # Check convergence
        opinions = [a.memory["opinion"] for a in self.agents]
        opinion_variance = np.var(opinions)
        avg_opinion = np.mean(opinions)
        convergence = 1.0 - np.sqrt(opinion_variance)

        return {
            "avg_opinion": avg_opinion,
            "opinion_variance": opinion_variance,
            "convergence": convergence,
            "converged": convergence > self.params["convergence_threshold"]
        }

    # ========================================================================
    # Division of Labor
    # ========================================================================

    def _division_of_labor_step(self) -> Dict[str, Any]:
        """
        Division of labor: Agents specialize into roles.
        Like bee hives with foragers, guards, nurses.
        """
        roles = self.params["roles"]
        role_threshold = self.params["role_threshold"]
        specialization_rate = self.params["specialization_rate"]

        # Initialize roles if not set
        for agent in self.agents:
            if agent.role is None:
                agent.role = random.choice(roles)
            if "role_proficiency" not in agent.memory:
                agent.memory["role_proficiency"] = {r: 0.5 for r in roles}

        # Simulate task performance and specialization
        role_counts = {r: 0 for r in roles}
        for agent in self.agents:
            # Perform role
            proficiency = agent.memory["role_proficiency"][agent.role]
            performance = proficiency * agent.energy

            # Increase proficiency in current role
            agent.memory["role_proficiency"][agent.role] = min(
                1.0,
                agent.memory["role_proficiency"][agent.role] + specialization_rate
            )

            # Decay other proficiencies
            for role in roles:
                if role != agent.role:
                    agent.memory["role_proficiency"][role] = max(
                        0.1,
                        agent.memory["role_proficiency"][role] - specialization_rate * 0.5
                    )

            # Maybe switch roles based on need
            if random.random() < 0.1:  # 10% chance to consider switching
                # Find understaffed role
                min_role = min(role_counts.items(), key=lambda x: x[1])[0]
                if role_counts[min_role] < len(self.agents) / len(roles) * 0.5:
                    agent.role = min_role

            role_counts[agent.role] += 1

        return {
            "role_distribution": role_counts,
            "avg_proficiency": np.mean([
                max(a.memory["role_proficiency"].values())
                for a in self.agents
            ])
        }

    # ========================================================================
    # Particle Swarm Optimization
    # ========================================================================

    def _particle_swarm_step(
        self,
        fitness_fn: Optional[Callable[[np.ndarray], float]] = None
    ) -> Dict[str, Any]:
        """
        Particle Swarm Optimization: Find optimal solution.
        Particles search space, sharing best positions found.
        """
        inertia = self.params["inertia_weight"]
        cognitive = self.params["cognitive_weight"]
        social = self.params["social_weight"]
        max_velocity = self.params["max_velocity"]

        # Default fitness: minimize distance from center
        if fitness_fn is None:
            def fitness_fn(pos):
                return -np.linalg.norm(pos - 50)  # Maximize = minimize distance from (50,50)

        # Initialize personal bests if not set
        for agent in self.agents:
            if "personal_best_pos" not in agent.memory:
                agent.memory["personal_best_pos"] = agent.position.copy()
                agent.memory["personal_best_fitness"] = fitness_fn(agent.position)

        # Find global best
        global_best_fitness = max(a.memory["personal_best_fitness"] for a in self.agents)
        global_best_pos = None
        for agent in self.agents:
            if agent.memory["personal_best_fitness"] == global_best_fitness:
                global_best_pos = agent.memory["personal_best_pos"]
                break

        # Update particles
        for agent in self.agents:
            # Random factors
            r1, r2 = random.random(), random.random()

            # Velocity update
            cognitive_component = cognitive * r1 * (agent.memory["personal_best_pos"] - agent.position)
            social_component = social * r2 * (global_best_pos - agent.position)

            agent.velocity = (
                inertia * agent.velocity +
                cognitive_component +
                social_component
            )

            # Limit velocity
            vel_mag = np.linalg.norm(agent.velocity)
            if vel_mag > max_velocity:
                agent.velocity = agent.velocity / vel_mag * max_velocity

            # Position update
            agent.position += agent.velocity

            # Update personal best
            fitness = fitness_fn(agent.position)
            if fitness > agent.memory["personal_best_fitness"]:
                agent.memory["personal_best_fitness"] = fitness
                agent.memory["personal_best_pos"] = agent.position.copy()

        return {
            "global_best_fitness": global_best_fitness,
            "global_best_pos": global_best_pos.tolist() if global_best_pos is not None else None,
            "avg_fitness": np.mean([fitness_fn(a.position) for a in self.agents])
        }

    # ========================================================================
    # Analysis Methods
    # ========================================================================

    def get_swarm_stats(self) -> Dict[str, Any]:
        """Get current swarm statistics"""
        if not self.agents:
            return {"status": "No agents"}

        positions = np.array([a.position for a in self.agents])
        velocities = np.array([a.velocity for a in self.agents])

        return {
            "num_agents": len(self.agents),
            "pattern": self.pattern.value,
            "step_count": self.step_count,
            "centroid": np.mean(positions, axis=0).tolist(),
            "dispersion": np.std(positions),
            "avg_speed": np.mean(np.linalg.norm(velocities, axis=1)),
            "energy": np.mean([a.energy for a in self.agents])
        }

    def run(self, steps: int, callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Run simulation for multiple steps"""
        results = []
        for i in range(steps):
            result = self.step()
            results.append(result)
            if callback:
                callback(i, result)
        return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Swarm Intelligence Patterns Demo")
    print("=" * 50)

    # Test Flocking
    print("\n1. Flocking (Boids) Pattern")
    engine = SwarmIntelligenceEngine(SwarmPattern.FLOCKING)
    engine.initialize_agents(num_agents=30, dimensions=2, space_size=100)

    for i in range(20):
        result = engine.step()
        if i % 5 == 0:
            print(f"   Step {i}: alignment={result['alignment_score']:.3f}, dispersion={result['dispersion']:.1f}")

    # Test Quorum Sensing
    print("\n2. Quorum Sensing Pattern")
    engine = SwarmIntelligenceEngine(SwarmPattern.QUORUM_SENSING)
    engine.initialize_agents(num_agents=50, dimensions=2, space_size=20)

    for i in range(10):
        result = engine.step()
        print(f"   Step {i}: yes_votes={result['yes_votes']}, decision_made={result['decision_made']}")

    # Test Consensus
    print("\n3. Consensus Pattern")
    engine = SwarmIntelligenceEngine(SwarmPattern.CONSENSUS)
    engine.initialize_agents(num_agents=20, dimensions=2, space_size=50)

    for i in range(20):
        result = engine.step()
        if i % 5 == 0:
            print(f"   Step {i}: avg_opinion={result['avg_opinion']:.3f}, convergence={result['convergence']:.3f}")

    # Test Particle Swarm
    print("\n4. Particle Swarm Optimization")
    engine = SwarmIntelligenceEngine(SwarmPattern.PARTICLE_SWARM)
    engine.initialize_agents(num_agents=30, dimensions=2, space_size=100)

    for i in range(30):
        result = engine.step()
        if i % 10 == 0:
            print(f"   Step {i}: best_fitness={result['global_best_fitness']:.3f}")

    print("\n5. Final Stats:")
    stats = engine.get_swarm_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
