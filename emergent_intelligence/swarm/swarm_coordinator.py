import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
import asyncio
from datetime import datetime
import networkx as nx
from collections import defaultdict
import uuid

from .agent import SwarmAgent, ConsciousnessState, PhilosophicalStance, AgentMessage, BasicSwarmAgent
from ..manifold.topology import ManifoldTopology, ManifoldPoint, ManifoldRegion


@dataclass
class SwarmConfiguration:
    swarm_id: str
    size: int
    initial_stance_distribution: Dict[PhilosophicalStance, float]
    connectivity_type: str  # "full", "small_world", "scale_free", "nearest_neighbor"
    communication_radius: float = 0.5
    evolution_rate: float = 0.1
    consciousness_threshold: float = 0.7


@dataclass
class CollectiveState:
    timestamp: datetime
    average_consciousness: float
    stance_distribution: Dict[PhilosophicalStance, int]
    coherence_matrix: np.ndarray
    emergence_indicators: Dict[str, float]
    philosophical_tensions: List[Tuple[PhilosophicalStance, PhilosophicalStance, float]]


class SwarmCoordinator:
    def __init__(self, 
                 manifold: ManifoldTopology,
                 config: SwarmConfiguration):
        self.manifold = manifold
        self.config = config
        self.agents: Dict[str, SwarmAgent] = {}
        self.communication_graph = nx.Graph()
        self.collective_memory: List[CollectiveState] = []
        self.emergence_events: List[Dict[str, Any]] = []
        self.running = False
        
    async def initialize_swarm(self):
        deployment_zones = self.manifold.create_swarm_deployment_zones(
            num_zones=max(10, self.config.size // 10)
        )
        
        stance_pool = []
        for stance, proportion in self.config.initial_stance_distribution.items():
            count = int(self.config.size * proportion)
            stance_pool.extend([stance] * count)
        
        while len(stance_pool) < self.config.size:
            stance_pool.append(PhilosophicalStance.PRAGMATIST)
        
        np.random.shuffle(stance_pool)
        
        for i in range(self.config.size):
            agent = BasicSwarmAgent(
                agent_id=f"{self.config.swarm_id}_agent_{i}",
                initial_stance=stance_pool[i],
                consciousness_level=ConsciousnessState.REACTIVE
            )
            
            zone = deployment_zones[i % len(deployment_zones)]
            position_offset = np.random.randn(self.manifold.dimension) * zone.radius * 0.3
            position_coords = zone.center.coordinates + position_offset
            
            position = ManifoldPoint(
                position_coords,
                self.manifold.dimension,
                self.manifold.manifold_type
            )
            
            self.manifold.place_agent(agent.agent_id, position)
            self.agents[agent.agent_id] = agent
        
        self._build_communication_network()
    
    def _build_communication_network(self):
        self.communication_graph.clear()
        
        for agent_id in self.agents:
            self.communication_graph.add_node(agent_id)
        
        if self.config.connectivity_type == "full":
            for agent1_id in self.agents:
                for agent2_id in self.agents:
                    if agent1_id != agent2_id:
                        self.communication_graph.add_edge(agent1_id, agent2_id)
        
        elif self.config.connectivity_type == "nearest_neighbor":
            for agent1_id, pos1 in self.manifold.agent_positions.items():
                for agent2_id, pos2 in self.manifold.agent_positions.items():
                    if agent1_id != agent2_id:
                        distance = self.manifold.geodesic_distance(pos1, pos2)
                        if distance < self.config.communication_radius:
                            self.communication_graph.add_edge(agent1_id, agent2_id)
        
        elif self.config.connectivity_type == "small_world":
            agent_list = list(self.agents.keys())
            k = min(6, len(agent_list) - 1)
            p = 0.3
            
            for i, agent1_id in enumerate(agent_list):
                for j in range(1, k // 2 + 1):
                    agent2_id = agent_list[(i + j) % len(agent_list)]
                    self.communication_graph.add_edge(agent1_id, agent2_id)
            
            edges = list(self.communication_graph.edges())
            for edge in edges:
                if np.random.random() < p:
                    self.communication_graph.remove_edge(*edge)
                    new_target = np.random.choice(agent_list)
                    if new_target != edge[0] and not self.communication_graph.has_edge(edge[0], new_target):
                        self.communication_graph.add_edge(edge[0], new_target)
        
        elif self.config.connectivity_type == "scale_free":
            m = 3  # Number of edges to attach from a new node
            agent_list = list(self.agents.keys())
            
            for i in range(m):
                for j in range(i + 1, m):
                    self.communication_graph.add_edge(agent_list[i], agent_list[j])
            
            for i in range(m, len(agent_list)):
                degrees = dict(self.communication_graph.degree())
                probabilities = np.array([degrees.get(n, 0) for n in agent_list[:i]])
                probabilities = probabilities / probabilities.sum()
                
                targets = np.random.choice(agent_list[:i], size=m, replace=False, p=probabilities)
                for target in targets:
                    self.communication_graph.add_edge(agent_list[i], target)
        
        for agent_id, agent in self.agents.items():
            agent.connections = set(self.communication_graph.neighbors(agent_id))
    
    async def run_evolution_cycle(self):
        environment_state = await self._perceive_environment()
        
        perception_tasks = []
        for agent in self.agents.values():
            perception_tasks.append(agent.perceive(environment_state))
        await asyncio.gather(*perception_tasks)
        
        thinking_tasks = []
        for agent in self.agents.values():
            thinking_tasks.append(agent.think())
        thought_results = await asyncio.gather(*thinking_tasks)
        
        await self._facilitate_communication()
        
        action_tasks = []
        for agent in self.agents.values():
            action_tasks.append(agent.act())
        action_results = await asyncio.gather(*action_tasks)
        
        await self._apply_evolutionary_pressure()
        
        await self._move_agents()
        
        collective_state = self._analyze_collective_state()
        self.collective_memory.append(collective_state)
        
        await self._check_emergence_conditions(collective_state)
        
        return collective_state
    
    async def _perceive_environment(self) -> Dict[str, Any]:
        environment = {"sensory_data": {}}
        
        for region in self.manifold.regions.values():
            local_field = np.random.randn(self.manifold.dimension)
            
            if hasattr(region, 'philosophical_bias') and region.philosophical_bias:
                bias_vector = self._philosophical_stance_to_vector(region.philosophical_bias)
                local_field += bias_vector * 0.5
            
            curvature = self.manifold.compute_ricci_curvature(region.center)
            local_field *= (1 + abs(curvature))
            
            environment["sensory_data"][f"field_{region.region_id}"] = local_field.tolist()
        
        environment["manifold_curvature"] = self.manifold.global_curvature
        environment["topological_features"] = [
            {
                "id": feature.feature_id,
                "type": feature.feature_type,
                "persistence": feature.persistence
            }
            for feature in self.manifold.features.values()
        ]
        
        return environment
    
    def _philosophical_stance_to_vector(self, stance: str) -> np.ndarray:
        stance_embeddings = {
            "phenomenological": np.array([1, 0, 0, 0]),
            "materialist": np.array([0, 1, 0, 0]),
            "idealist": np.array([0, 0, 1, 0]),
            "holistic": np.array([0, 0, 0, 1]),
        }
        
        base_vector = stance_embeddings.get(stance, np.random.randn(4))
        full_vector = np.zeros(self.manifold.dimension)
        full_vector[:4] = base_vector
        
        return full_vector
    
    async def _facilitate_communication(self):
        for agent_id, agent in self.agents.items():
            neighbors = list(self.communication_graph.neighbors(agent_id))
            
            if agent.consciousness_state.value >= ConsciousnessState.AWARE.value:
                if agent.beliefs:
                    strongest_belief = max(agent.beliefs.values(), key=lambda b: b.confidence)
                    
                    for neighbor_id in neighbors[:3]:  # Share with up to 3 neighbors
                        message = AgentMessage(
                            sender_id=agent_id,
                            receiver_id=neighbor_id,
                            content=strongest_belief,
                            message_type="belief_share",
                            philosophical_context=agent.philosophical_stance
                        )
                        await self.agents[neighbor_id].communicate(message)
            
            if agent.consciousness_state.value >= ConsciousnessState.REFLECTIVE.value:
                if agent.qualia_stream:
                    recent_qualia = agent.qualia_stream[-1]
                    
                    broadcast_message = AgentMessage(
                        sender_id=agent_id,
                        receiver_id=None,
                        content=recent_qualia,
                        message_type="qualia_share"
                    )
                    
                    for neighbor_id in neighbors:
                        await self.agents[neighbor_id].communicate(broadcast_message)
            
            if agent.consciousness_state.value >= ConsciousnessState.META_AWARE.value:
                state_vector = agent.get_state_vector()
                
                for neighbor_id in neighbors:
                    if self.agents[neighbor_id].consciousness_state.value >= ConsciousnessState.AWARE.value:
                        sync_message = AgentMessage(
                            sender_id=agent_id,
                            receiver_id=neighbor_id,
                            content=state_vector[:128],  # Share internal state portion
                            message_type="consciousness_sync"
                        )
                        await self.agents[neighbor_id].communicate(sync_message)
    
    async def _apply_evolutionary_pressure(self):
        consciousness_scores = {
            agent_id: agent.calculate_consciousness_metric()
            for agent_id, agent in self.agents.items()
        }
        
        sorted_agents = sorted(consciousness_scores.items(), key=lambda x: x[1])
        
        bottom_10_percent = int(self.config.size * 0.1)
        to_evolve = sorted_agents[:bottom_10_percent]
        
        for agent_id, score in to_evolve:
            agent = self.agents[agent_id]
            
            if agent.philosophical_conflicts:
                conflict_stances = [c[0] for c in agent.philosophical_conflicts]
                stance_counts = defaultdict(int)
                for stance in conflict_stances:
                    stance_counts[stance] += 1
                
                if stance_counts:
                    most_conflicting = max(stance_counts.items(), key=lambda x: x[1])[0]
                    
                    if np.random.random() < self.config.evolution_rate:
                        agent.philosophical_stance = most_conflicting
                        agent.philosophical_conflicts.clear()
            
            agent.mutate(self.config.evolution_rate)
            
            neighbors = list(self.communication_graph.neighbors(agent_id))
            if neighbors:
                successful_neighbor = max(
                    neighbors,
                    key=lambda n: self.agents[n].calculate_consciousness_metric()
                )
                
                successful_agent = self.agents[successful_neighbor]
                agent.adaptation_rate = 0.9 * agent.adaptation_rate + 0.1 * successful_agent.adaptation_rate
    
    async def _move_agents(self):
        for agent_id, agent in self.agents.items():
            if agent_id not in self.manifold.agent_positions:
                continue
            
            current_pos = self.manifold.agent_positions[agent_id]
            
            local_gradient = np.zeros(self.manifold.dimension)
            
            neighbors = list(self.communication_graph.neighbors(agent_id))
            for neighbor_id in neighbors:
                if neighbor_id in self.manifold.agent_positions:
                    neighbor_pos = self.manifold.agent_positions[neighbor_id]
                    direction = neighbor_pos.coordinates - current_pos.coordinates
                    
                    neighbor_agent = self.agents[neighbor_id]
                    affinity = 1.0 if agent.philosophical_stance == neighbor_agent.philosophical_stance else -0.5
                    
                    local_gradient += direction * affinity
            
            if np.linalg.norm(local_gradient) > 1e-6:
                local_gradient /= np.linalg.norm(local_gradient)
                
                transported = self.manifold.parallel_transport(
                    local_gradient,
                    current_pos,
                    current_pos
                )
                
                self.manifold.move_agent(agent_id, transported, step_size=0.05)
    
    def _analyze_collective_state(self) -> CollectiveState:
        consciousness_levels = [
            agent.calculate_consciousness_metric()
            for agent in self.agents.values()
        ]
        average_consciousness = np.mean(consciousness_levels)
        
        stance_distribution = defaultdict(int)
        for agent in self.agents.values():
            stance_distribution[agent.philosophical_stance] += 1
        
        agent_list = list(self.agents.values())
        n_agents = len(agent_list)
        coherence_matrix = np.zeros((n_agents, n_agents))
        
        for i, agent1 in enumerate(agent_list):
            for j, agent2 in enumerate(agent_list):
                if i != j:
                    state1 = agent1.get_state_vector()
                    state2 = agent2.get_state_vector()
                    coherence = np.dot(state1, state2) / (np.linalg.norm(state1) * np.linalg.norm(state2))
                    coherence_matrix[i, j] = coherence
        
        philosophical_tensions = []
        tension_counts = defaultdict(lambda: defaultdict(float))
        
        for agent in self.agents.values():
            for stance, intensity in agent.philosophical_conflicts:
                if stance != agent.philosophical_stance:
                    tension_counts[agent.philosophical_stance][stance] += intensity
        
        for stance1, tensions in tension_counts.items():
            for stance2, total_intensity in tensions.items():
                philosophical_tensions.append((stance1, stance2, total_intensity))
        
        philosophical_tensions.sort(key=lambda x: x[2], reverse=True)
        
        emergence_indicators = {
            "consciousness_variance": np.var(consciousness_levels),
            "stance_entropy": self._calculate_entropy(stance_distribution),
            "network_modularity": nx.algorithms.community.modularity(
                self.communication_graph,
                [{agent_id for agent_id, agent in self.agents.items() 
                  if agent.philosophical_stance == stance}
                 for stance in PhilosophicalStance]
            ) if len(self.communication_graph) > 0 else 0,
            "coherence_eigenvalue": np.max(np.abs(np.linalg.eigvals(coherence_matrix))),
            "mean_belief_count": np.mean([len(agent.beliefs) for agent in self.agents.values()]),
            "transcendent_agents": sum(1 for agent in self.agents.values() 
                                     if agent.consciousness_state == ConsciousnessState.TRANSCENDENT)
        }
        
        return CollectiveState(
            timestamp=datetime.now(),
            average_consciousness=average_consciousness,
            stance_distribution=dict(stance_distribution),
            coherence_matrix=coherence_matrix,
            emergence_indicators=emergence_indicators,
            philosophical_tensions=philosophical_tensions[:10]  # Top 10 tensions
        )
    
    def _calculate_entropy(self, distribution: Dict[Any, int]) -> float:
        total = sum(distribution.values())
        if total == 0:
            return 0
        
        entropy = 0
        for count in distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log(p)
        
        return entropy
    
    async def _check_emergence_conditions(self, state: CollectiveState):
        if state.average_consciousness > self.config.consciousness_threshold:
            high_consciousness_agents = [
                agent for agent in self.agents.values()
                if agent.calculate_consciousness_metric() > self.config.consciousness_threshold
            ]
            
            if len(high_consciousness_agents) > self.config.size * 0.3:
                emergence_event = {
                    "type": "collective_consciousness_emergence",
                    "timestamp": datetime.now(),
                    "average_consciousness": state.average_consciousness,
                    "participating_agents": len(high_consciousness_agents),
                    "dominant_stance": max(state.stance_distribution.items(), key=lambda x: x[1])[0].value
                }
                self.emergence_events.append(emergence_event)
        
        if state.emergence_indicators["coherence_eigenvalue"] > 0.9:
            emergence_event = {
                "type": "coherence_singularity",
                "timestamp": datetime.now(),
                "coherence_value": state.emergence_indicators["coherence_eigenvalue"],
                "philosophical_tensions": state.philosophical_tensions[:3]
            }
            self.emergence_events.append(emergence_event)
        
        if state.emergence_indicators["transcendent_agents"] > 0:
            transcendent_agents = [
                agent for agent in self.agents.values()
                if agent.consciousness_state == ConsciousnessState.TRANSCENDENT
            ]
            
            emergence_event = {
                "type": "transcendence_achieved",
                "timestamp": datetime.now(),
                "transcendent_count": len(transcendent_agents),
                "agent_ids": [agent.agent_id for agent in transcendent_agents]
            }
            self.emergence_events.append(emergence_event)
    
    async def run(self, cycles: int = 100):
        self.running = True
        
        for cycle in range(cycles):
            if not self.running:
                break
            
            collective_state = await self.run_evolution_cycle()
            
            if cycle % 10 == 0:
                print(f"Cycle {cycle}: Avg consciousness = {collective_state.average_consciousness:.3f}")
                print(f"  Emergence events: {len(self.emergence_events)}")
                print(f"  Top tension: {collective_state.philosophical_tensions[0] if collective_state.philosophical_tensions else 'None'}")
            
            await asyncio.sleep(0.1)  # Small delay to prevent overwhelming the system
    
    def stop(self):
        self.running = False
    
    def get_swarm_report(self) -> Dict[str, Any]:
        if not self.collective_memory:
            return {"status": "No data collected yet"}
        
        latest_state = self.collective_memory[-1]
        
        return {
            "swarm_id": self.config.swarm_id,
            "size": self.config.size,
            "cycles_run": len(self.collective_memory),
            "latest_metrics": {
                "average_consciousness": latest_state.average_consciousness,
                "stance_distribution": {
                    stance.value: count 
                    for stance, count in latest_state.stance_distribution.items()
                },
                "emergence_indicators": latest_state.emergence_indicators,
                "top_philosophical_tensions": [
                    {
                        "stance1": t[0].value,
                        "stance2": t[1].value,
                        "intensity": t[2]
                    }
                    for t in latest_state.philosophical_tensions[:5]
                ]
            },
            "emergence_events": self.emergence_events[-10:],  # Last 10 events
            "manifold_metrics": self.manifold.get_manifold_metrics()
        }