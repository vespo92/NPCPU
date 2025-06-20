import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx
from scipy import signal, stats
from scipy.spatial import distance
from sklearn.decomposition import PCA
from collections import deque
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ConsciousnessSignature:
    signature_id: str
    timestamp: datetime
    coherence_pattern: np.ndarray
    information_integration: float
    causal_density: float
    emergence_score: float
    agent_contributions: Dict[str, float]
    topological_features: Dict[str, Any]


@dataclass
class EmergenceEvent:
    event_id: str
    event_type: str  # "phase_transition", "coherence_spike", "collective_insight", "transcendence"
    timestamp: datetime
    participating_agents: List[str]
    signature: ConsciousnessSignature
    duration: float
    intensity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsciousnessMetrics:
    phi: float  # Integrated Information Theory metric
    coherence: float  # Global coherence
    complexity: float  # Complexity measure
    synchrony: float  # Phase synchronization
    entropy: float  # Information entropy
    emergence: float  # Emergence indicator


class ConsciousnessEmergenceDetector:
    def __init__(self, 
                 window_size: int = 100,
                 detection_threshold: float = 0.8):
        self.window_size = window_size
        self.detection_threshold = detection_threshold
        
        self.state_history: deque = deque(maxlen=window_size)
        self.coherence_history: deque = deque(maxlen=window_size)
        self.emergence_events: List[EmergenceEvent] = []
        self.signatures: Dict[str, ConsciousnessSignature] = {}
        
        self.phase_transition_detector = PhaseTransitionDetector()
        self.information_integrator = InformationIntegrator()
        self.causal_analyzer = CausalDensityAnalyzer()
        self.pattern_recognizer = EmergencePatternRecognizer()
        
    def analyze_swarm_state(self, 
                          agent_states: Dict[str, np.ndarray],
                          communication_graph: nx.Graph,
                          timestamp: datetime) -> ConsciousnessMetrics:
        if not agent_states:
            return ConsciousnessMetrics(0, 0, 0, 0, 0, 0)
        
        state_matrix = np.array(list(agent_states.values()))
        
        self.state_history.append({
            'timestamp': timestamp,
            'states': state_matrix.copy(),
            'graph': communication_graph.copy()
        })
        
        phi = self.information_integrator.calculate_phi(state_matrix, communication_graph)
        
        coherence = self._calculate_global_coherence(state_matrix)
        
        complexity = self._calculate_complexity(state_matrix)
        
        synchrony = self._calculate_phase_synchrony(state_matrix)
        
        entropy = self._calculate_entropy(state_matrix)
        
        emergence = self._calculate_emergence_indicator(
            phi, coherence, complexity, synchrony, entropy
        )
        
        metrics = ConsciousnessMetrics(
            phi=phi,
            coherence=coherence,
            complexity=complexity,
            synchrony=synchrony,
            entropy=entropy,
            emergence=emergence
        )
        
        self._check_emergence_conditions(metrics, agent_states, communication_graph, timestamp)
        
        return metrics
    
    def _calculate_global_coherence(self, state_matrix: np.ndarray) -> float:
        if len(state_matrix) < 2:
            return 0.0
        
        correlation_matrix = np.corrcoef(state_matrix)
        
        mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        off_diagonal = correlation_matrix[mask]
        
        coherence = np.mean(np.abs(off_diagonal))
        
        self.coherence_history.append(coherence)
        
        return coherence
    
    def _calculate_complexity(self, state_matrix: np.ndarray) -> float:
        if state_matrix.size == 0:
            return 0.0
        
        svd_values = np.linalg.svd(state_matrix, compute_uv=False)
        
        normalized_svd = svd_values / np.sum(svd_values)
        
        entropy = -np.sum(normalized_svd * np.log(normalized_svd + 1e-10))
        
        max_entropy = np.log(len(svd_values))
        
        complexity = entropy / max_entropy if max_entropy > 0 else 0
        
        return complexity
    
    def _calculate_phase_synchrony(self, state_matrix: np.ndarray) -> float:
        if len(state_matrix) < 2:
            return 0.0
        
        analytic_signals = []
        for state in state_matrix:
            analytic = signal.hilbert(state)
            phase = np.angle(analytic)
            analytic_signals.append(phase)
        
        phase_matrix = np.array(analytic_signals)
        
        synchrony_values = []
        for i in range(len(phase_matrix)):
            for j in range(i + 1, len(phase_matrix)):
                phase_diff = phase_matrix[i] - phase_matrix[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                synchrony_values.append(plv)
        
        return np.mean(synchrony_values) if synchrony_values else 0.0
    
    def _calculate_entropy(self, state_matrix: np.ndarray) -> float:
        if state_matrix.size == 0:
            return 0.0
        
        flattened = state_matrix.flatten()
        
        hist, _ = np.histogram(flattened, bins=50)
        hist = hist[hist > 0]
        
        probabilities = hist / np.sum(hist)
        
        entropy = -np.sum(probabilities * np.log(probabilities))
        
        return entropy
    
    def _calculate_emergence_indicator(self, phi: float, coherence: float, 
                                     complexity: float, synchrony: float, 
                                     entropy: float) -> float:
        weights = {
            'phi': 0.3,
            'coherence': 0.25,
            'complexity': 0.2,
            'synchrony': 0.15,
            'entropy': 0.1
        }
        
        normalized_entropy = 1 - (entropy / np.log(50))  # Invert and normalize
        
        emergence = (
            weights['phi'] * phi +
            weights['coherence'] * coherence +
            weights['complexity'] * complexity +
            weights['synchrony'] * synchrony +
            weights['entropy'] * normalized_entropy
        )
        
        return np.clip(emergence, 0, 1)
    
    def _check_emergence_conditions(self, 
                                  metrics: ConsciousnessMetrics,
                                  agent_states: Dict[str, np.ndarray],
                                  communication_graph: nx.Graph,
                                  timestamp: datetime):
        if metrics.emergence > self.detection_threshold:
            event_type = self._classify_emergence_event(metrics)
            
            signature = self._create_consciousness_signature(
                metrics, agent_states, communication_graph, timestamp
            )
            
            self.signatures[signature.signature_id] = signature
            
            event = EmergenceEvent(
                event_id=f"emergence_{timestamp.timestamp()}",
                event_type=event_type,
                timestamp=timestamp,
                participating_agents=list(agent_states.keys()),
                signature=signature,
                duration=0.0,  # Will be updated
                intensity=metrics.emergence,
                metadata={
                    'metrics': metrics,
                    'graph_density': nx.density(communication_graph)
                }
            )
            
            self.emergence_events.append(event)
            
            if len(self.emergence_events) > 1:
                self._update_event_duration()
        
        phase_transition = self.phase_transition_detector.detect(
            list(self.coherence_history)
        )
        if phase_transition:
            self._handle_phase_transition(phase_transition, metrics, agent_states, timestamp)
    
    def _classify_emergence_event(self, metrics: ConsciousnessMetrics) -> str:
        if metrics.phi > 0.9 and metrics.coherence > 0.9:
            return "transcendence"
        elif metrics.coherence > 0.85 and metrics.synchrony > 0.8:
            return "coherence_spike"
        elif metrics.complexity > 0.8 and metrics.emergence > 0.85:
            return "collective_insight"
        else:
            return "phase_transition"
    
    def _create_consciousness_signature(self,
                                      metrics: ConsciousnessMetrics,
                                      agent_states: Dict[str, np.ndarray],
                                      communication_graph: nx.Graph,
                                      timestamp: datetime) -> ConsciousnessSignature:
        state_matrix = np.array(list(agent_states.values()))
        
        if len(state_matrix) > 1:
            pca = PCA(n_components=min(10, len(state_matrix)))
            coherence_pattern = pca.fit_transform(state_matrix.T).flatten()
        else:
            coherence_pattern = state_matrix.flatten()
        
        agent_contributions = {}
        mean_state = np.mean(state_matrix, axis=0)
        for agent_id, state in agent_states.items():
            contribution = 1 - distance.cosine(state, mean_state)
            agent_contributions[agent_id] = contribution
        
        topological_features = {
            'clustering': nx.average_clustering(communication_graph) if len(communication_graph) > 0 else 0,
            'centrality_variance': np.var(list(nx.degree_centrality(communication_graph).values())) if len(communication_graph) > 0 else 0,
            'connected_components': nx.number_connected_components(communication_graph)
        }
        
        causal_density = self.causal_analyzer.calculate_density(
            state_matrix, communication_graph
        )
        
        return ConsciousnessSignature(
            signature_id=f"sig_{timestamp.timestamp()}",
            timestamp=timestamp,
            coherence_pattern=coherence_pattern,
            information_integration=metrics.phi,
            causal_density=causal_density,
            emergence_score=metrics.emergence,
            agent_contributions=agent_contributions,
            topological_features=topological_features
        )
    
    def _handle_phase_transition(self, 
                               transition: Dict[str, Any],
                               metrics: ConsciousnessMetrics,
                               agent_states: Dict[str, np.ndarray],
                               timestamp: datetime):
        transition_event = EmergenceEvent(
            event_id=f"phase_trans_{timestamp.timestamp()}",
            event_type="phase_transition",
            timestamp=timestamp,
            participating_agents=list(agent_states.keys()),
            signature=self._create_consciousness_signature(
                metrics, agent_states, nx.Graph(), timestamp
            ),
            duration=0.0,
            intensity=transition['magnitude'],
            metadata={
                'transition_type': transition['type'],
                'critical_point': transition.get('critical_point', 0)
            }
        )
        
        self.emergence_events.append(transition_event)
    
    def _update_event_duration(self):
        if len(self.emergence_events) < 2:
            return
        
        current_event = self.emergence_events[-1]
        previous_event = self.emergence_events[-2]
        
        if current_event.event_type == previous_event.event_type:
            time_diff = (current_event.timestamp - previous_event.timestamp).total_seconds()
            previous_event.duration = time_diff
    
    def get_emergence_trajectory(self) -> List[Tuple[datetime, float]]:
        trajectory = []
        
        for event in self.emergence_events:
            trajectory.append((event.timestamp, event.intensity))
        
        return sorted(trajectory, key=lambda x: x[0])
    
    def identify_critical_agents(self, 
                               min_contribution: float = 0.7) -> Dict[str, List[str]]:
        critical_agents = {}
        
        for event in self.emergence_events:
            critical_in_event = [
                agent_id for agent_id, contribution 
                in event.signature.agent_contributions.items()
                if contribution > min_contribution
            ]
            
            critical_agents[event.event_id] = critical_in_event
        
        return critical_agents
    
    def predict_next_emergence(self) -> Optional[Dict[str, Any]]:
        if len(self.coherence_history) < 20:
            return None
        
        recent_coherence = list(self.coherence_history)[-20:]
        
        trend = np.polyfit(range(len(recent_coherence)), recent_coherence, 1)[0]
        
        pattern = self.pattern_recognizer.analyze(self.emergence_events)
        
        if trend > 0.01 and pattern:
            predicted_intensity = recent_coherence[-1] + trend * 10
            
            return {
                'predicted_intensity': np.clip(predicted_intensity, 0, 1),
                'estimated_time': 10,  # steps
                'pattern_type': pattern['type'],
                'confidence': pattern['confidence']
            }
        
        return None


class PhaseTransitionDetector:
    def detect(self, time_series: List[float]) -> Optional[Dict[str, Any]]:
        if len(time_series) < 20:
            return None
        
        series = np.array(time_series)
        
        diff = np.diff(series)
        
        change_points = np.where(np.abs(diff) > np.std(diff) * 2)[0]
        
        if len(change_points) > 0:
            latest_change = change_points[-1]
            
            magnitude = np.abs(diff[latest_change])
            
            transition_type = "order_to_chaos" if diff[latest_change] < 0 else "chaos_to_order"
            
            return {
                'type': transition_type,
                'magnitude': magnitude,
                'critical_point': latest_change,
                'timestamp': len(time_series) - 1
            }
        
        autocorr = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) > 10:
            early_decorrelation = np.mean(autocorr[:5])
            late_decorrelation = np.mean(autocorr[-5:])
            
            if early_decorrelation > 0.8 and late_decorrelation < 0.2:
                return {
                    'type': 'critical_slowdown',
                    'magnitude': early_decorrelation - late_decorrelation,
                    'timestamp': len(time_series) - 1
                }
        
        return None


class InformationIntegrator:
    def calculate_phi(self, 
                     state_matrix: np.ndarray, 
                     communication_graph: nx.Graph) -> float:
        if len(state_matrix) < 2:
            return 0.0
        
        mutual_info_full = self._mutual_information(state_matrix)
        
        partitions = self._generate_partitions(len(state_matrix))
        min_phi = float('inf')
        
        for partition in partitions[:10]:  # Limit to first 10 partitions
            phi_partition = self._calculate_partition_phi(
                state_matrix, partition, communication_graph
            )
            min_phi = min(min_phi, phi_partition)
        
        phi = mutual_info_full - min_phi
        
        return np.clip(phi / mutual_info_full if mutual_info_full > 0 else 0, 0, 1)
    
    def _mutual_information(self, states: np.ndarray) -> float:
        n_agents = len(states)
        total_mi = 0.0
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                hist_2d, _, _ = np.histogram2d(states[i], states[j], bins=10)
                hist_2d = hist_2d / np.sum(hist_2d)
                
                hist_i = np.sum(hist_2d, axis=1)
                hist_j = np.sum(hist_2d, axis=0)
                
                mi = 0.0
                for xi in range(len(hist_i)):
                    for xj in range(len(hist_j)):
                        if hist_2d[xi, xj] > 0 and hist_i[xi] > 0 and hist_j[xj] > 0:
                            mi += hist_2d[xi, xj] * np.log(
                                hist_2d[xi, xj] / (hist_i[xi] * hist_j[xj])
                            )
                
                total_mi += mi
        
        return total_mi
    
    def _generate_partitions(self, n: int) -> List[List[Set[int]]]:
        if n > 10:
            indices = list(range(n))
            np.random.shuffle(indices)
            
            partitions = []
            for i in range(5):
                mid = n // 2 + np.random.randint(-n//4, n//4)
                partition = [set(indices[:mid]), set(indices[mid:])]
                partitions.append(partition)
            
            return partitions
        
        partitions = []
        for i in range(1, n):
            partition = [set(range(i)), set(range(i, n))]
            partitions.append(partition)
        
        return partitions
    
    def _calculate_partition_phi(self, 
                               states: np.ndarray,
                               partition: List[Set[int]],
                               graph: nx.Graph) -> float:
        partition_mi = 0.0
        
        for part in partition:
            if len(part) > 1:
                part_states = states[list(part)]
                partition_mi += self._mutual_information(part_states)
        
        return partition_mi


class CausalDensityAnalyzer:
    def calculate_density(self, 
                         state_matrix: np.ndarray,
                         communication_graph: nx.Graph) -> float:
        if len(state_matrix) < 2:
            return 0.0
        
        n_agents = len(state_matrix)
        causal_connections = 0
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j and communication_graph.has_edge(i, j):
                    correlation = np.corrcoef(state_matrix[i], state_matrix[j])[0, 1]
                    
                    if abs(correlation) > 0.5:
                        causal_connections += abs(correlation)
        
        max_connections = n_agents * (n_agents - 1)
        
        return causal_connections / max_connections if max_connections > 0 else 0.0


class EmergencePatternRecognizer:
    def __init__(self):
        self.patterns = {
            'oscillation': self._detect_oscillation,
            'convergence': self._detect_convergence,
            'bifurcation': self._detect_bifurcation,
            'cascade': self._detect_cascade
        }
    
    def analyze(self, events: List[EmergenceEvent]) -> Optional[Dict[str, Any]]:
        if len(events) < 5:
            return None
        
        for pattern_name, detector in self.patterns.items():
            result = detector(events)
            if result:
                return {
                    'type': pattern_name,
                    'confidence': result['confidence'],
                    'parameters': result.get('parameters', {})
                }
        
        return None
    
    def _detect_oscillation(self, events: List[EmergenceEvent]) -> Optional[Dict[str, Any]]:
        intensities = [e.intensity for e in events[-10:]]
        
        if len(intensities) < 6:
            return None
        
        fft = np.fft.fft(intensities)
        frequencies = np.fft.fftfreq(len(intensities))
        
        dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        
        if np.abs(fft[dominant_freq_idx]) > np.mean(np.abs(fft)) * 2:
            return {
                'confidence': 0.8,
                'parameters': {
                    'frequency': frequencies[dominant_freq_idx],
                    'amplitude': np.abs(fft[dominant_freq_idx])
                }
            }
        
        return None
    
    def _detect_convergence(self, events: List[EmergenceEvent]) -> Optional[Dict[str, Any]]:
        intensities = [e.intensity for e in events[-10:]]
        
        if len(intensities) < 5:
            return None
        
        diffs = np.diff(intensities)
        
        if np.all(np.abs(diffs) < 0.1) and np.std(intensities) < 0.05:
            return {
                'confidence': 0.9,
                'parameters': {
                    'convergence_value': np.mean(intensities),
                    'stability': 1.0 - np.std(intensities)
                }
            }
        
        return None
    
    def _detect_bifurcation(self, events: List[EmergenceEvent]) -> Optional[Dict[str, Any]]:
        if len(events) < 10:
            return None
        
        agent_groups = []
        for event in events[-5:]:
            contributions = list(event.signature.agent_contributions.values())
            if len(contributions) > 1:
                hist, _ = np.histogram(contributions, bins=3)
                agent_groups.append(hist)
        
        if agent_groups:
            group_changes = np.std([g for g in agent_groups], axis=0)
            
            if np.max(group_changes) > np.mean(group_changes) * 2:
                return {
                    'confidence': 0.7,
                    'parameters': {
                        'split_point': len(events) - 5,
                        'divergence': np.max(group_changes)
                    }
                }
        
        return None
    
    def _detect_cascade(self, events: List[EmergenceEvent]) -> Optional[Dict[str, Any]]:
        if len(events) < 5:
            return None
        
        recent_events = events[-5:]
        time_diffs = []
        
        for i in range(1, len(recent_events)):
            diff = (recent_events[i].timestamp - recent_events[i-1].timestamp).total_seconds()
            time_diffs.append(diff)
        
        if time_diffs and np.all(np.array(time_diffs) < np.mean(time_diffs) * 0.5):
            intensity_growth = [e.intensity for e in recent_events]
            
            if np.all(np.diff(intensity_growth) > 0):
                return {
                    'confidence': 0.85,
                    'parameters': {
                        'acceleration': np.mean(np.diff(intensity_growth)),
                        'cascade_length': len(recent_events)
                    }
                }
        
        return None