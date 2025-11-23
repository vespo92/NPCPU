"""
Deep Introspection System for ORACLE-Z Metacognitive Agent

Implements deep state inspection capabilities that allow the system
to examine its internal cognitive mechanisms, processing states,
and emergent patterns.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Generator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import json
import hashlib
from abc import ABC, abstractmethod


class IntrospectionDepth(Enum):
    """Depth levels for introspection"""
    SURFACE = "surface"         # Basic state observation
    SHALLOW = "shallow"         # Process-level inspection
    MODERATE = "moderate"       # Mechanism analysis
    DEEP = "deep"               # Structural examination
    PROFOUND = "profound"       # Emergent pattern analysis


class CognitiveProcess(Enum):
    """Identifiable cognitive processes"""
    ATTENTION = "attention"
    WORKING_MEMORY = "working_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    PATTERN_RECOGNITION = "pattern_recognition"
    INFERENCE = "inference"
    PLANNING = "planning"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    METACOGNITION = "metacognition"
    EMOTIONAL_PROCESSING = "emotional_processing"


class ProcessState(Enum):
    """States of cognitive processes"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETING = "completing"
    ERROR = "error"
    SUSPENDED = "suspended"


@dataclass
class ProcessSnapshot:
    """Snapshot of a cognitive process state"""
    process: CognitiveProcess
    state: ProcessState
    activation_level: float  # 0.0 to 1.0
    processing_load: float  # 0.0 to 1.0
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    internal_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_vector(self) -> np.ndarray:
        """Convert to vector representation"""
        return np.array([
            list(ProcessState).index(self.state) / len(ProcessState),
            self.activation_level,
            self.processing_load
        ])


@dataclass
class IntrospectionResult:
    """Result of an introspection operation"""
    depth: IntrospectionDepth
    target: str
    findings: Dict[str, Any]
    patterns: List[str]
    anomalies: List[str]
    insights: List[str]
    confidence: float
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth": self.depth.value,
            "target": self.target,
            "findings": self.findings,
            "patterns": self.patterns,
            "anomalies": self.anomalies,
            "insights": self.insights,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CognitiveTrace:
    """Trace of cognitive activity over time"""
    trace_id: str
    process: CognitiveProcess
    events: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def add_event(self, event_type: str, data: Dict[str, Any]):
        """Add an event to the trace"""
        self.events.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

    def close(self):
        """Close the trace"""
        self.end_time = datetime.now()


class IntrospectionEngine:
    """
    Core introspection engine for ORACLE-Z.

    Provides deep state inspection capabilities:
    - Process state monitoring
    - Cognitive trace analysis
    - Pattern detection
    - Anomaly identification
    - Emergent behavior analysis
    """

    def __init__(self):
        # Process monitoring
        self.process_snapshots: Dict[CognitiveProcess, List[ProcessSnapshot]] = {
            p: [] for p in CognitiveProcess
        }
        self.max_snapshots = 1000

        # Trace management
        self.active_traces: Dict[str, CognitiveTrace] = {}
        self.completed_traces: List[CognitiveTrace] = []

        # Introspection history
        self.introspection_history: List[IntrospectionResult] = []

        # Pattern detection
        self.known_patterns: Dict[str, Dict[str, Any]] = {}
        self.pattern_thresholds: Dict[str, float] = {}

        # Anomaly detection
        self.baseline_metrics: Dict[str, np.ndarray] = {}
        self.anomaly_threshold: float = 2.0  # Standard deviations

    def capture_snapshot(self,
                        process: CognitiveProcess,
                        state: ProcessState,
                        activation_level: float,
                        processing_load: float,
                        inputs: Optional[Dict[str, Any]] = None,
                        outputs: Optional[Dict[str, Any]] = None,
                        internal_state: Optional[Dict[str, Any]] = None):
        """Capture a snapshot of a cognitive process"""
        snapshot = ProcessSnapshot(
            process=process,
            state=state,
            activation_level=activation_level,
            processing_load=processing_load,
            inputs=inputs or {},
            outputs=outputs or {},
            internal_state=internal_state or {}
        )

        self.process_snapshots[process].append(snapshot)

        # Limit history
        if len(self.process_snapshots[process]) > self.max_snapshots:
            self.process_snapshots[process] = self.process_snapshots[process][-self.max_snapshots:]

    def start_trace(self, process: CognitiveProcess) -> str:
        """Start a new cognitive trace"""
        trace_id = f"trace_{process.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.active_traces[trace_id] = CognitiveTrace(
            trace_id=trace_id,
            process=process
        )
        return trace_id

    def add_trace_event(self, trace_id: str, event_type: str, data: Dict[str, Any]):
        """Add an event to an active trace"""
        if trace_id in self.active_traces:
            self.active_traces[trace_id].add_event(event_type, data)

    def end_trace(self, trace_id: str):
        """End and archive a trace"""
        if trace_id in self.active_traces:
            trace = self.active_traces.pop(trace_id)
            trace.close()
            self.completed_traces.append(trace)

            # Limit history
            if len(self.completed_traces) > 100:
                self.completed_traces = self.completed_traces[-100:]

    def introspect(self,
                   target: str = "all",
                   depth: IntrospectionDepth = IntrospectionDepth.MODERATE) -> IntrospectionResult:
        """
        Perform introspection at the specified depth.

        Args:
            target: What to introspect ("all", process name, or specific component)
            depth: How deep to introspect

        Returns:
            IntrospectionResult with findings
        """
        start_time = datetime.now()

        findings = {}
        patterns = []
        anomalies = []
        insights = []

        if target == "all" or target in [p.value for p in CognitiveProcess]:
            # Introspect cognitive processes
            if target == "all":
                processes = list(CognitiveProcess)
            else:
                processes = [CognitiveProcess(target)]

            for process in processes:
                process_findings = self._introspect_process(process, depth)
                findings[process.value] = process_findings

                # Detect patterns
                process_patterns = self._detect_patterns(process, depth)
                patterns.extend(process_patterns)

                # Detect anomalies
                process_anomalies = self._detect_anomalies(process)
                anomalies.extend(process_anomalies)

        # Generate insights based on depth
        if depth in [IntrospectionDepth.DEEP, IntrospectionDepth.PROFOUND]:
            insights = self._generate_insights(findings, patterns, anomalies)

        # Calculate confidence based on data availability
        confidence = self._calculate_introspection_confidence(target, depth)

        duration = (datetime.now() - start_time).total_seconds() * 1000

        result = IntrospectionResult(
            depth=depth,
            target=target,
            findings=findings,
            patterns=patterns,
            anomalies=anomalies,
            insights=insights,
            confidence=confidence,
            duration_ms=duration
        )

        self.introspection_history.append(result)
        return result

    def _introspect_process(self,
                           process: CognitiveProcess,
                           depth: IntrospectionDepth) -> Dict[str, Any]:
        """Introspect a specific cognitive process"""
        snapshots = self.process_snapshots.get(process, [])

        if not snapshots:
            return {"status": "no_data"}

        findings = {
            "snapshot_count": len(snapshots),
            "current_state": snapshots[-1].state.value if snapshots else "unknown"
        }

        if depth.value in ["shallow", "moderate", "deep", "profound"]:
            # Add activation statistics
            activations = [s.activation_level for s in snapshots[-100:]]
            loads = [s.processing_load for s in snapshots[-100:]]

            findings["activation"] = {
                "current": snapshots[-1].activation_level,
                "mean": np.mean(activations),
                "std": np.std(activations),
                "trend": self._calculate_trend(activations)
            }

            findings["load"] = {
                "current": snapshots[-1].processing_load,
                "mean": np.mean(loads),
                "std": np.std(loads),
                "trend": self._calculate_trend(loads)
            }

        if depth.value in ["moderate", "deep", "profound"]:
            # Add state transition analysis
            state_transitions = self._analyze_state_transitions(snapshots[-100:])
            findings["state_transitions"] = state_transitions

        if depth.value in ["deep", "profound"]:
            # Add internal state analysis
            findings["internal_state_summary"] = self._summarize_internal_states(snapshots[-50:])

        if depth == IntrospectionDepth.PROFOUND:
            # Add emergent behavior analysis
            findings["emergent_behaviors"] = self._detect_emergent_behaviors(snapshots)

        return findings

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"

        slope = np.polyfit(range(len(values)), values, 1)[0]

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _analyze_state_transitions(self, snapshots: List[ProcessSnapshot]) -> Dict[str, Any]:
        """Analyze state transitions"""
        if len(snapshots) < 2:
            return {}

        transitions = {}
        for i in range(1, len(snapshots)):
            prev_state = snapshots[i-1].state.value
            curr_state = snapshots[i].state.value

            if prev_state != curr_state:
                key = f"{prev_state} -> {curr_state}"
                transitions[key] = transitions.get(key, 0) + 1

        return {
            "transition_counts": transitions,
            "total_transitions": sum(transitions.values()),
            "most_common": max(transitions, key=transitions.get) if transitions else None
        }

    def _summarize_internal_states(self, snapshots: List[ProcessSnapshot]) -> Dict[str, Any]:
        """Summarize internal states"""
        if not snapshots:
            return {}

        # Collect all keys across internal states
        all_keys = set()
        for s in snapshots:
            all_keys.update(s.internal_state.keys())

        summary = {}
        for key in list(all_keys)[:10]:  # Limit to top 10 keys
            values = [
                s.internal_state.get(key)
                for s in snapshots
                if key in s.internal_state
            ]
            if values:
                if all(isinstance(v, (int, float)) for v in values):
                    summary[key] = {
                        "type": "numeric",
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }
                else:
                    summary[key] = {
                        "type": "categorical",
                        "unique_values": len(set(str(v) for v in values))
                    }

        return summary

    def _detect_emergent_behaviors(self, snapshots: List[ProcessSnapshot]) -> List[Dict[str, Any]]:
        """Detect emergent behaviors in snapshot sequences"""
        behaviors = []

        if len(snapshots) < 20:
            return behaviors

        # Detect oscillation patterns
        activations = [s.activation_level for s in snapshots[-50:]]
        fft = np.fft.fft(activations)
        frequencies = np.abs(fft[:len(fft)//2])

        if np.max(frequencies[1:]) > np.mean(frequencies) * 3:
            dominant_freq = np.argmax(frequencies[1:]) + 1
            behaviors.append({
                "type": "oscillation",
                "frequency": dominant_freq,
                "strength": float(np.max(frequencies[1:]) / np.mean(frequencies))
            })

        # Detect synchronization between processes
        # (This would need cross-process analysis)

        # Detect phase transitions
        loads = [s.processing_load for s in snapshots[-50:]]
        if np.std(loads[:25]) < 0.1 and np.std(loads[25:]) < 0.1:
            if abs(np.mean(loads[:25]) - np.mean(loads[25:])) > 0.3:
                behaviors.append({
                    "type": "phase_transition",
                    "from_state": np.mean(loads[:25]),
                    "to_state": np.mean(loads[25:])
                })

        return behaviors

    def _detect_patterns(self, process: CognitiveProcess, depth: IntrospectionDepth) -> List[str]:
        """Detect patterns in process behavior"""
        patterns = []
        snapshots = self.process_snapshots.get(process, [])

        if len(snapshots) < 10:
            return patterns

        recent = snapshots[-50:]

        # Pattern: Sustained high activation
        high_activation = sum(1 for s in recent if s.activation_level > 0.8)
        if high_activation > len(recent) * 0.7:
            patterns.append(f"{process.value}: sustained_high_activation")

        # Pattern: Frequent state changes
        state_changes = sum(
            1 for i in range(1, len(recent))
            if recent[i].state != recent[i-1].state
        )
        if state_changes > len(recent) * 0.3:
            patterns.append(f"{process.value}: high_state_volatility")

        # Pattern: Load spikes
        loads = [s.processing_load for s in recent]
        if max(loads) > 0.9 and np.std(loads) > 0.2:
            patterns.append(f"{process.value}: load_spiking")

        return patterns

    def _detect_anomalies(self, process: CognitiveProcess) -> List[str]:
        """Detect anomalies in process behavior"""
        anomalies = []
        snapshots = self.process_snapshots.get(process, [])

        if len(snapshots) < 20:
            return anomalies

        # Build baseline if not exists
        baseline_key = process.value
        if baseline_key not in self.baseline_metrics:
            baseline_snapshots = snapshots[:min(100, len(snapshots)//2)]
            if baseline_snapshots:
                self.baseline_metrics[baseline_key] = np.array([
                    [s.activation_level, s.processing_load]
                    for s in baseline_snapshots
                ])

        if baseline_key in self.baseline_metrics:
            baseline = self.baseline_metrics[baseline_key]
            mean = np.mean(baseline, axis=0)
            std = np.std(baseline, axis=0) + 1e-6

            # Check recent snapshots for anomalies
            recent = snapshots[-10:]
            for s in recent:
                current = np.array([s.activation_level, s.processing_load])
                z_scores = np.abs((current - mean) / std)

                if z_scores[0] > self.anomaly_threshold:
                    anomalies.append(f"{process.value}: anomalous_activation ({z_scores[0]:.2f}σ)")

                if z_scores[1] > self.anomaly_threshold:
                    anomalies.append(f"{process.value}: anomalous_load ({z_scores[1]:.2f}σ)")

        # Check for error states
        error_count = sum(1 for s in snapshots[-10:] if s.state == ProcessState.ERROR)
        if error_count > 2:
            anomalies.append(f"{process.value}: frequent_errors ({error_count}/10)")

        return anomalies

    def _generate_insights(self,
                          findings: Dict[str, Any],
                          patterns: List[str],
                          anomalies: List[str]) -> List[str]:
        """Generate high-level insights from introspection data"""
        insights = []

        # Cross-process insights
        active_processes = [
            p for p, f in findings.items()
            if isinstance(f, dict) and f.get("current_state") == "active"
        ]

        if len(active_processes) > 5:
            insights.append(
                f"High cognitive load: {len(active_processes)} processes currently active"
            )

        # Pattern-based insights
        volatility_count = sum(1 for p in patterns if "volatility" in p)
        if volatility_count > 2:
            insights.append(
                "System shows signs of instability across multiple processes"
            )

        # Anomaly-based insights
        if len(anomalies) > 3:
            insights.append(
                f"Multiple anomalies detected ({len(anomalies)}). Consider system health check."
            )

        # Resource insights
        high_load_processes = [
            p for p, f in findings.items()
            if isinstance(f, dict) and
            isinstance(f.get("load"), dict) and
            f["load"].get("current", 0) > 0.8
        ]

        if high_load_processes:
            insights.append(
                f"Resource bottleneck in: {', '.join(high_load_processes)}"
            )

        return insights

    def _calculate_introspection_confidence(self, target: str, depth: IntrospectionDepth) -> float:
        """Calculate confidence in introspection results"""
        confidence = 0.5

        # More data = higher confidence
        if target == "all":
            total_snapshots = sum(len(s) for s in self.process_snapshots.values())
        else:
            try:
                process = CognitiveProcess(target)
                total_snapshots = len(self.process_snapshots.get(process, []))
            except ValueError:
                total_snapshots = 0

        data_confidence = min(total_snapshots / 100, 1.0)
        confidence = confidence * 0.5 + data_confidence * 0.5

        # Deeper introspection has lower confidence (more inference)
        depth_penalty = {
            IntrospectionDepth.SURFACE: 0.0,
            IntrospectionDepth.SHALLOW: 0.05,
            IntrospectionDepth.MODERATE: 0.1,
            IntrospectionDepth.DEEP: 0.15,
            IntrospectionDepth.PROFOUND: 0.2
        }

        confidence -= depth_penalty.get(depth, 0)

        return np.clip(confidence, 0.1, 0.95)

    def get_process_summary(self, process: CognitiveProcess) -> Dict[str, Any]:
        """Get a summary of a specific process"""
        snapshots = self.process_snapshots.get(process, [])

        if not snapshots:
            return {"process": process.value, "status": "no_data"}

        recent = snapshots[-10:]

        return {
            "process": process.value,
            "current_state": recent[-1].state.value,
            "activation": recent[-1].activation_level,
            "load": recent[-1].processing_load,
            "snapshot_count": len(snapshots),
            "recent_states": [s.state.value for s in recent]
        }

    def get_system_overview(self) -> Dict[str, Any]:
        """Get overview of all cognitive processes"""
        overview = {
            "timestamp": datetime.now().isoformat(),
            "processes": {},
            "active_traces": len(self.active_traces),
            "completed_traces": len(self.completed_traces),
            "introspection_count": len(self.introspection_history)
        }

        for process in CognitiveProcess:
            overview["processes"][process.value] = self.get_process_summary(process)

        return overview

    def export_state(self, filepath: str):
        """Export introspection engine state"""
        state = {
            "overview": self.get_system_overview(),
            "recent_introspections": [
                r.to_dict() for r in self.introspection_history[-10:]
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
