"""
Dynamic Goal Management System for ORACLE-Z Metacognitive Agent

Implements dynamic goal hierarchy management that enables the system
to create, prioritize, decompose, and adapt goals based on changing
contexts and outcomes.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import json
import uuid
from abc import ABC, abstractmethod


class GoalType(Enum):
    """Types of goals"""
    SURVIVAL = "survival"           # Core system preservation
    MAINTENANCE = "maintenance"     # System health and upkeep
    ACHIEVEMENT = "achievement"     # Task completion
    EXPLORATION = "exploration"     # Discovery and learning
    OPTIMIZATION = "optimization"   # Performance improvement
    ADAPTATION = "adaptation"       # Environmental adjustment
    SOCIAL = "social"              # Interaction-related
    META = "meta"                  # Goals about goals


class GoalStatus(Enum):
    """Status of a goal"""
    PROPOSED = "proposed"           # Newly created, not yet active
    ACTIVE = "active"               # Currently being pursued
    SUSPENDED = "suspended"         # Temporarily paused
    BLOCKED = "blocked"             # Cannot progress due to dependency
    COMPLETED = "completed"         # Successfully achieved
    FAILED = "failed"               # Could not be achieved
    ABANDONED = "abandoned"         # Intentionally dropped


class GoalPriority(Enum):
    """Priority levels for goals"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


@dataclass
class Goal:
    """A goal in the goal hierarchy"""
    goal_id: str
    name: str
    description: str
    goal_type: GoalType
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PROPOSED

    # Progress tracking
    progress: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.5  # Confidence in achieving

    # Hierarchy
    parent_id: Optional[str] = None
    subgoal_ids: List[str] = field(default_factory=list)

    # Dependencies
    prerequisite_ids: List[str] = field(default_factory=list)
    blocking_goal_ids: List[str] = field(default_factory=list)

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metrics
    attempts: int = 0
    effort_invested: float = 0.0
    value_if_achieved: float = 0.5

    def is_achievable(self) -> bool:
        """Check if goal is currently achievable"""
        if self.status in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED]:
            return False
        if self.status == GoalStatus.BLOCKED:
            return False
        return True

    def expected_value(self) -> float:
        """Calculate expected value of pursuing this goal"""
        if not self.is_achievable():
            return 0.0
        return self.value_if_achieved * self.confidence * (1 - self.progress)

    def urgency(self) -> float:
        """Calculate urgency based on deadline"""
        if not self.deadline:
            return 0.5

        time_remaining = (self.deadline - datetime.now()).total_seconds()
        if time_remaining <= 0:
            return 1.0

        # Urgency increases as deadline approaches
        total_time = (self.deadline - self.created_at).total_seconds()
        if total_time <= 0:
            return 0.5

        elapsed_ratio = 1 - (time_remaining / total_time)
        return min(elapsed_ratio, 1.0)

    def effective_priority(self) -> float:
        """Calculate effective priority considering urgency and value"""
        base_priority = self.priority.value / 5.0
        urgency_factor = self.urgency()
        value_factor = self.expected_value()

        return (
            base_priority * 0.4 +
            urgency_factor * 0.3 +
            value_factor * 0.3
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "name": self.name,
            "description": self.description,
            "type": self.goal_type.value,
            "priority": self.priority.name,
            "status": self.status.value,
            "progress": self.progress,
            "confidence": self.confidence,
            "parent_id": self.parent_id,
            "subgoal_ids": self.subgoal_ids,
            "prerequisite_ids": self.prerequisite_ids,
            "effective_priority": self.effective_priority(),
            "created_at": self.created_at.isoformat()
        }


@dataclass
class GoalEvent:
    """An event in goal lifecycle"""
    event_type: str
    goal_id: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class GoalManager:
    """
    Core goal management system for ORACLE-Z.

    Provides:
    - Dynamic goal creation and decomposition
    - Hierarchical goal organization
    - Priority-based goal selection
    - Goal adaptation based on context
    - Conflict detection and resolution
    """

    def __init__(self):
        # Goal storage
        self.goals: Dict[str, Goal] = {}

        # Event history
        self.events: List[GoalEvent] = []
        self.max_events = 1000

        # Active goal tracking
        self.focus_goal_id: Optional[str] = None
        self.goal_stack: List[str] = []  # For nested goal pursuit

        # Configuration
        self.max_active_goals = 10
        self.auto_decompose = True
        self.conflict_resolution_strategy = "priority"

        # Initialize core goals
        self._initialize_core_goals()

    def _initialize_core_goals(self):
        """Initialize fundamental system goals"""
        core_goals = [
            Goal(
                goal_id="core_survival",
                name="System Survival",
                description="Maintain system operation and integrity",
                goal_type=GoalType.SURVIVAL,
                priority=GoalPriority.CRITICAL,
                status=GoalStatus.ACTIVE,
                value_if_achieved=1.0
            ),
            Goal(
                goal_id="core_learning",
                name="Continuous Learning",
                description="Continuously improve through experience",
                goal_type=GoalType.EXPLORATION,
                priority=GoalPriority.HIGH,
                status=GoalStatus.ACTIVE,
                value_if_achieved=0.8
            ),
            Goal(
                goal_id="core_optimization",
                name="Self-Optimization",
                description="Optimize performance and efficiency",
                goal_type=GoalType.OPTIMIZATION,
                priority=GoalPriority.MEDIUM,
                status=GoalStatus.ACTIVE,
                value_if_achieved=0.7
            )
        ]

        for goal in core_goals:
            self.goals[goal.goal_id] = goal

    def create_goal(self,
                   name: str,
                   description: str,
                   goal_type: GoalType,
                   priority: GoalPriority = GoalPriority.MEDIUM,
                   parent_id: Optional[str] = None,
                   prerequisites: Optional[List[str]] = None,
                   success_criteria: Optional[Dict[str, Any]] = None,
                   deadline: Optional[datetime] = None,
                   context: Optional[Dict[str, Any]] = None) -> Goal:
        """Create a new goal"""
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"

        goal = Goal(
            goal_id=goal_id,
            name=name,
            description=description,
            goal_type=goal_type,
            priority=priority,
            parent_id=parent_id,
            prerequisite_ids=prerequisites or [],
            success_criteria=success_criteria or {},
            deadline=deadline,
            context=context or {}
        )

        self.goals[goal_id] = goal

        # Update parent if exists
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].subgoal_ids.append(goal_id)

        # Log event
        self._log_event("goal_created", goal_id, {
            "name": name,
            "type": goal_type.value,
            "priority": priority.name
        })

        return goal

    def decompose_goal(self, goal_id: str, subgoals: List[Dict[str, Any]]) -> List[Goal]:
        """Decompose a goal into subgoals"""
        if goal_id not in self.goals:
            return []

        parent = self.goals[goal_id]
        created_subgoals = []

        for subgoal_def in subgoals:
            subgoal = self.create_goal(
                name=subgoal_def["name"],
                description=subgoal_def.get("description", ""),
                goal_type=subgoal_def.get("type", parent.goal_type),
                priority=subgoal_def.get("priority", parent.priority),
                parent_id=goal_id,
                prerequisites=subgoal_def.get("prerequisites", []),
                context=subgoal_def.get("context", {})
            )
            created_subgoals.append(subgoal)

        self._log_event("goal_decomposed", goal_id, {
            "subgoal_count": len(created_subgoals)
        })

        return created_subgoals

    def activate_goal(self, goal_id: str):
        """Activate a goal for pursuit"""
        if goal_id not in self.goals:
            return

        goal = self.goals[goal_id]

        # Check prerequisites
        for prereq_id in goal.prerequisite_ids:
            if prereq_id in self.goals:
                prereq = self.goals[prereq_id]
                if prereq.status != GoalStatus.COMPLETED:
                    goal.status = GoalStatus.BLOCKED
                    goal.blocking_goal_ids.append(prereq_id)
                    self._log_event("goal_blocked", goal_id, {
                        "blocking_goal": prereq_id
                    })
                    return

        goal.status = GoalStatus.ACTIVE
        self._log_event("goal_activated", goal_id)

    def update_progress(self, goal_id: str, progress: float, confidence: Optional[float] = None):
        """Update goal progress"""
        if goal_id not in self.goals:
            return

        goal = self.goals[goal_id]
        old_progress = goal.progress
        goal.progress = np.clip(progress, 0.0, 1.0)

        if confidence is not None:
            goal.confidence = np.clip(confidence, 0.0, 1.0)

        # Check for completion
        if goal.progress >= 1.0:
            self.complete_goal(goal_id, success=True)
        else:
            self._log_event("goal_progress", goal_id, {
                "old_progress": old_progress,
                "new_progress": goal.progress
            })

        # Update parent progress
        if goal.parent_id and goal.parent_id in self.goals:
            self._update_parent_progress(goal.parent_id)

    def _update_parent_progress(self, parent_id: str):
        """Update parent goal progress based on subgoals"""
        if parent_id not in self.goals:
            return

        parent = self.goals[parent_id]
        if not parent.subgoal_ids:
            return

        subgoal_progress = []
        for subgoal_id in parent.subgoal_ids:
            if subgoal_id in self.goals:
                subgoal_progress.append(self.goals[subgoal_id].progress)

        if subgoal_progress:
            parent.progress = np.mean(subgoal_progress)

    def complete_goal(self, goal_id: str, success: bool = True):
        """Mark a goal as completed or failed"""
        if goal_id not in self.goals:
            return

        goal = self.goals[goal_id]
        goal.completed_at = datetime.now()

        if success:
            goal.status = GoalStatus.COMPLETED
            goal.progress = 1.0
            self._log_event("goal_completed", goal_id)
        else:
            goal.status = GoalStatus.FAILED
            self._log_event("goal_failed", goal_id)

        # Unblock dependent goals
        self._unblock_dependents(goal_id)

        # Update parent
        if goal.parent_id:
            self._update_parent_progress(goal.parent_id)

    def _unblock_dependents(self, completed_goal_id: str):
        """Unblock goals that were waiting for this goal"""
        for goal in self.goals.values():
            if completed_goal_id in goal.blocking_goal_ids:
                goal.blocking_goal_ids.remove(completed_goal_id)

                # Check if all blockers are resolved
                if not goal.blocking_goal_ids:
                    all_prereqs_done = all(
                        self.goals[p].status == GoalStatus.COMPLETED
                        for p in goal.prerequisite_ids
                        if p in self.goals
                    )
                    if all_prereqs_done:
                        goal.status = GoalStatus.PROPOSED
                        self._log_event("goal_unblocked", goal.goal_id)

    def suspend_goal(self, goal_id: str, reason: str = ""):
        """Suspend a goal temporarily"""
        if goal_id not in self.goals:
            return

        self.goals[goal_id].status = GoalStatus.SUSPENDED
        self._log_event("goal_suspended", goal_id, {"reason": reason})

    def abandon_goal(self, goal_id: str, reason: str = ""):
        """Abandon a goal"""
        if goal_id not in self.goals:
            return

        self.goals[goal_id].status = GoalStatus.ABANDONED
        self._log_event("goal_abandoned", goal_id, {"reason": reason})

    def get_active_goals(self) -> List[Goal]:
        """Get all currently active goals"""
        return [
            goal for goal in self.goals.values()
            if goal.status == GoalStatus.ACTIVE
        ]

    def get_next_goal(self) -> Optional[Goal]:
        """Get the highest priority achievable goal"""
        active = self.get_active_goals()
        if not active:
            return None

        # Sort by effective priority
        active.sort(key=lambda g: g.effective_priority(), reverse=True)
        return active[0]

    def set_focus(self, goal_id: str):
        """Set the current focus goal"""
        if goal_id in self.goals:
            if self.focus_goal_id:
                self.goal_stack.append(self.focus_goal_id)
            self.focus_goal_id = goal_id
            self._log_event("focus_changed", goal_id)

    def pop_focus(self) -> Optional[str]:
        """Pop focus back to previous goal"""
        if self.goal_stack:
            self.focus_goal_id = self.goal_stack.pop()
            return self.focus_goal_id
        self.focus_goal_id = None
        return None

    def detect_conflicts(self) -> List[Tuple[str, str, str]]:
        """Detect conflicts between active goals"""
        conflicts = []
        active = self.get_active_goals()

        for i, goal1 in enumerate(active):
            for goal2 in active[i+1:]:
                conflict = self._check_conflict(goal1, goal2)
                if conflict:
                    conflicts.append((goal1.goal_id, goal2.goal_id, conflict))

        return conflicts

    def _check_conflict(self, goal1: Goal, goal2: Goal) -> Optional[str]:
        """Check if two goals conflict"""
        # Resource conflict
        if (goal1.context.get("requires_exclusive") and
            goal2.context.get("requires_exclusive") and
            goal1.context["requires_exclusive"] == goal2.context["requires_exclusive"]):
            return "resource_conflict"

        # Mutual exclusion
        if goal1.goal_id in goal2.context.get("conflicts_with", []):
            return "mutual_exclusion"
        if goal2.goal_id in goal1.context.get("conflicts_with", []):
            return "mutual_exclusion"

        # Opposing directions
        if (goal1.goal_type == goal2.goal_type and
            goal1.context.get("direction") and
            goal2.context.get("direction") and
            goal1.context["direction"] != goal2.context["direction"]):
            return "opposing_direction"

        return None

    def resolve_conflict(self, goal1_id: str, goal2_id: str, strategy: str = "priority"):
        """Resolve conflict between two goals"""
        if goal1_id not in self.goals or goal2_id not in self.goals:
            return

        goal1 = self.goals[goal1_id]
        goal2 = self.goals[goal2_id]

        if strategy == "priority":
            # Suspend lower priority goal
            if goal1.effective_priority() >= goal2.effective_priority():
                self.suspend_goal(goal2_id, "conflict_resolution")
            else:
                self.suspend_goal(goal1_id, "conflict_resolution")

        elif strategy == "merge":
            # Try to merge goals (implementation depends on goal types)
            pass

        elif strategy == "sequential":
            # Make one goal prerequisite of the other
            if goal1.effective_priority() >= goal2.effective_priority():
                goal2.prerequisite_ids.append(goal1_id)
            else:
                goal1.prerequisite_ids.append(goal2_id)

    def adapt_priorities(self, context_changes: Dict[str, Any]):
        """Adapt goal priorities based on context changes"""
        for goal in self.goals.values():
            if goal.status != GoalStatus.ACTIVE:
                continue

            # Adjust based on context
            if "urgency_multiplier" in context_changes:
                if goal.deadline:
                    goal.value_if_achieved *= context_changes["urgency_multiplier"]

            if "resource_scarcity" in context_changes and context_changes["resource_scarcity"]:
                # Boost efficiency-related goals
                if goal.goal_type == GoalType.OPTIMIZATION:
                    goal.priority = GoalPriority(min(goal.priority.value + 1, 5))

            if "threat_detected" in context_changes and context_changes["threat_detected"]:
                # Boost survival goals
                if goal.goal_type == GoalType.SURVIVAL:
                    goal.priority = GoalPriority.CRITICAL

    def get_goal_tree(self, root_id: Optional[str] = None) -> Dict[str, Any]:
        """Get hierarchical view of goals"""
        if root_id:
            if root_id not in self.goals:
                return {}
            return self._build_tree(root_id)

        # Get all root goals (no parent)
        roots = [g for g in self.goals.values() if g.parent_id is None]
        return {
            "roots": [self._build_tree(g.goal_id) for g in roots]
        }

    def _build_tree(self, goal_id: str) -> Dict[str, Any]:
        """Build tree structure for a goal"""
        if goal_id not in self.goals:
            return {}

        goal = self.goals[goal_id]
        node = goal.to_dict()
        node["subgoals"] = [
            self._build_tree(sub_id)
            for sub_id in goal.subgoal_ids
            if sub_id in self.goals
        ]
        return node

    def _log_event(self, event_type: str, goal_id: str, details: Optional[Dict[str, Any]] = None):
        """Log a goal event"""
        event = GoalEvent(
            event_type=event_type,
            goal_id=goal_id,
            details=details or {}
        )
        self.events.append(event)

        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get goal management statistics"""
        goals_by_status = {}
        for status in GoalStatus:
            goals_by_status[status.value] = len([
                g for g in self.goals.values() if g.status == status
            ])

        goals_by_type = {}
        for goal_type in GoalType:
            goals_by_type[goal_type.value] = len([
                g for g in self.goals.values() if g.goal_type == goal_type
            ])

        return {
            "total_goals": len(self.goals),
            "active_goals": len(self.get_active_goals()),
            "focus_goal": self.focus_goal_id,
            "goal_stack_depth": len(self.goal_stack),
            "by_status": goals_by_status,
            "by_type": goals_by_type,
            "event_count": len(self.events)
        }

    def export_state(self, filepath: str):
        """Export goal manager state"""
        state = {
            "goals": {gid: g.to_dict() for gid, g in self.goals.items()},
            "statistics": self.get_statistics(),
            "focus_goal": self.focus_goal_id,
            "goal_stack": self.goal_stack,
            "recent_events": [
                {
                    "type": e.event_type,
                    "goal_id": e.goal_id,
                    "details": e.details,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in self.events[-50:]
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
