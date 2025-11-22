"""
Reinforcement Learning System for NPCPU

A Q-learning based behavior adaptation system for organisms:
- State-action value estimation (Q-values)
- Exploration vs exploitation strategies
- Experience replay for stable learning
- Policy updates with configurable learning rates

Example:
    from core.simple_organism import SimpleOrganism
    from learning.reinforcement import ReinforcementLearner

    organism = SimpleOrganism("Learner")
    learner = ReinforcementLearner(
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=0.3
    )
    organism.add_subsystem(learner)

    # During simulation
    state = learner.get_state_key({"energy": 50, "threat": 0.2})
    action = learner.select_action(state, ["eat", "flee", "explore"])

    # After receiving reward
    next_state = learner.get_state_key({"energy": 70, "threat": 0.0})
    learner.learn_from_experience(state, action, reward=10.0, next_state=next_state)
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import random
import json
import hashlib

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.abstractions import BaseSubsystem, BaseOrganism
from core.events import get_event_bus


class ExplorationStrategy(Enum):
    """Strategies for exploration vs exploitation"""
    EPSILON_GREEDY = auto()  # Random action with probability epsilon
    SOFTMAX = auto()         # Probability based on Q-values
    UCB = auto()             # Upper Confidence Bound
    DECAYING_EPSILON = auto() # Epsilon decreases over time


@dataclass
class Experience:
    """
    A single experience tuple for replay.

    Attributes:
        state: State key (hashed representation)
        action: Action taken
        reward: Reward received
        next_state: Resulting state key
        done: Whether episode ended
        tick: Simulation tick when experience occurred
    """
    state: str
    action: str
    reward: float
    next_state: str
    done: bool = False
    tick: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "tick": self.tick,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        return cls(
            state=data["state"],
            action=data["action"],
            reward=data["reward"],
            next_state=data["next_state"],
            done=data.get("done", False),
            tick=data.get("tick", 0),
        )


class ReinforcementLearner(BaseSubsystem):
    """
    Q-learning based behavior adaptation system.

    Features:
    - State-action value estimation (Q-table)
    - Multiple exploration strategies
    - Experience replay buffer
    - Eligibility traces for faster learning
    - Configurable hyperparameters

    The learner maintains a Q-table mapping (state, action) pairs to expected
    future rewards. Through experience, it learns which actions are most
    beneficial in different states.

    Example:
        from core.simple_organism import SimpleOrganism
        from learning.reinforcement import ReinforcementLearner, ExplorationStrategy

        organism = SimpleOrganism("Agent")
        learner = ReinforcementLearner(
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=0.2,
            exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
            replay_buffer_size=1000,
        )
        organism.add_subsystem(learner)

        # Define available actions
        actions = ["eat", "flee", "explore", "rest"]

        # Simulation loop
        state = get_current_state()
        state_key = learner.get_state_key(state)

        action = learner.select_action(state_key, actions)
        # ... execute action, observe reward and new state ...

        new_state = get_current_state()
        new_state_key = learner.get_state_key(new_state)

        learner.learn_from_experience(
            state=state_key,
            action=action,
            reward=reward,
            next_state=new_state_key
        )
    """

    def __init__(
        self,
        owner: Optional[BaseOrganism] = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.3,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        replay_buffer_size: int = 10000,
        batch_size: int = 32,
        use_experience_replay: bool = True,
        eligibility_trace_decay: float = 0.9,
    ):
        """
        Initialize reinforcement learner.

        Args:
            owner: The organism that owns this learner
            learning_rate: Alpha - how quickly to update Q-values (0.0 to 1.0)
            discount_factor: Gamma - importance of future rewards (0.0 to 1.0)
            exploration_rate: Epsilon - probability of random action (0.0 to 1.0)
            exploration_strategy: Strategy for balancing exploration/exploitation
            exploration_decay: Rate at which exploration decreases
            min_exploration_rate: Minimum exploration rate
            replay_buffer_size: Maximum experiences to store
            batch_size: Number of experiences per replay batch
            use_experience_replay: Whether to use experience replay
            eligibility_trace_decay: Lambda - decay rate for eligibility traces
        """
        super().__init__("reinforcement_learner", owner)

        # Learning hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.initial_exploration_rate = exploration_rate
        self.exploration_strategy = exploration_strategy
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Experience replay settings
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.use_experience_replay = use_experience_replay

        # Eligibility traces (for TD(λ))
        self.eligibility_trace_decay = eligibility_trace_decay
        self._eligibility_traces: Dict[Tuple[str, str], float] = {}

        # Q-table: maps (state, action) -> Q-value
        self._q_table: Dict[str, Dict[str, float]] = {}

        # Action visit counts (for UCB)
        self._action_counts: Dict[str, Dict[str, int]] = {}
        self._total_steps: int = 0

        # Experience replay buffer
        self._replay_buffer: deque = deque(maxlen=replay_buffer_size)

        # Current tick
        self._current_tick: int = 0

        # Statistics
        self._stats = {
            "total_experiences": 0,
            "total_updates": 0,
            "total_replays": 0,
            "average_reward": 0.0,
            "cumulative_reward": 0.0,
        }

    def tick(self) -> None:
        """Process one time step"""
        if not self.enabled:
            return

        self._current_tick += 1
        self._total_steps += 1

        # Decay exploration rate
        if self.exploration_strategy == ExplorationStrategy.DECAYING_EPSILON:
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )

        # Decay eligibility traces
        for key in list(self._eligibility_traces.keys()):
            self._eligibility_traces[key] *= self.eligibility_trace_decay * self.discount_factor
            if self._eligibility_traces[key] < 0.01:
                del self._eligibility_traces[key]

        # Periodic experience replay
        if self.use_experience_replay and len(self._replay_buffer) >= self.batch_size:
            if self._current_tick % 10 == 0:  # Replay every 10 ticks
                self._experience_replay()

    def get_state_key(self, state: Dict[str, Any], discretize: bool = True) -> str:
        """
        Convert a state dictionary to a hashable state key.

        Args:
            state: Dictionary describing the current state
            discretize: Whether to discretize continuous values

        Returns:
            String key representing the state
        """
        processed_state = {}

        for key, value in state.items():
            if discretize and isinstance(value, float):
                # Discretize continuous values into bins
                processed_state[key] = self._discretize(value)
            else:
                processed_state[key] = value

        # Create consistent hash
        state_str = json.dumps(processed_state, sort_keys=True, default=str)
        return hashlib.md5(state_str.encode()).hexdigest()[:16]

    def _discretize(self, value: float, num_bins: int = 10) -> int:
        """Discretize a continuous value into bins"""
        # Assume values are typically 0-1 range, but handle others
        if value <= 0:
            return 0
        elif value >= 1:
            return num_bins - 1
        else:
            return int(value * num_bins)

    def get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for a state-action pair"""
        if state not in self._q_table:
            return 0.0
        return self._q_table[state].get(action, 0.0)

    def set_q_value(self, state: str, action: str, value: float) -> None:
        """Set Q-value for a state-action pair"""
        if state not in self._q_table:
            self._q_table[state] = {}
        self._q_table[state][action] = value

    def select_action(
        self,
        state: str,
        available_actions: List[str],
        greedy: bool = False,
    ) -> str:
        """
        Select an action based on current policy.

        Args:
            state: Current state key
            available_actions: List of possible actions
            greedy: If True, always select best action (no exploration)

        Returns:
            Selected action
        """
        if not available_actions:
            raise ValueError("No available actions")

        if len(available_actions) == 1:
            return available_actions[0]

        if greedy:
            return self._get_best_action(state, available_actions)

        # Apply exploration strategy
        if self.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy(state, available_actions)
        elif self.exploration_strategy == ExplorationStrategy.DECAYING_EPSILON:
            return self._epsilon_greedy(state, available_actions)
        elif self.exploration_strategy == ExplorationStrategy.SOFTMAX:
            return self._softmax_selection(state, available_actions)
        elif self.exploration_strategy == ExplorationStrategy.UCB:
            return self._ucb_selection(state, available_actions)
        else:
            return self._epsilon_greedy(state, available_actions)

    def _get_best_action(self, state: str, available_actions: List[str]) -> str:
        """Get action with highest Q-value"""
        best_action = available_actions[0]
        best_value = self.get_q_value(state, best_action)

        for action in available_actions[1:]:
            value = self.get_q_value(state, action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _epsilon_greedy(self, state: str, available_actions: List[str]) -> str:
        """Epsilon-greedy action selection"""
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        return self._get_best_action(state, available_actions)

    def _softmax_selection(self, state: str, available_actions: List[str]) -> str:
        """Softmax (Boltzmann) action selection"""
        import math

        temperature = max(0.1, self.exploration_rate * 10)  # Use exploration as temperature

        q_values = [self.get_q_value(state, a) for a in available_actions]
        max_q = max(q_values)

        # Compute softmax probabilities (with overflow protection)
        exp_values = []
        for q in q_values:
            exp_val = math.exp((q - max_q) / temperature)
            exp_values.append(exp_val)

        total = sum(exp_values)
        probabilities = [e / total for e in exp_values]

        # Sample action based on probabilities
        r = random.random()
        cumulative = 0
        for action, prob in zip(available_actions, probabilities):
            cumulative += prob
            if r <= cumulative:
                return action

        return available_actions[-1]

    def _ucb_selection(self, state: str, available_actions: List[str]) -> str:
        """Upper Confidence Bound action selection"""
        import math

        c = 2.0  # Exploration constant

        # Ensure all actions have been tried at least once
        if state not in self._action_counts:
            self._action_counts[state] = {}

        for action in available_actions:
            if action not in self._action_counts[state]:
                self._action_counts[state][action] = 0
                return action

        # Calculate UCB values
        best_action = available_actions[0]
        best_ucb = float('-inf')

        for action in available_actions:
            q_value = self.get_q_value(state, action)
            count = self._action_counts[state][action]

            if count == 0:
                return action

            # Ensure total_steps is at least 1 to avoid log(0)
            total = max(1, self._total_steps)
            ucb = q_value + c * math.sqrt(math.log(total) / count)

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action

    def learn_from_experience(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        done: bool = False,
        available_actions: Optional[List[str]] = None,
    ) -> float:
        """
        Update Q-values based on experience (Q-learning update).

        Q(s,a) <- Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current state key
            action: Action taken
            reward: Reward received
            next_state: Resulting state key
            done: Whether episode ended
            available_actions: Actions available in next state (for max Q calculation)

        Returns:
            TD error (delta)
        """
        # Update action counts
        if state not in self._action_counts:
            self._action_counts[state] = {}
        self._action_counts[state][action] = self._action_counts[state].get(action, 0) + 1

        # Get current Q-value
        current_q = self.get_q_value(state, action)

        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            # Get max Q-value for next state
            if available_actions:
                max_next_q = max(self.get_q_value(next_state, a) for a in available_actions)
            elif next_state in self._q_table:
                max_next_q = max(self._q_table[next_state].values()) if self._q_table[next_state] else 0.0
            else:
                max_next_q = 0.0

            target_q = reward + self.discount_factor * max_next_q

        # Calculate TD error
        td_error = target_q - current_q

        # Update Q-value
        new_q = current_q + self.learning_rate * td_error
        self.set_q_value(state, action, new_q)

        # Update eligibility traces and apply TD(λ)
        self._update_eligibility_traces(state, action, td_error)

        # Store experience in replay buffer
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            tick=self._current_tick,
        )
        self._replay_buffer.append(experience)

        # Update statistics
        self._stats["total_experiences"] += 1
        self._stats["total_updates"] += 1
        self._stats["cumulative_reward"] += reward
        self._stats["average_reward"] = (
            self._stats["cumulative_reward"] / self._stats["total_experiences"]
        )

        # Emit event
        bus = get_event_bus()
        bus.emit("learning.updated", {
            "organism_id": self.owner.id if self.owner else "unknown",
            "state": state,
            "action": action,
            "reward": reward,
            "td_error": td_error,
        }, source="reinforcement_learner")

        return td_error

    def _update_eligibility_traces(self, state: str, action: str, td_error: float) -> None:
        """Update eligibility traces for TD(λ)"""
        # Set trace for current state-action
        self._eligibility_traces[(state, action)] = 1.0

        # Update all Q-values based on traces
        for (s, a), trace in self._eligibility_traces.items():
            if trace > 0.01:
                current_q = self.get_q_value(s, a)
                update = self.learning_rate * td_error * trace
                self.set_q_value(s, a, current_q + update)

    def _experience_replay(self) -> None:
        """Sample and learn from past experiences"""
        if len(self._replay_buffer) < self.batch_size:
            return

        # Sample random batch
        batch = random.sample(list(self._replay_buffer), self.batch_size)

        for exp in batch:
            self.learn_from_experience(
                state=exp.state,
                action=exp.action,
                reward=exp.reward,
                next_state=exp.next_state,
                done=exp.done,
            )

        self._stats["total_replays"] += 1

    def get_policy(self, state: str) -> Dict[str, float]:
        """
        Get the current policy (action probabilities) for a state.

        Returns:
            Dictionary mapping actions to probabilities
        """
        if state not in self._q_table or not self._q_table[state]:
            return {}

        actions = list(self._q_table[state].keys())
        q_values = list(self._q_table[state].values())

        # Softmax to get probabilities
        import math
        max_q = max(q_values)
        exp_values = [math.exp(q - max_q) for q in q_values]
        total = sum(exp_values)

        return {
            action: exp_val / total
            for action, exp_val in zip(actions, exp_values)
        }

    def reset_learning(self) -> None:
        """Reset all learned values (keep hyperparameters)"""
        self._q_table.clear()
        self._action_counts.clear()
        self._eligibility_traces.clear()
        self._replay_buffer.clear()
        self.exploration_rate = self.initial_exploration_rate
        self._total_steps = 0
        self._stats = {
            "total_experiences": 0,
            "total_updates": 0,
            "total_replays": 0,
            "average_reward": 0.0,
            "cumulative_reward": 0.0,
        }

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Get current learner state"""
        return {
            "current_tick": self._current_tick,
            "total_steps": self._total_steps,
            "q_table": self._q_table,
            "action_counts": self._action_counts,
            "eligibility_traces": {
                f"{s}:{a}": v for (s, a), v in self._eligibility_traces.items()
            },
            "replay_buffer": [exp.to_dict() for exp in self._replay_buffer],
            "exploration_rate": self.exploration_rate,
            "stats": self._stats.copy(),
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "initial_exploration_rate": self.initial_exploration_rate,
                "exploration_strategy": self.exploration_strategy.name,
                "exploration_decay": self.exploration_decay,
                "min_exploration_rate": self.min_exploration_rate,
                "replay_buffer_size": self.replay_buffer_size,
                "batch_size": self.batch_size,
                "use_experience_replay": self.use_experience_replay,
                "eligibility_trace_decay": self.eligibility_trace_decay,
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore learner state"""
        self._current_tick = state.get("current_tick", 0)
        self._total_steps = state.get("total_steps", 0)
        self._q_table = state.get("q_table", {})
        self._action_counts = state.get("action_counts", {})

        # Restore eligibility traces
        self._eligibility_traces = {}
        for key, value in state.get("eligibility_traces", {}).items():
            parts = key.split(":")
            if len(parts) == 2:
                self._eligibility_traces[(parts[0], parts[1])] = value

        # Restore replay buffer
        self._replay_buffer = deque(maxlen=self.replay_buffer_size)
        for exp_data in state.get("replay_buffer", []):
            self._replay_buffer.append(Experience.from_dict(exp_data))

        self.exploration_rate = state.get("exploration_rate", self.initial_exploration_rate)
        self._stats = state.get("stats", self._stats)

        # Restore hyperparameters
        hp = state.get("hyperparameters", {})
        self.learning_rate = hp.get("learning_rate", self.learning_rate)
        self.discount_factor = hp.get("discount_factor", self.discount_factor)
        self.initial_exploration_rate = hp.get("initial_exploration_rate", self.initial_exploration_rate)
        self.exploration_decay = hp.get("exploration_decay", self.exploration_decay)
        self.min_exploration_rate = hp.get("min_exploration_rate", self.min_exploration_rate)
        self.batch_size = hp.get("batch_size", self.batch_size)
        self.use_experience_replay = hp.get("use_experience_replay", self.use_experience_replay)
        self.eligibility_trace_decay = hp.get("eligibility_trace_decay", self.eligibility_trace_decay)

        if "exploration_strategy" in hp:
            try:
                self.exploration_strategy = ExplorationStrategy[hp["exploration_strategy"]]
            except KeyError:
                pass

    def reset(self) -> None:
        """Reset to initial state"""
        super().reset()
        self.reset_learning()
        self._current_tick = 0
