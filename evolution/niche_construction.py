"""
Niche Construction System

Implements organism-environment feedback mechanisms:
- Environmental modification by organisms
- Ecological inheritance
- Ecosystem engineering
- Feedback loops between evolution and environment
- Niche creation and destruction

Example:
    from evolution.niche_construction import NicheConstructionEngine

    engine = NicheConstructionEngine()

    # Define construction behavior
    engine.add_construction_behavior(
        behavior_gene="dam_building",
        environmental_effect="water_level",
        effect_magnitude=0.3
    )

    # Process construction effects
    engine.process_construction(population, environment, generation)
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import random
import math
import copy
import uuid

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.events import get_event_bus
from .genetic_engine import Genome


class ConstructionType(Enum):
    """Types of niche construction"""
    PERTURBATION = auto()      # Active environmental change
    RELOCATION = auto()        # Moving to different environment
    COUNTERACTIVE = auto()     # Counteracting environmental change
    INCEPTIVE = auto()         # Initiating new environmental state


class EffectPersistence(Enum):
    """How long construction effects last"""
    TEMPORARY = auto()     # One generation
    MEDIUM = auto()        # Several generations
    PERMANENT = auto()     # Until explicitly reversed


@dataclass
class EnvironmentalVariable:
    """
    Represents an environmental variable that can be modified.
    """
    name: str
    value: float = 0.5          # Current value (0-1)
    baseline: float = 0.5       # Natural equilibrium value
    recovery_rate: float = 0.1  # Rate of return to baseline
    min_value: float = 0.0
    max_value: float = 1.0

    def modify(self, delta: float) -> float:
        """Modify value and return new value"""
        self.value = max(self.min_value, min(self.max_value, self.value + delta))
        return self.value

    def recover(self) -> float:
        """Move toward baseline"""
        diff = self.baseline - self.value
        self.value += diff * self.recovery_rate
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "baseline": self.baseline,
            "deviation": self.value - self.baseline
        }


@dataclass
class ConstructionBehavior:
    """
    Defines a niche construction behavior.

    Links a genetic trait to an environmental effect.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    behavior_gene: str = ""            # Gene controlling behavior
    environmental_variable: str = ""    # Variable being modified
    effect_magnitude: float = 0.1       # How much behavior changes environment
    construction_type: ConstructionType = ConstructionType.PERTURBATION
    persistence: EffectPersistence = EffectPersistence.MEDIUM
    threshold: float = 0.5              # Minimum gene expression to trigger
    cost: float = 0.05                  # Fitness cost of behavior
    collective: bool = False            # Requires multiple organisms

    def calculate_effect(
        self,
        genome: Genome,
        population_contribution: float = 0.0
    ) -> float:
        """Calculate environmental effect from this behavior"""
        if self.behavior_gene not in genome.genes:
            return 0.0

        expression = genome.express(self.behavior_gene)
        if expression < self.threshold:
            return 0.0

        effect = expression * self.effect_magnitude

        if self.collective:
            effect *= population_contribution

        return effect

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "gene": self.behavior_gene,
            "variable": self.environmental_variable,
            "magnitude": self.effect_magnitude,
            "type": self.construction_type.name,
            "persistence": self.persistence.name
        }


@dataclass
class ConstructedNiche:
    """
    A niche created by organisms.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    creator_species: str = ""
    creation_generation: int = 0
    environmental_state: Dict[str, float] = field(default_factory=dict)
    beneficiary_species: List[str] = field(default_factory=list)
    maintenance_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "creator": self.creator_species,
            "created": self.creation_generation,
            "state": self.environmental_state,
            "beneficiaries": self.beneficiary_species
        }


@dataclass
class SelectionFeedback:
    """
    Feedback from environmental change to selection.
    """
    environmental_variable: str
    affected_gene: str
    feedback_strength: float = 0.5  # How much env change affects fitness
    direction: float = 1.0           # +1 = higher env helps higher gene

    def calculate_fitness_modifier(
        self,
        env_value: float,
        gene_expression: float
    ) -> float:
        """Calculate fitness modifier from environment-gene interaction"""
        # If environment is high and gene is high, boost fitness (if direction is +1)
        alignment = env_value * gene_expression if self.direction > 0 else env_value * (1 - gene_expression)
        return 1.0 + (alignment - 0.5) * self.feedback_strength


class Environment:
    """
    Simple environment model for niche construction.
    """

    def __init__(
        self,
        variables: Optional[List[EnvironmentalVariable]] = None
    ):
        self.variables: Dict[str, EnvironmentalVariable] = {}
        if variables:
            for var in variables:
                self.variables[var.name] = var

    def add_variable(
        self,
        name: str,
        initial_value: float = 0.5,
        baseline: float = 0.5,
        recovery_rate: float = 0.1
    ) -> EnvironmentalVariable:
        """Add an environmental variable"""
        var = EnvironmentalVariable(
            name=name,
            value=initial_value,
            baseline=baseline,
            recovery_rate=recovery_rate
        )
        self.variables[name] = var
        return var

    def get_value(self, name: str) -> float:
        """Get current value of a variable"""
        if name in self.variables:
            return self.variables[name].value
        return 0.5

    def modify(self, name: str, delta: float) -> float:
        """Modify a variable"""
        if name in self.variables:
            return self.variables[name].modify(delta)
        return 0.5

    def recover(self) -> None:
        """All variables move toward baseline"""
        for var in self.variables.values():
            var.recover()

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: var.to_dict()
            for name, var in self.variables.items()
        }


class NicheConstructionEngine:
    """
    Engine for managing niche construction dynamics.

    Features:
    - Environmental modification by organisms
    - Feedback from environment to selection
    - Ecological inheritance
    - Niche tracking
    - Multi-species interactions

    Example:
        engine = NicheConstructionEngine()

        # Create environment
        env = engine.create_environment([
            ("soil_quality", 0.5),
            ("water_level", 0.5)
        ])

        # Add construction behavior
        engine.add_construction_behavior(
            name="burrowing",
            behavior_gene="digging_ability",
            environmental_variable="soil_quality",
            effect_magnitude=0.2
        )

        # Process each generation
        results = engine.process_generation(population, generation)
    """

    def __init__(
        self,
        environment: Optional[Environment] = None,
        enable_feedback: bool = True,
        collective_threshold: float = 0.1  # Fraction of pop for collective effect
    ):
        self.environment = environment or Environment()
        self.enable_feedback = enable_feedback
        self.collective_threshold = collective_threshold

        # Construction behaviors
        self.behaviors: List[ConstructionBehavior] = []

        # Selection feedback rules
        self.feedbacks: List[SelectionFeedback] = []

        # Constructed niches
        self.niches: Dict[str, ConstructedNiche] = {}

        # History
        self.modification_history: List[Dict[str, Any]] = []

        # Statistics
        self.total_modifications = 0

    def create_environment(
        self,
        variables: List[Tuple[str, float]]
    ) -> Environment:
        """Create environment with specified variables"""
        self.environment = Environment()
        for name, value in variables:
            self.environment.add_variable(name, value, value)
        return self.environment

    def add_construction_behavior(
        self,
        name: str,
        behavior_gene: str,
        environmental_variable: str,
        effect_magnitude: float = 0.1,
        construction_type: ConstructionType = ConstructionType.PERTURBATION,
        threshold: float = 0.5,
        cost: float = 0.05,
        collective: bool = False
    ) -> ConstructionBehavior:
        """Add a niche construction behavior"""
        # Ensure environment variable exists
        if environmental_variable not in self.environment.variables:
            self.environment.add_variable(environmental_variable)

        behavior = ConstructionBehavior(
            name=name,
            behavior_gene=behavior_gene,
            environmental_variable=environmental_variable,
            effect_magnitude=effect_magnitude,
            construction_type=construction_type,
            threshold=threshold,
            cost=cost,
            collective=collective
        )

        self.behaviors.append(behavior)
        return behavior

    def add_selection_feedback(
        self,
        environmental_variable: str,
        affected_gene: str,
        feedback_strength: float = 0.5,
        positive: bool = True
    ) -> SelectionFeedback:
        """Add feedback from environment to selection"""
        feedback = SelectionFeedback(
            environmental_variable=environmental_variable,
            affected_gene=affected_gene,
            feedback_strength=feedback_strength,
            direction=1.0 if positive else -1.0
        )

        self.feedbacks.append(feedback)
        return feedback

    def calculate_construction_effects(
        self,
        population: List[Genome]
    ) -> Dict[str, float]:
        """
        Calculate total environmental effects from population.

        Returns:
            Dict mapping environmental variable to total effect
        """
        effects: Dict[str, float] = {}

        # Calculate population contribution for collective behaviors
        pop_size = len(population)

        for behavior in self.behaviors:
            total_effect = 0.0
            contributors = 0

            for genome in population:
                if behavior.behavior_gene not in genome.genes:
                    continue

                expression = genome.express(behavior.behavior_gene)
                if expression >= behavior.threshold:
                    contributors += 1

                    if behavior.collective:
                        # Effect scales with population fraction
                        pop_contribution = contributors / pop_size
                        effect = behavior.calculate_effect(genome, pop_contribution)
                    else:
                        effect = behavior.calculate_effect(genome)

                    total_effect += effect

            if total_effect != 0:
                var_name = behavior.environmental_variable
                effects[var_name] = effects.get(var_name, 0) + total_effect

        return effects

    def apply_construction_effects(
        self,
        effects: Dict[str, float],
        generation: int
    ) -> Dict[str, Any]:
        """
        Apply construction effects to environment.

        Returns:
            Summary of modifications
        """
        modifications = []

        for var_name, effect in effects.items():
            if var_name not in self.environment.variables:
                continue

            old_value = self.environment.get_value(var_name)
            new_value = self.environment.modify(var_name, effect)

            if abs(new_value - old_value) > 0.01:
                mod = {
                    "variable": var_name,
                    "old_value": old_value,
                    "new_value": new_value,
                    "change": effect,
                    "generation": generation
                }
                modifications.append(mod)
                self.total_modifications += 1

        if modifications:
            self.modification_history.append({
                "generation": generation,
                "modifications": modifications
            })

            # Emit event
            bus = get_event_bus()
            bus.emit("niche.construction", {
                "generation": generation,
                "modifications": len(modifications)
            })

        return {
            "modifications": modifications,
            "count": len(modifications)
        }

    def calculate_fitness_modifiers(
        self,
        genome: Genome
    ) -> Dict[str, float]:
        """
        Calculate fitness modifiers from environmental feedback.

        Returns:
            Dict with modifier details
        """
        modifiers = {}
        total_modifier = 1.0

        for feedback in self.feedbacks:
            if feedback.affected_gene not in genome.genes:
                continue

            env_value = self.environment.get_value(feedback.environmental_variable)
            gene_expression = genome.express(feedback.affected_gene)

            modifier = feedback.calculate_fitness_modifier(env_value, gene_expression)
            modifiers[feedback.affected_gene] = modifier
            total_modifier *= modifier

        modifiers["total"] = total_modifier
        return modifiers

    def calculate_construction_costs(
        self,
        population: List[Genome]
    ) -> Dict[str, float]:
        """
        Calculate fitness costs of construction behaviors.

        Returns:
            Dict mapping genome ID to total cost
        """
        costs = {}

        for genome in population:
            total_cost = 0.0

            for behavior in self.behaviors:
                if behavior.behavior_gene not in genome.genes:
                    continue

                expression = genome.express(behavior.behavior_gene)
                if expression >= behavior.threshold:
                    total_cost += behavior.cost * expression

            costs[genome.id] = total_cost

        return costs

    def detect_niche_creation(
        self,
        generation: int,
        threshold: float = 0.3
    ) -> Optional[ConstructedNiche]:
        """
        Detect if a new stable niche has been created.

        Returns:
            ConstructedNiche if created, None otherwise
        """
        # Check if any variable has deviated significantly and stabilized
        for var_name, var in self.environment.variables.items():
            deviation = abs(var.value - var.baseline)

            if deviation > threshold:
                # Check if stable (not recovering quickly)
                if deviation > threshold * 0.8:  # Still significantly modified
                    niche = ConstructedNiche(
                        name=f"Modified_{var_name}_{generation}",
                        creation_generation=generation,
                        environmental_state={var_name: var.value}
                    )
                    self.niches[niche.id] = niche

                    # Emit event
                    bus = get_event_bus()
                    bus.emit("niche.created", {
                        "niche_id": niche.id,
                        "generation": generation,
                        "variable": var_name
                    })

                    return niche

        return None

    def process_generation(
        self,
        population: List[Genome],
        generation: int
    ) -> Dict[str, Any]:
        """
        Process niche construction for a generation.

        Args:
            population: Current population
            generation: Current generation

        Returns:
            Summary of niche construction activity
        """
        # Calculate construction effects
        effects = self.calculate_construction_effects(population)

        # Apply effects
        modifications = self.apply_construction_effects(effects, generation)

        # Environmental recovery (natural baseline restoration)
        self.environment.recover()

        # Check for niche creation
        new_niche = self.detect_niche_creation(generation)

        # Calculate fitness effects if feedback enabled
        fitness_modifiers = {}
        if self.enable_feedback:
            for genome in population[:10]:  # Sample
                mods = self.calculate_fitness_modifiers(genome)
                fitness_modifiers[genome.id] = mods.get("total", 1.0)

        return {
            "generation": generation,
            "effects_calculated": len(effects),
            "modifications_applied": modifications["count"],
            "environment": self.environment.to_dict(),
            "new_niche": new_niche.to_dict() if new_niche else None,
            "sample_fitness_modifiers": fitness_modifiers
        }

    def get_ecological_inheritance(
        self,
        offspring_generation: int
    ) -> Dict[str, float]:
        """
        Get environmental state inherited by offspring.

        This represents ecological inheritance - the modified
        environment passed to the next generation.
        """
        return {
            var_name: var.value
            for var_name, var in self.environment.variables.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get niche construction statistics"""
        return {
            "total_modifications": self.total_modifications,
            "behaviors": [b.to_dict() for b in self.behaviors],
            "feedbacks": len(self.feedbacks),
            "constructed_niches": len(self.niches),
            "current_environment": self.environment.to_dict(),
            "recent_modifications": self.modification_history[-5:]
        }


# Convenience functions

def create_beaver_model() -> NicheConstructionEngine:
    """Create engine modeling beaver-like ecosystem engineering"""
    engine = NicheConstructionEngine()

    # Environment
    engine.create_environment([
        ("water_level", 0.3),
        ("wetland_area", 0.2),
        ("fish_habitat", 0.3)
    ])

    # Dam building behavior
    engine.add_construction_behavior(
        name="dam_building",
        behavior_gene="engineering_ability",
        environmental_variable="water_level",
        effect_magnitude=0.15,
        collective=True,
        cost=0.1
    )

    # Wetland creation
    engine.add_construction_behavior(
        name="wetland_creation",
        behavior_gene="engineering_ability",
        environmental_variable="wetland_area",
        effect_magnitude=0.1,
        cost=0.05
    )

    # Feedback: wetland benefits aquatic foraging
    engine.add_selection_feedback(
        environmental_variable="wetland_area",
        affected_gene="aquatic_foraging",
        feedback_strength=0.4,
        positive=True
    )

    return engine
