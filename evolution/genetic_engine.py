"""
Genetic Evolution Engine

A genetic algorithm system for evolving organism traits.

Features:
- Genome representation with dominant/recessive alleles
- Multiple crossover mechanisms (single-point, two-point, uniform)
- Selection strategies (tournament, roulette wheel, rank-based)
- Speciation tracking for genetic diversity
- Fitness function composition
- Integration with NPCPU event system

Example:
    from evolution.genetic_engine import Genome, EvolutionEngine, SelectionStrategy

    # Create initial population
    engine = EvolutionEngine(
        population_size=100,
        mutation_rate=0.1,
        selection_strategy=SelectionStrategy.TOURNAMENT
    )

    # Define fitness function
    def fitness_fn(organism, environment):
        speed = organism.genome.express("speed")
        strength = organism.genome.express("strength")
        return speed * 0.6 + strength * 0.4

    # Evolve population
    engine.evolve(generations=100, fitness_function=fitness_fn)
"""

from typing import Dict, Any, List, Optional, Callable, Tuple, Set, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import math
import copy
import uuid
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.events import get_event_bus
from core.abstractions import BaseOrganism

from .mutations import (
    MutationEngine,
    MutationRecord,
    MutationType,
    GaussianMutator,
    AdaptiveMutator,
    BaseMutator
)


class SelectionStrategy(Enum):
    """Selection strategies for choosing parents"""
    TOURNAMENT = auto()      # Tournament selection
    ROULETTE = auto()        # Fitness-proportionate selection
    RANK = auto()            # Rank-based selection
    TRUNCATION = auto()      # Select top N individuals
    BOLTZMANN = auto()       # Temperature-based selection


class CrossoverMethod(Enum):
    """Methods for combining parent genomes"""
    SINGLE_POINT = auto()    # Single crossover point
    TWO_POINT = auto()       # Two crossover points
    UNIFORM = auto()         # Random gene selection
    ARITHMETIC = auto()      # Weighted average of genes
    BLX_ALPHA = auto()       # Blend crossover


@dataclass
class Allele:
    """
    Represents a single allele (gene variant).

    Alleles can be dominant or recessive, affecting how traits
    are expressed when inherited from two parents.
    """
    value: float
    dominant: bool = True
    origin: str = ""  # Track which parent this came from

    def __repr__(self) -> str:
        dom = "D" if self.dominant else "r"
        return f"Allele({self.value:.3f}, {dom})"


@dataclass
class Gene:
    """
    Represents a gene with two alleles (diploid).

    Expression follows Mendelian inheritance:
    - If both alleles are equal dominance, average
    - If one is dominant, express the dominant allele
    """
    name: str
    allele1: Allele
    allele2: Allele
    min_value: float = 0.0
    max_value: float = 1.0

    def express(self) -> float:
        """
        Express the phenotype value based on allele dominance.

        Returns:
            Expressed trait value
        """
        if self.allele1.dominant and not self.allele2.dominant:
            return self.allele1.value
        elif self.allele2.dominant and not self.allele1.dominant:
            return self.allele2.value
        else:
            # Co-dominance: blend values
            return (self.allele1.value + self.allele2.value) / 2

    def get_random_allele(self) -> Allele:
        """Get a random allele from this gene (for reproduction)"""
        return copy.deepcopy(random.choice([self.allele1, self.allele2]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "allele1": {"value": self.allele1.value, "dominant": self.allele1.dominant},
            "allele2": {"value": self.allele2.value, "dominant": self.allele2.dominant},
            "expressed": self.express()
        }


class Genome:
    """
    Genetic representation of organism traits.

    Features:
    - Gene encoding for traits
    - Dominant/recessive alleles
    - Mutation operators
    - Crossover mechanisms

    Example:
        genome = Genome()
        genome.add_gene("speed", 0.5, dominant=True)
        genome.add_gene("strength", 0.7, dominant=False)

        # Express traits
        speed = genome.express("speed")

        # Create offspring
        child_genome = Genome.crossover(parent1.genome, parent2.genome)
    """

    def __init__(self, genes: Optional[Dict[str, Gene]] = None):
        self._id = str(uuid.uuid4())[:8]
        self._genes: Dict[str, Gene] = genes or {}
        self._generation = 0
        self._parent_ids: List[str] = []
        self._mutation_count = 0
        self._species_id: Optional[str] = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def generation(self) -> int:
        return self._generation

    @generation.setter
    def generation(self, value: int) -> None:
        self._generation = value

    @property
    def genes(self) -> Dict[str, Gene]:
        return self._genes.copy()

    @property
    def gene_names(self) -> List[str]:
        return list(self._genes.keys())

    @property
    def species_id(self) -> Optional[str]:
        return self._species_id

    @species_id.setter
    def species_id(self, value: str) -> None:
        self._species_id = value

    def add_gene(
        self,
        name: str,
        value: float,
        dominant: bool = True,
        value2: Optional[float] = None,
        dominant2: Optional[bool] = None,
        min_value: float = 0.0,
        max_value: float = 1.0
    ) -> None:
        """
        Add a gene to the genome.

        Args:
            name: Gene identifier
            value: Value for first allele
            dominant: Whether first allele is dominant
            value2: Value for second allele (defaults to value)
            dominant2: Whether second allele is dominant (defaults to not dominant)
            min_value: Minimum expressible value
            max_value: Maximum expressible value
        """
        allele1 = Allele(value=value, dominant=dominant, origin=self._id)
        allele2 = Allele(
            value=value2 if value2 is not None else value,
            dominant=dominant2 if dominant2 is not None else not dominant,
            origin=self._id
        )

        self._genes[name] = Gene(
            name=name,
            allele1=allele1,
            allele2=allele2,
            min_value=min_value,
            max_value=max_value
        )

    def express(self, gene_name: str) -> float:
        """
        Get the expressed value of a gene.

        Args:
            gene_name: Name of the gene to express

        Returns:
            Expressed phenotype value

        Raises:
            KeyError: If gene doesn't exist
        """
        if gene_name not in self._genes:
            raise KeyError(f"Gene '{gene_name}' not found in genome")
        return self._genes[gene_name].express()

    def express_all(self) -> Dict[str, float]:
        """Get expressed values for all genes"""
        return {name: gene.express() for name, gene in self._genes.items()}

    def get_raw_values(self) -> Dict[str, Tuple[float, float]]:
        """Get raw allele values for all genes"""
        return {
            name: (gene.allele1.value, gene.allele2.value)
            for name, gene in self._genes.items()
        }

    def distance_to(self, other: 'Genome') -> float:
        """
        Calculate genetic distance to another genome.

        Uses Euclidean distance of expressed values.
        """
        shared_genes = set(self._genes.keys()) & set(other._genes.keys())
        if not shared_genes:
            return float('inf')

        sum_sq = 0.0
        for gene_name in shared_genes:
            diff = self.express(gene_name) - other.express(gene_name)
            sum_sq += diff ** 2

        return math.sqrt(sum_sq / len(shared_genes))

    @classmethod
    def crossover(
        cls,
        parent1: 'Genome',
        parent2: 'Genome',
        method: CrossoverMethod = CrossoverMethod.UNIFORM
    ) -> 'Genome':
        """
        Create offspring genome by combining two parent genomes.

        Args:
            parent1: First parent genome
            parent2: Second parent genome
            method: Crossover method to use

        Returns:
            New child genome
        """
        child = cls()
        child._generation = max(parent1._generation, parent2._generation) + 1
        child._parent_ids = [parent1._id, parent2._id]

        # Get shared genes
        all_genes = set(parent1._genes.keys()) | set(parent2._genes.keys())

        if method == CrossoverMethod.UNIFORM:
            child._genes = cls._uniform_crossover(parent1, parent2, all_genes)
        elif method == CrossoverMethod.SINGLE_POINT:
            child._genes = cls._single_point_crossover(parent1, parent2, all_genes)
        elif method == CrossoverMethod.TWO_POINT:
            child._genes = cls._two_point_crossover(parent1, parent2, all_genes)
        elif method == CrossoverMethod.ARITHMETIC:
            child._genes = cls._arithmetic_crossover(parent1, parent2, all_genes)
        elif method == CrossoverMethod.BLX_ALPHA:
            child._genes = cls._blx_crossover(parent1, parent2, all_genes)
        else:
            child._genes = cls._uniform_crossover(parent1, parent2, all_genes)

        # Emit crossover event
        bus = get_event_bus()
        bus.emit("evolution.crossover", {
            "parent1_id": parent1._id,
            "parent2_id": parent2._id,
            "child_id": child._id,
            "method": method.name
        })

        return child

    @staticmethod
    def _uniform_crossover(
        parent1: 'Genome',
        parent2: 'Genome',
        gene_names: Set[str]
    ) -> Dict[str, Gene]:
        """Uniform crossover: randomly select alleles from each parent"""
        genes = {}

        for gene_name in gene_names:
            g1 = parent1._genes.get(gene_name)
            g2 = parent2._genes.get(gene_name)

            if g1 and g2:
                # Get one allele from each parent
                allele1 = g1.get_random_allele()
                allele2 = g2.get_random_allele()
                genes[gene_name] = Gene(
                    name=gene_name,
                    allele1=allele1,
                    allele2=allele2,
                    min_value=g1.min_value,
                    max_value=g1.max_value
                )
            elif g1:
                genes[gene_name] = copy.deepcopy(g1)
            elif g2:
                genes[gene_name] = copy.deepcopy(g2)

        return genes

    @staticmethod
    def _single_point_crossover(
        parent1: 'Genome',
        parent2: 'Genome',
        gene_names: Set[str]
    ) -> Dict[str, Gene]:
        """Single-point crossover: split genes at one point"""
        genes = {}
        gene_list = sorted(gene_names)
        crossover_point = random.randint(1, len(gene_list) - 1) if len(gene_list) > 1 else 0

        for i, gene_name in enumerate(gene_list):
            source = parent1 if i < crossover_point else parent2
            fallback = parent2 if i < crossover_point else parent1

            g = source._genes.get(gene_name) or fallback._genes.get(gene_name)
            if g:
                genes[gene_name] = copy.deepcopy(g)

        return genes

    @staticmethod
    def _two_point_crossover(
        parent1: 'Genome',
        parent2: 'Genome',
        gene_names: Set[str]
    ) -> Dict[str, Gene]:
        """Two-point crossover: swap genes between two points"""
        genes = {}
        gene_list = sorted(gene_names)

        if len(gene_list) > 2:
            point1 = random.randint(0, len(gene_list) - 2)
            point2 = random.randint(point1 + 1, len(gene_list) - 1)
        else:
            point1, point2 = 0, len(gene_list)

        for i, gene_name in enumerate(gene_list):
            if point1 <= i < point2:
                source = parent2
                fallback = parent1
            else:
                source = parent1
                fallback = parent2

            g = source._genes.get(gene_name) or fallback._genes.get(gene_name)
            if g:
                genes[gene_name] = copy.deepcopy(g)

        return genes

    @staticmethod
    def _arithmetic_crossover(
        parent1: 'Genome',
        parent2: 'Genome',
        gene_names: Set[str]
    ) -> Dict[str, Gene]:
        """Arithmetic crossover: weighted average of gene values"""
        genes = {}
        alpha = random.random()

        for gene_name in gene_names:
            g1 = parent1._genes.get(gene_name)
            g2 = parent2._genes.get(gene_name)

            if g1 and g2:
                val1 = alpha * g1.allele1.value + (1 - alpha) * g2.allele1.value
                val2 = alpha * g1.allele2.value + (1 - alpha) * g2.allele2.value

                genes[gene_name] = Gene(
                    name=gene_name,
                    allele1=Allele(val1, g1.allele1.dominant),
                    allele2=Allele(val2, g2.allele2.dominant),
                    min_value=g1.min_value,
                    max_value=g1.max_value
                )
            elif g1:
                genes[gene_name] = copy.deepcopy(g1)
            elif g2:
                genes[gene_name] = copy.deepcopy(g2)

        return genes

    @staticmethod
    def _blx_crossover(
        parent1: 'Genome',
        parent2: 'Genome',
        gene_names: Set[str],
        alpha: float = 0.5
    ) -> Dict[str, Gene]:
        """BLX-alpha crossover: sample from extended range"""
        genes = {}

        for gene_name in gene_names:
            g1 = parent1._genes.get(gene_name)
            g2 = parent2._genes.get(gene_name)

            if g1 and g2:
                # Calculate extended range
                min_val = min(g1.express(), g2.express())
                max_val = max(g1.express(), g2.express())
                range_val = max_val - min_val

                # Extend range by alpha
                lower = max(0.0, min_val - alpha * range_val)
                upper = min(1.0, max_val + alpha * range_val)

                # Sample child values
                val1 = random.uniform(lower, upper)
                val2 = random.uniform(lower, upper)

                genes[gene_name] = Gene(
                    name=gene_name,
                    allele1=Allele(val1, g1.allele1.dominant),
                    allele2=Allele(val2, g2.allele2.dominant),
                    min_value=g1.min_value,
                    max_value=g1.max_value
                )
            elif g1:
                genes[gene_name] = copy.deepcopy(g1)
            elif g2:
                genes[gene_name] = copy.deepcopy(g2)

        return genes

    def mutate(
        self,
        mutation_engine: MutationEngine,
        generation: int = 0
    ) -> List[MutationRecord]:
        """
        Apply mutations to this genome.

        Args:
            mutation_engine: Engine to use for mutations
            generation: Current generation number

        Returns:
            List of mutation records
        """
        all_records = []

        for gene_name, gene in self._genes.items():
            # Mutate allele 1
            new_val1, record1 = mutation_engine.mutate_gene(
                gene_name + "_a1",
                gene.allele1.value,
                generation
            )
            if record1:
                gene.allele1.value = max(gene.min_value, min(gene.max_value, new_val1))
                record1.gene_name = gene_name
                all_records.append(record1)
                self._mutation_count += 1

            # Mutate allele 2
            new_val2, record2 = mutation_engine.mutate_gene(
                gene_name + "_a2",
                gene.allele2.value,
                generation
            )
            if record2:
                gene.allele2.value = max(gene.min_value, min(gene.max_value, new_val2))
                record2.gene_name = gene_name
                all_records.append(record2)
                self._mutation_count += 1

        return all_records

    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary"""
        return {
            "id": self._id,
            "generation": self._generation,
            "parent_ids": self._parent_ids,
            "mutation_count": self._mutation_count,
            "species_id": self._species_id,
            "genes": {name: gene.to_dict() for name, gene in self._genes.items()},
            "expressed": self.express_all()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genome':
        """Deserialize genome from dictionary"""
        genome = cls()
        genome._id = data.get("id", genome._id)
        genome._generation = data.get("generation", 0)
        genome._parent_ids = data.get("parent_ids", [])
        genome._mutation_count = data.get("mutation_count", 0)
        genome._species_id = data.get("species_id")

        for gene_data in data.get("genes", {}).values():
            genome.add_gene(
                name=gene_data["name"],
                value=gene_data["allele1"]["value"],
                dominant=gene_data["allele1"]["dominant"],
                value2=gene_data["allele2"]["value"],
                dominant2=gene_data["allele2"]["dominant"]
            )

        return genome

    @classmethod
    def random(cls, gene_specs: Dict[str, Tuple[float, float]]) -> 'Genome':
        """
        Create a random genome from gene specifications.

        Args:
            gene_specs: Dict mapping gene names to (min, max) value ranges

        Example:
            genome = Genome.random({
                "speed": (0.0, 1.0),
                "strength": (0.0, 1.0),
                "intelligence": (0.2, 0.8)
            })
        """
        genome = cls()

        for gene_name, (min_val, max_val) in gene_specs.items():
            value1 = random.uniform(min_val, max_val)
            value2 = random.uniform(min_val, max_val)
            dominant = random.choice([True, False])

            genome.add_gene(
                name=gene_name,
                value=value1,
                dominant=dominant,
                value2=value2,
                dominant2=not dominant,
                min_value=min_val,
                max_value=max_val
            )

        return genome


@dataclass
class Species:
    """
    Represents a species (group of genetically similar organisms).

    Used for speciation tracking and maintaining genetic diversity.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    representative: Optional[Genome] = None
    members: List[str] = field(default_factory=list)  # Genome IDs
    created_generation: int = 0
    fitness_history: List[float] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    def add_member(self, genome_id: str) -> None:
        if genome_id not in self.members:
            self.members.append(genome_id)

    def remove_member(self, genome_id: str) -> None:
        if genome_id in self.members:
            self.members.remove(genome_id)


@dataclass
class Individual:
    """Wraps a genome with fitness information for evolution"""
    genome: Genome
    fitness: float = 0.0
    adjusted_fitness: float = 0.0  # For speciation-based fitness sharing
    rank: int = 0

    @property
    def id(self) -> str:
        return self.genome.id


FitnessFunction = Callable[[BaseOrganism, Any], float]


class EvolutionEngine:
    """
    Manages population-level evolution.

    Features:
    - Fitness function composition
    - Selection strategies (tournament, roulette)
    - Speciation tracking
    - Genetic diversity metrics

    Example:
        engine = EvolutionEngine(
            population_size=100,
            mutation_rate=0.1,
            crossover_rate=0.8
        )

        # Initialize population
        gene_specs = {"speed": (0, 1), "strength": (0, 1)}
        engine.initialize_population(gene_specs)

        # Define fitness
        def fitness(organism, env):
            return organism.genome.express("speed") * 0.5 + organism.genome.express("strength") * 0.5

        # Run evolution
        for gen in range(100):
            engine.evolve_generation(fitness)
            print(f"Gen {gen}: Best fitness = {engine.best_fitness:.4f}")
    """

    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_count: int = 2,
        selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT,
        crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM,
        tournament_size: int = 3,
        speciation_threshold: float = 0.3,
        enable_speciation: bool = True,
        mutator: Optional[BaseMutator] = None
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.selection_strategy = selection_strategy
        self.crossover_method = crossover_method
        self.tournament_size = tournament_size
        self.speciation_threshold = speciation_threshold
        self.enable_speciation = enable_speciation

        # Internal state
        self.population: List[Individual] = []
        self.species: Dict[str, Species] = {}
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []

        # Mutation engine
        self.mutation_engine = MutationEngine(
            base_rate=mutation_rate,
            mutator=mutator or GaussianMutator(sigma=0.1)
        )

    @property
    def best_fitness(self) -> float:
        """Get the best fitness in current population"""
        if not self.population:
            return 0.0
        return max(ind.fitness for ind in self.population)

    @property
    def average_fitness(self) -> float:
        """Get average fitness of current population"""
        if not self.population:
            return 0.0
        return sum(ind.fitness for ind in self.population) / len(self.population)

    def initialize_population(
        self,
        gene_specs: Dict[str, Tuple[float, float]],
        initial_genomes: Optional[List[Genome]] = None
    ) -> None:
        """
        Initialize the population with random or provided genomes.

        Args:
            gene_specs: Dict mapping gene names to (min, max) value ranges
            initial_genomes: Optional list of initial genomes to include
        """
        self.population = []
        self.generation = 0

        # Add initial genomes if provided
        if initial_genomes:
            for genome in initial_genomes[:self.population_size]:
                self.population.append(Individual(genome=genome))

        # Fill remaining with random genomes
        while len(self.population) < self.population_size:
            genome = Genome.random(gene_specs)
            self.population.append(Individual(genome=genome))

        # Initialize speciation
        if self.enable_speciation:
            self._update_speciation()

        # Emit event
        bus = get_event_bus()
        bus.emit("evolution.population_initialized", {
            "size": len(self.population),
            "gene_count": len(gene_specs)
        })

    def calculate_fitness(
        self,
        fitness_function: Callable[[Genome], float]
    ) -> None:
        """
        Calculate fitness for all individuals in the population.

        Args:
            fitness_function: Function that takes a Genome and returns fitness score
        """
        for individual in self.population:
            individual.fitness = fitness_function(individual.genome)

            # Update best ever
            if self.best_ever is None or individual.fitness > self.best_ever.fitness:
                self.best_ever = copy.deepcopy(individual)

        # Apply fitness sharing if speciation is enabled
        if self.enable_speciation:
            self._apply_fitness_sharing()

        # Update rankings
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for rank, ind in enumerate(sorted_pop):
            ind.rank = rank

    def _apply_fitness_sharing(self) -> None:
        """Apply fitness sharing within species to maintain diversity"""
        for species in self.species.values():
            species_members = [
                ind for ind in self.population
                if ind.genome.species_id == species.id
            ]

            if not species_members:
                continue

            species_size = len(species_members)
            for ind in species_members:
                ind.adjusted_fitness = ind.fitness / species_size

    def select_parents(self, count: int = 2) -> List[Individual]:
        """
        Select parents for reproduction based on selection strategy.

        Args:
            count: Number of parents to select

        Returns:
            List of selected individuals
        """
        if self.selection_strategy == SelectionStrategy.TOURNAMENT:
            return [self._tournament_select() for _ in range(count)]
        elif self.selection_strategy == SelectionStrategy.ROULETTE:
            return [self._roulette_select() for _ in range(count)]
        elif self.selection_strategy == SelectionStrategy.RANK:
            return [self._rank_select() for _ in range(count)]
        elif self.selection_strategy == SelectionStrategy.TRUNCATION:
            return self._truncation_select(count)
        elif self.selection_strategy == SelectionStrategy.BOLTZMANN:
            return [self._boltzmann_select() for _ in range(count)]
        else:
            return [self._tournament_select() for _ in range(count)]

    def _tournament_select(self) -> Individual:
        """Tournament selection: best of random subset"""
        tournament = random.sample(
            self.population,
            min(self.tournament_size, len(self.population))
        )
        return max(tournament, key=lambda x: x.fitness)

    def _roulette_select(self) -> Individual:
        """Roulette wheel selection: probability proportional to fitness"""
        # Shift fitness to be positive
        min_fitness = min(ind.fitness for ind in self.population)
        adjusted = [(ind, ind.fitness - min_fitness + 0.001) for ind in self.population]
        total = sum(f for _, f in adjusted)

        pick = random.uniform(0, total)
        current = 0.0

        for ind, fitness in adjusted:
            current += fitness
            if current >= pick:
                return ind

        return self.population[-1]

    def _rank_select(self) -> Individual:
        """Rank-based selection: probability based on rank, not absolute fitness"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness)
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)

        pick = random.uniform(0, total_rank)
        current = 0.0

        for ind, rank in zip(sorted_pop, ranks):
            current += rank
            if current >= pick:
                return ind

        return sorted_pop[-1]

    def _truncation_select(self, count: int) -> List[Individual]:
        """Truncation selection: select from top N%"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        top_n = max(count, len(sorted_pop) // 4)  # At least top 25%
        return random.sample(sorted_pop[:top_n], count)

    def _boltzmann_select(self, temperature: float = 1.0) -> Individual:
        """Boltzmann selection: temperature-controlled probability"""
        fitnesses = [ind.fitness for ind in self.population]
        max_fit = max(fitnesses)

        # Calculate Boltzmann probabilities
        probs = []
        for f in fitnesses:
            prob = math.exp((f - max_fit) / temperature)
            probs.append(prob)

        total = sum(probs)
        probs = [p / total for p in probs]

        return random.choices(self.population, weights=probs, k=1)[0]

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """
        Perform crossover between two parent genomes.

        Args:
            parent1: First parent genome
            parent2: Second parent genome

        Returns:
            Child genome
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copy of random parent
            return copy.deepcopy(random.choice([parent1, parent2]))

        return Genome.crossover(parent1, parent2, method=self.crossover_method)

    def mutate(self, genome: Genome) -> List[MutationRecord]:
        """
        Apply mutations to a genome.

        Args:
            genome: Genome to mutate

        Returns:
            List of mutation records
        """
        return genome.mutate(self.mutation_engine, self.generation)

    def evolve_generation(
        self,
        fitness_function: Callable[[Genome], float]
    ) -> Dict[str, Any]:
        """
        Evolve the population by one generation.

        Args:
            fitness_function: Function to evaluate genome fitness

        Returns:
            Dictionary with generation statistics
        """
        # Evaluate fitness
        self.calculate_fitness(fitness_function)

        # Record statistics
        best = max(self.population, key=lambda x: x.fitness)
        avg = self.average_fitness
        diversity = self.calculate_diversity()

        self.fitness_history.append(best.fitness)
        self.diversity_history.append(diversity)

        # Create next generation
        next_population = []

        # Elitism: keep best individuals
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for i in range(min(self.elitism_count, len(sorted_pop))):
            elite = copy.deepcopy(sorted_pop[i])
            elite.genome.generation = self.generation + 1
            next_population.append(elite)

        # Create offspring
        while len(next_population) < self.population_size:
            parents = self.select_parents(2)
            child_genome = self.crossover(parents[0].genome, parents[1].genome)
            self.mutate(child_genome)
            child_genome.generation = self.generation + 1
            next_population.append(Individual(genome=child_genome))

        self.population = next_population
        self.generation += 1

        # Update speciation
        if self.enable_speciation:
            self._update_speciation()

        # Emit event
        bus = get_event_bus()
        bus.emit("evolution.generation_complete", {
            "generation": self.generation,
            "best_fitness": best.fitness,
            "avg_fitness": avg,
            "diversity": diversity,
            "species_count": len(self.species)
        })

        return {
            "generation": self.generation,
            "best_fitness": best.fitness,
            "average_fitness": avg,
            "diversity": diversity,
            "best_genome": best.genome.to_dict()
        }

    def calculate_diversity(self) -> float:
        """
        Calculate genetic diversity of the population.

        Returns:
            Diversity score (0 = no diversity, higher = more diverse)
        """
        if len(self.population) < 2:
            return 0.0

        distances = []
        sample_size = min(50, len(self.population))  # Sample for efficiency
        sample = random.sample(self.population, sample_size)

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                dist = sample[i].genome.distance_to(sample[j].genome)
                distances.append(dist)

        return sum(distances) / len(distances) if distances else 0.0

    def _update_speciation(self) -> None:
        """Update species assignments based on genetic similarity"""
        # Clear current species memberships
        for species in self.species.values():
            species.members.clear()

        # Assign individuals to species
        for individual in self.population:
            assigned = False

            for species in self.species.values():
                if species.representative:
                    distance = individual.genome.distance_to(species.representative)
                    if distance < self.speciation_threshold:
                        species.add_member(individual.genome.id)
                        individual.genome.species_id = species.id
                        assigned = True
                        break

            # Create new species if no match
            if not assigned:
                new_species = Species(
                    name=f"Species_{len(self.species) + 1}",
                    representative=copy.deepcopy(individual.genome),
                    created_generation=self.generation
                )
                new_species.add_member(individual.genome.id)
                individual.genome.species_id = new_species.id
                self.species[new_species.id] = new_species

        # Remove empty species
        empty_species = [
            sid for sid, s in self.species.items()
            if s.size == 0
        ]
        for sid in empty_species:
            del self.species[sid]

        # Update representatives
        for species in self.species.values():
            if species.members:
                # Set representative to a random member
                member_id = random.choice(species.members)
                for ind in self.population:
                    if ind.genome.id == member_id:
                        species.representative = copy.deepcopy(ind.genome)
                        break

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics"""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_fitness,
            "average_fitness": self.average_fitness,
            "best_ever_fitness": self.best_ever.fitness if self.best_ever else 0,
            "diversity": self.calculate_diversity(),
            "species_count": len(self.species),
            "mutation_stats": self.mutation_engine.get_mutation_stats(),
            "fitness_history": self.fitness_history[-10:],
            "diversity_history": self.diversity_history[-10:]
        }

    def get_best_genome(self) -> Optional[Genome]:
        """Get the best genome from current population"""
        if not self.population:
            return None
        best = max(self.population, key=lambda x: x.fitness)
        return best.genome

    def get_species_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for each species"""
        stats = []
        for species in self.species.values():
            members = [
                ind for ind in self.population
                if ind.genome.species_id == species.id
            ]
            if members:
                stats.append({
                    "id": species.id,
                    "name": species.name,
                    "size": species.size,
                    "avg_fitness": sum(m.fitness for m in members) / len(members),
                    "best_fitness": max(m.fitness for m in members),
                    "created_generation": species.created_generation
                })
        return stats


# Convenience functions for common fitness calculations

def compose_fitness(
    *fitness_fns: Tuple[Callable[[Genome], float], float]
) -> Callable[[Genome], float]:
    """
    Compose multiple fitness functions with weights.

    Example:
        fitness = compose_fitness(
            (speed_fitness, 0.4),
            (strength_fitness, 0.4),
            (endurance_fitness, 0.2)
        )
    """
    def composed(genome: Genome) -> float:
        total = 0.0
        total_weight = sum(w for _, w in fitness_fns)
        for fn, weight in fitness_fns:
            total += fn(genome) * weight
        return total / total_weight if total_weight > 0 else 0.0

    return composed


def trait_fitness(trait_name: str, target: float = 1.0) -> Callable[[Genome], float]:
    """
    Create a fitness function that maximizes a single trait.

    Example:
        speed_fitness = trait_fitness("speed", target=1.0)
    """
    def fitness(genome: Genome) -> float:
        try:
            value = genome.express(trait_name)
            return 1.0 - abs(target - value)
        except KeyError:
            return 0.0

    return fitness


def balanced_fitness(gene_names: List[str]) -> Callable[[Genome], float]:
    """
    Create fitness function that rewards balanced traits.

    Example:
        fitness = balanced_fitness(["speed", "strength", "intelligence"])
    """
    def fitness(genome: Genome) -> float:
        values = []
        for name in gene_names:
            try:
                values.append(genome.express(name))
            except KeyError:
                pass

        if not values:
            return 0.0

        # Reward high average with low variance
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)

        return avg * (1.0 - variance)

    return fitness
