"""
Phylogenetic Visualization for NPCPU

Provides tools for visualizing evolutionary relationships, genetic
diversity, and trait evolution over time.

Features:
- Phylogenetic tree construction and visualization
- Genetic diversity heatmaps
- Trait evolution tracking
- Species radiation diagrams
- Fitness landscape visualization
- ASCII art for terminal display

Usage:
    from evolution.phylogenetic_viz import PhylogeneticVisualizer

    viz = PhylogeneticVisualizer(evolution_engine)

    # ASCII phylogenetic tree
    tree = viz.generate_ascii_tree()
    print(tree)

    # Diversity report
    report = viz.generate_diversity_report()
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math
import random

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evolution.genetic_engine import (
    Genome, Individual, Species, EvolutionEngine
)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PhyloNode:
    """A node in the phylogenetic tree"""
    id: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    genome_id: Optional[str] = None
    generation: int = 0
    fitness: float = 0.0
    traits: Dict[str, float] = field(default_factory=dict)
    species_id: Optional[str] = None
    branch_length: float = 1.0


@dataclass
class PhyloTree:
    """Complete phylogenetic tree structure"""
    nodes: Dict[str, PhyloNode] = field(default_factory=dict)
    root_id: Optional[str] = None
    generation_layers: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        if not self.generation_layers:
            return 0
        return max(self.generation_layers.keys()) + 1

    @property
    def width(self) -> int:
        if not self.generation_layers:
            return 0
        return max(len(nodes) for nodes in self.generation_layers.values())


@dataclass
class DiversityMetrics:
    """Metrics measuring genetic diversity"""
    total_unique_genomes: int = 0
    species_count: int = 0
    avg_genetic_distance: float = 0.0
    trait_variance: Dict[str, float] = field(default_factory=dict)
    shannon_diversity: float = 0.0
    simpson_diversity: float = 0.0


# ============================================================================
# Phylogenetic Tree Builder
# ============================================================================

class PhylogeneticTreeBuilder:
    """
    Constructs phylogenetic trees from evolution data.

    Builds trees based on:
    - Parent-child relationships in genome lineage
    - Genetic distance for clustering
    - Species boundaries
    """

    def __init__(self):
        self._nodes: Dict[str, PhyloNode] = {}
        self._edges: List[Tuple[str, str]] = []

    def build_from_engine(self, engine: EvolutionEngine) -> PhyloTree:
        """
        Build phylogenetic tree from evolution engine state.

        Args:
            engine: Evolution engine with population and history

        Returns:
            PhyloTree structure
        """
        tree = PhyloTree()
        self._nodes.clear()

        # Create nodes for all individuals
        for individual in engine.population:
            genome = individual.genome
            node = PhyloNode(
                id=genome.id,
                genome_id=genome.id,
                generation=genome.generation,
                fitness=individual.fitness,
                traits=genome.express_all(),
                species_id=genome.species_id
            )

            # Set parent if available
            if genome._parent_ids:
                node.parent_id = genome._parent_ids[0]

            self._nodes[node.id] = node
            tree.nodes[node.id] = node

            # Add to generation layer
            if node.generation not in tree.generation_layers:
                tree.generation_layers[node.generation] = []
            tree.generation_layers[node.generation].append(node.id)

        # Build parent-child relationships
        for node_id, node in tree.nodes.items():
            if node.parent_id and node.parent_id in tree.nodes:
                tree.nodes[node.parent_id].children.append(node_id)

        # Find root nodes (no parent or parent not in tree)
        roots = [
            node_id for node_id, node in tree.nodes.items()
            if node.parent_id is None or node.parent_id not in tree.nodes
        ]

        if roots:
            # Create virtual root if multiple roots
            if len(roots) > 1:
                virtual_root = PhyloNode(id="root", generation=-1)
                virtual_root.children = roots
                tree.nodes["root"] = virtual_root
                tree.root_id = "root"
                tree.generation_layers[-1] = ["root"]
            else:
                tree.root_id = roots[0]

        return tree

    def build_from_genomes(
        self,
        genomes: List[Genome],
        method: str = "distance"
    ) -> PhyloTree:
        """
        Build phylogenetic tree from a list of genomes.

        Args:
            genomes: List of genomes to cluster
            method: "distance" for distance-based, "lineage" for parent-based

        Returns:
            PhyloTree structure
        """
        if method == "lineage":
            return self._build_lineage_tree(genomes)
        else:
            return self._build_distance_tree(genomes)

    def _build_lineage_tree(self, genomes: List[Genome]) -> PhyloTree:
        """Build tree based on parent-child lineage"""
        tree = PhyloTree()

        for genome in genomes:
            node = PhyloNode(
                id=genome.id,
                genome_id=genome.id,
                generation=genome.generation,
                traits=genome.express_all(),
                species_id=genome.species_id
            )

            if genome._parent_ids:
                node.parent_id = genome._parent_ids[0]

            tree.nodes[node.id] = node

            if node.generation not in tree.generation_layers:
                tree.generation_layers[node.generation] = []
            tree.generation_layers[node.generation].append(node.id)

        # Build relationships
        for node_id, node in tree.nodes.items():
            if node.parent_id and node.parent_id in tree.nodes:
                tree.nodes[node.parent_id].children.append(node_id)

        # Find root
        roots = [n for n in tree.nodes.values() if n.parent_id is None]
        if roots:
            tree.root_id = roots[0].id

        return tree

    def _build_distance_tree(self, genomes: List[Genome]) -> PhyloTree:
        """Build tree using UPGMA-like clustering based on genetic distance"""
        if not genomes:
            return PhyloTree()

        tree = PhyloTree()

        # Create leaf nodes
        clusters: Dict[str, List[str]] = {}
        for genome in genomes:
            node = PhyloNode(
                id=genome.id,
                genome_id=genome.id,
                generation=genome.generation,
                traits=genome.express_all()
            )
            tree.nodes[node.id] = node
            clusters[node.id] = [node.id]

        # UPGMA-style clustering
        node_counter = 0
        while len(clusters) > 1:
            # Find closest pair
            min_dist = float('inf')
            closest_pair = None

            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    dist = self._cluster_distance(
                        clusters[cluster_ids[i]],
                        clusters[cluster_ids[j]],
                        genomes
                    )
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (cluster_ids[i], cluster_ids[j])

            if closest_pair is None:
                break

            # Merge clusters
            c1, c2 = closest_pair
            new_id = f"internal_{node_counter}"
            node_counter += 1

            # Create internal node
            internal_node = PhyloNode(
                id=new_id,
                children=[c1, c2],
                branch_length=min_dist / 2
            )
            tree.nodes[new_id] = internal_node

            # Update parent references
            tree.nodes[c1].parent_id = new_id
            tree.nodes[c2].parent_id = new_id

            # Update clusters
            clusters[new_id] = clusters[c1] + clusters[c2]
            del clusters[c1]
            del clusters[c2]

        # Set root
        if clusters:
            tree.root_id = list(clusters.keys())[0]

        return tree

    def _cluster_distance(
        self,
        cluster1: List[str],
        cluster2: List[str],
        genomes: List[Genome]
    ) -> float:
        """Calculate average distance between two clusters"""
        genome_map = {g.id: g for g in genomes}

        distances = []
        for id1 in cluster1:
            for id2 in cluster2:
                if id1 in genome_map and id2 in genome_map:
                    dist = genome_map[id1].distance_to(genome_map[id2])
                    if dist != float('inf'):
                        distances.append(dist)

        return sum(distances) / len(distances) if distances else float('inf')


# ============================================================================
# Visualization
# ============================================================================

class PhylogeneticVisualizer:
    """
    Generates visualizations of evolutionary data.

    Provides multiple output formats:
    - ASCII art for terminal display
    - Text reports
    - Data structures for plotting libraries
    """

    def __init__(self, engine: Optional[EvolutionEngine] = None):
        self.engine = engine
        self.tree_builder = PhylogeneticTreeBuilder()
        self._tree: Optional[PhyloTree] = None

    def set_engine(self, engine: EvolutionEngine) -> None:
        """Set the evolution engine to visualize"""
        self.engine = engine
        self._tree = None

    def build_tree(self) -> PhyloTree:
        """Build phylogenetic tree from current engine state"""
        if self.engine is None:
            return PhyloTree()

        self._tree = self.tree_builder.build_from_engine(self.engine)
        return self._tree

    def generate_ascii_tree(self, max_depth: int = 10, max_width: int = 80) -> str:
        """
        Generate ASCII representation of phylogenetic tree.

        Args:
            max_depth: Maximum tree depth to display
            max_width: Maximum character width

        Returns:
            ASCII art string
        """
        if self._tree is None:
            self.build_tree()

        if not self._tree or not self._tree.root_id:
            return "No phylogenetic tree available"

        lines = []
        lines.append("=" * max_width)
        lines.append("PHYLOGENETIC TREE".center(max_width))
        lines.append("=" * max_width)
        lines.append("")

        # Build ASCII representation
        self._render_node_ascii(
            self._tree.root_id,
            lines,
            prefix="",
            is_last=True,
            depth=0,
            max_depth=max_depth
        )

        lines.append("")
        lines.append("-" * max_width)
        lines.append(f"Total nodes: {len(self._tree.nodes)}")
        lines.append(f"Tree depth: {self._tree.depth}")
        lines.append("=" * max_width)

        return "\n".join(lines)

    def _render_node_ascii(
        self,
        node_id: str,
        lines: List[str],
        prefix: str,
        is_last: bool,
        depth: int,
        max_depth: int
    ) -> None:
        """Recursively render node as ASCII"""
        if depth > max_depth:
            return

        node = self._tree.nodes.get(node_id)
        if not node:
            return

        # Build connector
        connector = "└── " if is_last else "├── "

        # Build node label
        if node.species_id:
            label = f"[{node.species_id[:6]}] {node_id[:8]}"
        else:
            label = node_id[:12]

        if node.fitness > 0:
            label += f" (fit={node.fitness:.2f})"

        lines.append(f"{prefix}{connector}{label}")

        # Render children
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child_id in enumerate(node.children):
            self._render_node_ascii(
                child_id,
                lines,
                child_prefix,
                is_last=(i == len(node.children) - 1),
                depth=depth + 1,
                max_depth=max_depth
            )

    def generate_diversity_report(self) -> str:
        """
        Generate a text report on genetic diversity.

        Returns:
            Formatted diversity report string
        """
        if self.engine is None:
            return "No evolution engine configured"

        lines = []
        lines.append("=" * 60)
        lines.append("GENETIC DIVERSITY REPORT".center(60))
        lines.append("=" * 60)
        lines.append("")

        # Basic stats
        lines.append("POPULATION STATISTICS")
        lines.append("-" * 40)
        lines.append(f"  Population size: {len(self.engine.population)}")
        lines.append(f"  Generation: {self.engine.generation}")
        lines.append(f"  Best fitness: {self.engine.best_fitness:.4f}")
        lines.append(f"  Average fitness: {self.engine.average_fitness:.4f}")
        lines.append("")

        # Diversity metrics
        diversity = self.engine.calculate_diversity()
        lines.append("DIVERSITY METRICS")
        lines.append("-" * 40)
        lines.append(f"  Genetic diversity: {diversity:.4f}")
        lines.append(f"  Species count: {len(self.engine.species)}")
        lines.append("")

        # Species breakdown
        if self.engine.species:
            lines.append("SPECIES DISTRIBUTION")
            lines.append("-" * 40)
            species_stats = self.engine.get_species_stats()
            for sp in sorted(species_stats, key=lambda x: x['size'], reverse=True)[:10]:
                bar_len = int(sp['size'] / len(self.engine.population) * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                lines.append(f"  {sp['name'][:12]:<12} [{bar}] {sp['size']:3d}")
            lines.append("")

        # Trait distribution
        if self.engine.population:
            lines.append("TRAIT DISTRIBUTIONS")
            lines.append("-" * 40)
            trait_stats = self._calculate_trait_stats()
            for trait, stats in list(trait_stats.items())[:10]:
                lines.append(f"  {trait}:")
                lines.append(f"    Mean: {stats['mean']:.3f}  Std: {stats['std']:.3f}")
                lines.append(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                # ASCII histogram
                hist = self._ascii_histogram(stats['values'], width=30)
                lines.append(f"    {hist}")
            lines.append("")

        # Fitness history
        if self.engine.fitness_history:
            lines.append("FITNESS EVOLUTION")
            lines.append("-" * 40)
            lines.append(self._ascii_fitness_chart(width=50, height=10))
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _calculate_trait_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each trait"""
        trait_values: Dict[str, List[float]] = defaultdict(list)

        for individual in self.engine.population:
            for trait, value in individual.genome.express_all().items():
                trait_values[trait].append(value)

        stats = {}
        for trait, values in trait_values.items():
            if values:
                import statistics
                stats[trait] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "values": values
                }

        return stats

    def _ascii_histogram(self, values: List[float], width: int = 30, bins: int = 10) -> str:
        """Create ASCII histogram"""
        if not values:
            return "No data"

        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return "█" * width

        bin_width = (max_val - min_val) / bins
        counts = [0] * bins

        for v in values:
            bin_idx = min(int((v - min_val) / bin_width), bins - 1)
            counts[bin_idx] += 1

        max_count = max(counts) if counts else 1
        chars = " ▁▂▃▄▅▆▇█"

        result = ""
        for count in counts:
            char_idx = int(count / max_count * (len(chars) - 1))
            result += chars[char_idx] * (width // bins)

        return result

    def _ascii_fitness_chart(self, width: int = 50, height: int = 10) -> str:
        """Create ASCII fitness evolution chart"""
        history = self.engine.fitness_history
        if not history:
            return "No fitness history"

        # Sample history if too long
        if len(history) > width:
            step = len(history) / width
            sampled = [history[int(i * step)] for i in range(width)]
        else:
            sampled = history

        min_fit = min(sampled)
        max_fit = max(sampled)
        range_fit = max_fit - min_fit if max_fit > min_fit else 1

        lines = []
        for row in range(height, 0, -1):
            threshold = min_fit + (row / height) * range_fit
            line = ""
            for val in sampled:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            label = f"{threshold:.2f}" if row in [1, height] else ""
            lines.append(f"{label:>6} |{line}|")

        # X-axis
        lines.append(" " * 7 + "+" + "-" * len(sampled) + "+")
        lines.append(" " * 7 + f"Gen 0{' ' * (len(sampled) - 10)}Gen {len(history)}")

        return "\n".join(lines)

    def generate_species_tree(self) -> str:
        """Generate species-level phylogenetic tree"""
        if self.engine is None or not self.engine.species:
            return "No species data available"

        lines = []
        lines.append("=" * 50)
        lines.append("SPECIES TREE".center(50))
        lines.append("=" * 50)
        lines.append("")

        sorted_species = sorted(
            self.engine.species.values(),
            key=lambda s: s.created_generation
        )

        for i, species in enumerate(sorted_species):
            is_last = (i == len(sorted_species) - 1)
            connector = "└── " if is_last else "├── "

            size_bar = "█" * min(species.size, 20)
            lines.append(f"{connector}{species.name} (gen {species.created_generation})")
            lines.append(f"    Size: {species.size} {size_bar}")

            # Show members sample
            if species.members:
                sample = species.members[:3]
                lines.append(f"    Members: {', '.join(m[:8] for m in sample)}...")

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)

    def get_trait_evolution_data(self, trait_name: str) -> Dict[str, List[float]]:
        """
        Get data for plotting trait evolution over time.

        Returns dict with 'generations' and 'values' lists for plotting.
        """
        # This would require tracking trait history during evolution
        # For now, return current generation snapshot
        if self.engine is None:
            return {"generations": [], "values": []}

        values = []
        for individual in self.engine.population:
            try:
                value = individual.genome.express(trait_name)
                values.append(value)
            except KeyError:
                pass

        return {
            "generation": self.engine.generation,
            "values": values,
            "mean": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Phylogenetic Visualization Demo")
    print("=" * 60)

    # Create and evolve a population
    from evolution.genetic_engine import EvolutionEngine

    engine = EvolutionEngine(
        population_size=50,
        mutation_rate=0.1,
        enable_speciation=True
    )

    # Initialize with random genes
    gene_specs = {
        "speed": (0.0, 1.0),
        "strength": (0.0, 1.0),
        "intelligence": (0.0, 1.0),
        "endurance": (0.0, 1.0)
    }
    engine.initialize_population(gene_specs)

    # Simple fitness function
    def fitness(genome):
        s = genome.express("speed")
        st = genome.express("strength")
        i = genome.express("intelligence")
        return s * 0.4 + st * 0.3 + i * 0.3

    # Evolve for a few generations
    print("\nEvolving population...")
    for gen in range(20):
        engine.evolve_generation(fitness)
        if gen % 5 == 0:
            print(f"  Generation {gen}: Best={engine.best_fitness:.3f}, "
                  f"Species={len(engine.species)}")

    # Create visualizer
    viz = PhylogeneticVisualizer(engine)

    # Generate reports
    print("\n" + viz.generate_ascii_tree(max_depth=5))
    print("\n" + viz.generate_diversity_report())
    print("\n" + viz.generate_species_tree())
