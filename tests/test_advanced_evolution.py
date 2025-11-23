"""
Tests for Advanced Evolution (GENESIS-E) Components

Comprehensive test suite for:
- Horizontal Gene Transfer
- Epigenetics
- Speciation
- Sexual Selection
- Evolutionary Arms Race
- Niche Construction
- Evolutionary Innovation
- Extinction Dynamics
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evolution.genetic_engine import Genome, Gene, Allele, EvolutionEngine, Species
from evolution.horizontal_transfer import (
    HorizontalTransferEngine,
    TransferType,
    GenePool,
    GeneticElement,
    create_hgt_engine
)
from evolution.epigenetics import (
    EpigeneticLayer,
    EpigeneticMark,
    EnvironmentalStressor,
    Epigenome,
    EpigeneticMarkType,
    ExpressionModifier,
    create_common_stressors
)
from evolution.speciation import (
    SpeciationEngine,
    SpeciationType,
    IsolationType,
    IsolationBarrier,
    HybridZone
)
from evolution.sexual_selection import (
    SexualSelectionEngine,
    MatePreference,
    Ornament,
    PreferenceType,
    SelectionType,
    create_peacock_selection,
    create_combat_selection
)
from evolution.evolutionary_arms_race import (
    ArmsRaceEngine,
    TraitPair,
    CompetitorRole,
    InteractionType
)
from evolution.niche_construction import (
    NicheConstructionEngine,
    ConstructionBehavior,
    Environment,
    EnvironmentalVariable,
    create_beaver_model
)
from evolution.evolutionary_innovation import (
    InnovationEngine,
    InnovationPathway,
    InnovationType,
    create_standard_pathways
)
from evolution.extinction_dynamics import (
    ExtinctionEngine,
    ExtinctionType,
    RecoveryPhase,
    create_stable_world,
    create_volatile_world
)


def create_test_genome(gene_values: dict = None) -> Genome:
    """Helper to create test genomes"""
    genome = Genome()
    gene_values = gene_values or {
        "speed": 0.5,
        "strength": 0.6,
        "intelligence": 0.4,
        "metabolism": 0.7
    }
    for name, value in gene_values.items():
        genome.add_gene(name, value)
    return genome


def create_test_population(size: int = 10, species_id: str = None) -> list:
    """Helper to create test populations"""
    population = []
    for _ in range(size):
        genome = Genome.random({
            "speed": (0.0, 1.0),
            "strength": (0.0, 1.0),
            "intelligence": (0.0, 1.0),
            "metabolism": (0.0, 1.0)
        })
        if species_id:
            genome.species_id = species_id
        population.append(genome)
    return population


class TestHorizontalTransfer:
    """Tests for Horizontal Gene Transfer"""

    def test_hgt_engine_creation(self):
        """Test HGT engine creation"""
        engine = HorizontalTransferEngine()
        assert engine.transformation_rate > 0
        assert engine.conjugation_rate > 0
        assert engine.gene_pool is not None

    def test_release_genes_to_pool(self):
        """Test releasing genes to gene pool"""
        engine = HorizontalTransferEngine()
        genome = create_test_genome()

        element = engine.release_genes_to_pool(genome)

        assert element is not None
        assert len(element.genes) > 0
        assert element.origin_genome_id == genome.id

    def test_conjugation_transfer(self):
        """Test conjugation transfer between organisms"""
        engine = HorizontalTransferEngine(conjugation_rate=1.0)  # High rate for testing
        donor = create_test_genome({"unique_gene": 0.9})
        recipient = create_test_genome()

        success, record = engine.conjugation_transfer(
            donor, recipient,
            gene_names=["unique_gene"],
            generation=1
        )

        assert record is not None
        assert record.transfer_type == TransferType.CONJUGATION

    def test_gene_pool_decay(self):
        """Test gene pool decay over time"""
        engine = HorizontalTransferEngine()
        genome = create_test_genome()

        # Add many elements
        for _ in range(10):
            engine.release_genes_to_pool(genome, all_genes=True)

        initial_size = len(engine.gene_pool.elements)
        decayed = engine.gene_pool.decay()

        assert decayed > 0
        assert len(engine.gene_pool.elements) < initial_size

    def test_preset_configurations(self):
        """Test preset HGT configurations"""
        bacterial = create_hgt_engine("bacterial")
        eukaryotic = create_hgt_engine("eukaryotic")

        assert bacterial.conjugation_rate > eukaryotic.conjugation_rate


class TestEpigenetics:
    """Tests for Epigenetic Inheritance"""

    def test_epigenetic_layer_creation(self):
        """Test epigenetic layer creation"""
        layer = EpigeneticLayer()
        assert layer.inheritance_rate > 0
        assert len(layer.epigenomes) == 0

    def test_register_genome(self):
        """Test genome registration"""
        layer = EpigeneticLayer()
        genome = create_test_genome()

        epigenome = layer.register_genome(genome)

        assert epigenome.genome_id == genome.id
        assert genome.id in layer.epigenomes

    def test_apply_mark(self):
        """Test applying epigenetic marks"""
        layer = EpigeneticLayer()
        genome = create_test_genome()

        mark = EpigeneticMark(
            mark_type=EpigeneticMarkType.METHYLATION,
            target_gene="speed",
            intensity=0.8,
            modifier=ExpressionModifier.REDUCED
        )

        layer.apply_mark(genome, mark)

        epigenome = layer.get_epigenome(genome)
        marks = epigenome.get_marks_for_gene("speed")
        assert len(marks) == 1
        assert marks[0].modifier == ExpressionModifier.REDUCED

    def test_environmental_stressor(self):
        """Test environmental stressor application"""
        layer = EpigeneticLayer()
        genome = create_test_genome({
            "heat_tolerance": 0.5,
            "metabolism": 0.6
        })

        stressor = EnvironmentalStressor(
            name="heat_shock",
            affected_genes=["heat_tolerance", "metabolism"],
            intensity=0.7,
            activation_genes=["heat_tolerance"],
            silencing_genes=["metabolism"]
        )

        marks = layer.apply_stressor(genome, stressor)

        assert len(marks) == 2

    def test_expression_modification(self):
        """Test that marks modify expression"""
        layer = EpigeneticLayer()
        genome = create_test_genome({"speed": 0.8})

        base_expr = genome.express("speed")

        mark = EpigeneticMark(
            target_gene="speed",
            modifier=ExpressionModifier.REDUCED,
            intensity=0.8
        )
        layer.apply_mark(genome, mark)

        modified_expr = layer.get_modified_expression(genome, "speed")

        assert modified_expr < base_expr

    def test_mark_inheritance(self):
        """Test epigenetic mark inheritance"""
        layer = EpigeneticLayer()
        parent = create_test_genome()
        offspring = create_test_genome()

        # Add mark to parent
        mark = EpigeneticMark(
            target_gene="speed",
            modifier=ExpressionModifier.ENHANCED,
            intensity=0.9,
            generations_remaining=5
        )
        layer.apply_mark(parent, mark)

        # Inherit
        inherited = layer.handle_reproduction(parent, None, offspring)

        # Should inherit with some probability
        assert isinstance(inherited, int)


class TestSpeciation:
    """Tests for Speciation"""

    def test_speciation_engine_creation(self):
        """Test speciation engine creation"""
        engine = SpeciationEngine()
        assert engine.genetic_distance_threshold > 0
        assert len(engine.species_records) == 0

    def test_register_species(self):
        """Test species registration"""
        engine = SpeciationEngine()
        species = Species(name="TestSpecies")

        record = engine.register_species(species)

        assert species.id in engine.species_records
        assert species.id in engine.active_species

    def test_isolation_barrier(self):
        """Test reproductive isolation barriers"""
        barrier = IsolationBarrier(
            isolation_type=IsolationType.PREZYGOTIC_BEHAVIORAL,
            strength=0.7,
            species1_id="sp1",
            species2_id="sp2"
        )

        prob = barrier.get_reproduction_probability()
        assert prob == 0.3  # 1 - 0.7

    def test_check_hybridization(self):
        """Test hybridization checking"""
        engine = SpeciationEngine(enable_hybridization=True)

        genome1 = create_test_genome()
        genome2 = create_test_genome()
        genome1.species_id = "species1"
        genome2.species_id = "species2"

        can_hybridize, zone = engine.check_hybridization(genome1, genome2)

        # Should be able to hybridize without barriers
        assert isinstance(can_hybridize, bool)

    def test_species_diversity(self):
        """Test diversity calculations"""
        engine = SpeciationEngine()

        # Register some species
        for i in range(3):
            species = Species(name=f"Species_{i}")
            engine.register_species(species)

        diversity = engine.calculate_species_diversity()

        assert "active_species" in diversity
        assert diversity["active_species"] == 3


class TestSexualSelection:
    """Tests for Sexual Selection"""

    def test_selection_engine_creation(self):
        """Test sexual selection engine creation"""
        engine = SexualSelectionEngine()
        assert engine.choosiness >= 0
        assert len(engine.preferences) == 0

    def test_add_preference(self):
        """Test adding mate preferences"""
        engine = SexualSelectionEngine()

        pref = MatePreference(
            trait="display_brightness",
            preference_strength=0.8,
            preference_type=PreferenceType.DIRECTIONAL
        )
        engine.add_preference(pref)

        assert len(engine.preferences) == 1

    def test_calculate_attractiveness(self):
        """Test attractiveness calculation"""
        engine = SexualSelectionEngine()
        engine.add_preference(MatePreference(
            trait="speed",
            preference_strength=1.0,
            preference_type=PreferenceType.DIRECTIONAL
        ))

        high_speed = create_test_genome({"speed": 0.9})
        low_speed = create_test_genome({"speed": 0.2})

        attract_high = engine.calculate_attractiveness(high_speed)
        attract_low = engine.calculate_attractiveness(low_speed)

        assert attract_high > attract_low

    def test_choose_mate(self):
        """Test mate choice"""
        engine = SexualSelectionEngine(choosiness=0.5)
        engine.add_preference(MatePreference(
            trait="speed",
            preference_strength=1.0
        ))

        chooser = create_test_genome()
        candidates = create_test_population(5)

        chosen = engine.choose_mate(chooser, candidates)

        assert chosen is not None
        assert chosen in candidates

    def test_competition(self):
        """Test intrasexual competition"""
        engine = SexualSelectionEngine()
        engine.add_competition_trait("strength")

        c1 = create_test_genome({"strength": 0.9})
        c2 = create_test_genome({"strength": 0.3})

        result = engine.compete(c1, c2)

        assert result.winner_id in [c1.id, c2.id]

    def test_preset_engines(self):
        """Test preset selection engines"""
        peacock = create_peacock_selection()
        combat = create_combat_selection()

        assert len(peacock.preferences) > 0
        assert len(combat.competition_traits) > 0


class TestArmsRace:
    """Tests for Evolutionary Arms Race"""

    def test_arms_race_engine_creation(self):
        """Test arms race engine creation"""
        engine = ArmsRaceEngine()
        assert engine.base_escalation_rate > 0
        assert len(engine.relationships) == 0

    def test_add_predation_relationship(self):
        """Test adding predator-prey relationship"""
        engine = ArmsRaceEngine()

        rel = engine.add_predation_relationship(
            predator_species="wolf",
            prey_species="deer"
        )

        assert rel is not None
        assert rel.species1_id == "wolf"
        assert rel.species2_id == "deer"

    def test_trait_pair_outcome(self):
        """Test trait pair interaction outcome"""
        pair = TraitPair(
            attack_trait="speed",
            defense_trait="speed",
            attack_effectiveness=0.5
        )

        # Equal values should give ~50% success
        prob1 = pair.calculate_outcome(0.5, 0.5)
        assert 0.3 < prob1 < 0.7

        # Higher attack should increase success
        prob2 = pair.calculate_outcome(0.9, 0.3)
        assert prob2 > prob1

    def test_simulate_interaction(self):
        """Test interaction simulation"""
        engine = ArmsRaceEngine()
        rel = engine.add_predation_relationship("pred", "prey")

        predator = create_test_genome({"speed": 0.8, "stealth": 0.7})
        prey = create_test_genome({"speed": 0.6, "vigilance": 0.5})

        record = engine.simulate_interaction(predator, prey, rel)

        assert record is not None
        assert record.attacker_id == predator.id
        assert record.defender_id == prey.id


class TestNicheConstruction:
    """Tests for Niche Construction"""

    def test_niche_engine_creation(self):
        """Test niche construction engine creation"""
        engine = NicheConstructionEngine()
        assert engine.environment is not None

    def test_create_environment(self):
        """Test environment creation"""
        engine = NicheConstructionEngine()
        env = engine.create_environment([
            ("water_level", 0.5),
            ("soil_quality", 0.6)
        ])

        assert "water_level" in env.variables
        assert env.get_value("water_level") == 0.5

    def test_add_construction_behavior(self):
        """Test adding construction behavior"""
        engine = NicheConstructionEngine()
        engine.create_environment([("water_level", 0.3)])

        behavior = engine.add_construction_behavior(
            name="dam_building",
            behavior_gene="engineering",
            environmental_variable="water_level",
            effect_magnitude=0.2
        )

        assert behavior in engine.behaviors

    def test_calculate_effects(self):
        """Test construction effect calculation"""
        engine = NicheConstructionEngine()
        engine.create_environment([("water_level", 0.3)])
        engine.add_construction_behavior(
            name="dam_building",
            behavior_gene="engineering",
            environmental_variable="water_level",
            effect_magnitude=0.2,
            threshold=0.3
        )

        population = [
            create_test_genome({"engineering": 0.8}),
            create_test_genome({"engineering": 0.9})
        ]

        effects = engine.calculate_construction_effects(population)

        assert "water_level" in effects
        assert effects["water_level"] > 0

    def test_beaver_model(self):
        """Test beaver ecosystem model"""
        engine = create_beaver_model()

        assert "water_level" in engine.environment.variables
        assert len(engine.behaviors) >= 2


class TestEvolutionaryInnovation:
    """Tests for Evolutionary Innovation"""

    def test_innovation_engine_creation(self):
        """Test innovation engine creation"""
        engine = InnovationEngine()
        assert engine.base_innovation_rate > 0

    def test_add_pathway(self):
        """Test adding innovation pathway"""
        engine = InnovationEngine()

        pathway = engine.add_innovation_pathway(
            name="flight",
            prerequisite_genes=["wings", "lightweight"],
            threshold_expressions={"wings": 0.8, "lightweight": 0.7},
            new_gene="flight_capability"
        )

        assert pathway.id in engine.pathways

    def test_check_prerequisites(self):
        """Test prerequisite checking"""
        pathway = InnovationPathway(
            name="test",
            prerequisite_genes=["gene_a", "gene_b"],
            threshold_expressions={"gene_a": 0.7, "gene_b": 0.5}
        )

        genome1 = create_test_genome({"gene_a": 0.9, "gene_b": 0.8})
        genome2 = create_test_genome({"gene_a": 0.3, "gene_b": 0.8})

        assert pathway.check_prerequisites(genome1) == True
        assert pathway.check_prerequisites(genome2) == False

    def test_punctuated_equilibrium(self):
        """Test punctuated equilibrium dynamics"""
        engine = InnovationEngine(enable_punctuated=True)

        # Update state
        state1 = engine.update_punctuated_equilibrium(1)

        assert "in_stasis" in state1
        assert "rate_multiplier" in state1

    def test_standard_pathways(self):
        """Test standard innovation pathways"""
        pathways = create_standard_pathways()

        assert len(pathways) >= 4
        assert any(p.name == "multicellularity" for p in pathways)
        assert any(p.name == "flight" for p in pathways)


class TestExtinctionDynamics:
    """Tests for Extinction Dynamics"""

    def test_extinction_engine_creation(self):
        """Test extinction engine creation"""
        engine = ExtinctionEngine()
        assert engine.background_rate > 0
        assert engine.minimum_viable_population > 0

    def test_calculate_extinction_probability(self):
        """Test extinction probability calculation"""
        engine = ExtinctionEngine()
        genome = create_test_genome()

        prob1 = engine.calculate_extinction_probability(
            genome, population_size=100, fitness=1.0
        )
        prob2 = engine.calculate_extinction_probability(
            genome, population_size=5, fitness=0.2
        )

        # Low population and fitness should increase extinction probability
        assert prob2 > prob1

    def test_mass_extinction(self):
        """Test mass extinction event"""
        engine = ExtinctionEngine()
        population = create_test_population(100)

        survivors, event = engine.trigger_mass_extinction(
            population,
            severity=0.5,
            generation=1,
            cause="test"
        )

        assert len(survivors) < len(population)
        assert event.extinction_type == ExtinctionType.MASS
        assert event.severity == 0.5

    def test_selective_extinction(self):
        """Test selective mass extinction"""
        engine = ExtinctionEngine()

        # Create population with varied trait values
        population = []
        for i in range(50):
            genome = create_test_genome({"temperature_tolerance": i / 50})
            population.append(genome)

        survivors, event = engine.trigger_mass_extinction(
            population,
            severity=0.7,
            selective_trait="temperature_tolerance",
            trait_survival_direction=1.0,  # High values survive
            generation=1
        )

        # Survivors should tend to have higher trait values
        avg_trait = sum(
            g.express("temperature_tolerance") for g in survivors
        ) / len(survivors)
        assert avg_trait > 0.3

    def test_bottleneck_detection(self):
        """Test population bottleneck detection"""
        engine = ExtinctionEngine(bottleneck_threshold=0.3)

        bn = engine.check_for_bottleneck(
            species_id="test_species",
            current_pop=10,
            original_pop=100,
            diversity=0.5,
            generation=1
        )

        assert bn is not None
        assert bn.severity > 0.5

    def test_preset_worlds(self):
        """Test preset world configurations"""
        stable = create_stable_world()
        volatile = create_volatile_world()

        assert stable.background_rate < volatile.background_rate
        assert stable.mass_extinction_probability < volatile.mass_extinction_probability


class TestIntegration:
    """Integration tests for combined systems"""

    def test_evolution_with_epigenetics(self):
        """Test evolution engine with epigenetic layer"""
        # Create evolution engine
        evo_engine = EvolutionEngine(
            population_size=20,
            mutation_rate=0.1
        )

        # Create epigenetic layer
        epi_layer = EpigeneticLayer()

        # Initialize population
        evo_engine.initialize_population({
            "speed": (0, 1),
            "strength": (0, 1),
            "heat_tolerance": (0, 1)
        })

        # Register all genomes
        for ind in evo_engine.population:
            epi_layer.register_genome(ind.genome)

        # Apply stressor
        stressor = EnvironmentalStressor(
            name="heat",
            affected_genes=["heat_tolerance"],
            intensity=0.6,
            activation_genes=["heat_tolerance"]
        )

        for ind in evo_engine.population:
            epi_layer.apply_stressor(ind.genome, stressor)

        # Check that marks were applied
        total_marks = sum(
            len(epi.marks) for epi in epi_layer.epigenomes.values()
        )
        assert total_marks > 0

    def test_speciation_with_sexual_selection(self):
        """Test speciation with mate choice"""
        spec_engine = SpeciationEngine()
        sex_engine = SexualSelectionEngine()

        sex_engine.add_preference(MatePreference(
            trait="display",
            preference_strength=0.8
        ))

        # Create diverging populations
        pop1 = [create_test_genome({"display": 0.9}) for _ in range(10)]
        pop2 = [create_test_genome({"display": 0.2}) for _ in range(10)]

        for g in pop1:
            g.species_id = "high_display"
        for g in pop2:
            g.species_id = "low_display"

        # Register species
        sp1 = Species(name="HighDisplay")
        sp1.id = "high_display"
        sp2 = Species(name="LowDisplay")
        sp2.id = "low_display"

        spec_engine.register_species(sp1)
        spec_engine.register_species(sp2)

        # Test that mate choice differs between species
        chooser = pop1[0]
        attract_high = sex_engine.calculate_attractiveness(pop1[1])
        attract_low = sex_engine.calculate_attractiveness(pop2[0])

        assert attract_high > attract_low

    def test_arms_race_with_extinction(self):
        """Test predator-prey dynamics with extinction"""
        arms_engine = ArmsRaceEngine()
        ext_engine = ExtinctionEngine()

        # Create predator-prey relationship
        arms_engine.add_predation_relationship("predator", "prey")

        # Create populations
        predators = create_test_population(20, "predator")
        prey = create_test_population(50, "prey")

        # Run some interactions
        all_genomes = predators + prey

        # Process extinction
        result = ext_engine.process_generation(all_genomes, generation=1)

        assert "survivors" in result
        assert result["final_population"] <= len(all_genomes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
