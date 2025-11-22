"""
Simulation State Persistence

Save and load complete simulation states for:
- Checkpointing long-running simulations
- Resuming interrupted runs
- Sharing and reproducing experiments
- Analysis and debugging
"""

import json
import gzip
import pickle
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import hashlib

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class SimulationSnapshot:
    """
    A complete snapshot of simulation state at a point in time.

    Contains everything needed to resume or analyze a simulation:
    - Configuration
    - World state
    - All organism states
    - Population dynamics
    - Metrics and history
    """
    # Metadata (required fields first)
    snapshot_id: str
    simulation_id: str
    simulation_name: str
    created_at: str

    # Timing
    tick: int
    start_time: float
    snapshot_time: float

    # Configuration
    config: Dict[str, Any]

    # World state
    world_state: Dict[str, Any]

    # Population state
    population_state: Dict[str, Any]
    organisms: List[Dict[str, Any]]

    # Metrics
    metrics: Dict[str, Any]
    population_history: List[int]
    fitness_history: List[float]
    generation_history: List[Dict[str, Any]]

    # Fields with defaults (must come last)
    version: str = "1.0.0"
    checksum: Optional[str] = None


class SimulationPersistence:
    """
    Handles saving and loading of simulation states.

    Supports multiple formats:
    - JSON (human-readable, larger)
    - Compressed JSON (smaller, still inspectable)
    - Pickle (fastest, most complete)

    Example:
        persistence = SimulationPersistence("./saves")

        # Save simulation
        snapshot = persistence.save(simulation)

        # Load and resume
        simulation = persistence.load(snapshot_id)
        simulation.resume()
    """

    VERSION = "1.0.0"

    def __init__(self, save_dir: str = "./simulation_saves"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        simulation,
        format: str = "json.gz",
        description: str = ""
    ) -> str:
        """
        Save simulation state to disk.

        Args:
            simulation: The Simulation instance to save
            format: Output format ('json', 'json.gz', 'pickle')
            description: Optional description of this save

        Returns:
            Snapshot ID that can be used to load this state
        """
        snapshot = self._create_snapshot(simulation)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"{simulation.config.name.replace(' ', '_')}_{timestamp}_{snapshot.tick}"
        snapshot.snapshot_id = snapshot_id

        # Calculate checksum
        snapshot.checksum = self._calculate_checksum(snapshot)

        # Save based on format
        if format == "json":
            filepath = self._save_json(snapshot, snapshot_id)
        elif format == "json.gz":
            filepath = self._save_json_compressed(snapshot, snapshot_id)
        elif format == "pickle":
            filepath = self._save_pickle(snapshot, snapshot_id)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Save metadata index
        self._update_index(snapshot, filepath, description)

        return snapshot_id

    def _create_snapshot(self, simulation) -> SimulationSnapshot:
        """Create a snapshot from simulation state"""
        import time

        # Serialize organisms
        organisms = []
        for org_id, organism in simulation.population.organisms.items():
            organisms.append(self._serialize_organism(organism))

        # Serialize world state
        world_state = self._serialize_world(simulation.world)

        # Serialize population state
        population_state = self._serialize_population(simulation.population)

        # Create snapshot
        return SimulationSnapshot(
            snapshot_id="",  # Will be set later
            simulation_id=simulation.id,
            simulation_name=simulation.config.name,
            created_at=datetime.now().isoformat(),
            version=self.VERSION,
            tick=simulation.tick_count,
            start_time=simulation.start_time or time.time(),
            snapshot_time=time.time(),
            config={
                "name": simulation.config.name,
                "seed": simulation.config.seed,
                "max_ticks": simulation.config.max_ticks,
                "initial_population": simulation.config.initial_population,
                "carrying_capacity": simulation.config.carrying_capacity,
                "tick_rate": simulation.config.tick_rate,
                "auto_save_interval": simulation.config.auto_save_interval,
                "output_dir": simulation.config.output_dir,
                "verbose": simulation.config.verbose
            },
            world_state=world_state,
            population_state=population_state,
            organisms=organisms,
            metrics=simulation.metrics.__dict__.copy(),
            population_history=simulation.population_history.copy(),
            fitness_history=simulation.fitness_history.copy(),
            generation_history=[
                {
                    "generation": g.generation,
                    "population": g.population,
                    "avg_fitness": g.avg_fitness,
                    "best_traits": g.best_traits,
                    "worst_traits": g.worst_traits,
                    "diversity": g.diversity
                }
                for g in simulation.generation_history
            ]
        )

    def _serialize_organism(self, organism) -> Dict[str, Any]:
        """Serialize an organism to a dictionary"""
        # Get ID and name from identity object
        org_id = getattr(organism.identity, 'id', str(id(organism)))
        org_name = getattr(organism.identity, 'name', 'Unknown')

        # Get age - might be in lifecycle or directly on organism
        age = getattr(organism, 'age', 0)
        if age == 0 and hasattr(organism, 'lifecycle'):
            age = getattr(organism.lifecycle, 'age', 0)

        # Build serialized data with safe attribute access
        data = {
            "id": org_id,
            "name": org_name,
            "state": organism.state.value if hasattr(organism.state, 'value') else str(organism.state),
            "age": age,
        }

        # Lifecycle data
        if hasattr(organism, 'lifecycle'):
            lc = organism.lifecycle
            data["lifecycle"] = {
                "stage": lc.stage.value if hasattr(lc.stage, 'value') else str(lc.stage),
                "maturity": getattr(lc, 'maturity', 0.0),
                "age_in_stage": getattr(lc, 'age_in_stage', 0),
                "health": getattr(lc, 'health', 1.0)
            }

        # Metabolism data
        if hasattr(organism, 'metabolism'):
            met = organism.metabolism
            data["metabolism"] = {
                "energy": getattr(met, 'energy', 0.0),
                "max_energy": getattr(met, 'max_energy', 100.0),
                "base_metabolic_rate": getattr(met, 'base_metabolic_rate', 1.0),
                "efficiency": getattr(met, 'efficiency', 1.0)
            }

        # Homeostasis data
        if hasattr(organism, 'homeostasis'):
            hom = organism.homeostasis
            data["homeostasis"] = {
                "temperature": getattr(hom, 'temperature', 37.0),
                "ph_level": getattr(hom, 'ph_level', 7.4),
                "hydration": getattr(hom, 'hydration', 1.0),
                "stress": getattr(hom, 'stress', 0.0)
            }

        # Identity data
        if hasattr(organism, 'identity'):
            ident = organism.identity
            lineage = getattr(ident, 'lineage', []) or getattr(ident, 'parent_ids', [])
            data["identity"] = {
                "name": getattr(ident, 'name', 'Unknown'),
                "traits": dict(getattr(ident, 'traits', {})),
                "lineage": list(lineage) if lineage else []
            }

        # Consciousness data
        if hasattr(organism, 'consciousness'):
            cons = organism.consciousness
            data["consciousness"] = {
                "awareness_level": getattr(cons, 'awareness_level', 0.5),
                "capabilities": dict(getattr(cons, 'capabilities', {}))
            }

        return data

    def _serialize_world(self, world) -> Dict[str, Any]:
        """Serialize world state"""
        return {
            "name": world.config.name,
            "tick": world.tick,
            "time_of_day": world.time_of_day,
            "day": world.day,
            "year": world.year,
            "season": world.season.value if hasattr(world.season, 'value') else str(world.season),
            "global_resources": {
                name: {
                    "amount": getattr(pool, 'amount', 0),
                    "max_capacity": getattr(pool, 'max_capacity', 1000),
                    "regeneration_rate": getattr(pool, 'regeneration_rate', 10)
                }
                for name, pool in world.global_resources.items()
            },
            "regions": [
                {
                    "name": region.name,
                    "resources": region.resources.copy()
                }
                for region in world.regions.values()
            ],
            "active_events": [
                {
                    "type": event.type.value if hasattr(event.type, 'value') else str(getattr(event, 'type', 'unknown')),
                    "name": getattr(event, 'name', ''),
                    "duration": getattr(event, 'duration', 0),
                    "magnitude": getattr(event, 'magnitude', 0.5),
                    "started_at": getattr(event, 'started_at', 0)
                }
                for event in getattr(world, 'active_events', [])
            ]
        }

    def _serialize_population(self, population) -> Dict[str, Any]:
        """Serialize population dynamics state"""
        result = {
            "name": getattr(population, 'name', 'unknown'),
            "carrying_capacity": getattr(population, 'carrying_capacity', 100),
            "generation": getattr(population, 'generation', 0),
            "total_births": getattr(population, 'total_births', 0),
            "total_deaths": getattr(population, 'total_deaths', 0),
            "organism_ids": list(population.organisms.keys()) if hasattr(population, 'organisms') else []
        }

        # Serialize social network if present
        if hasattr(population, 'social'):
            social = population.social
            # Simplify relationships to just counts/keys
            relationships = getattr(social, 'relationships', {})
            result["social_network"] = {
                "relationship_count": len(relationships),
                "relationship_pairs": list(relationships.keys()) if isinstance(relationships, dict) else [],
                "interaction_count": dict(getattr(social, 'interaction_count', {}))
            }

        return result

    def _calculate_checksum(self, snapshot: SimulationSnapshot) -> str:
        """Calculate checksum for integrity verification"""
        # Create a copy without the checksum field
        data = asdict(snapshot)
        data['checksum'] = None
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _save_json(self, snapshot: SimulationSnapshot, snapshot_id: str) -> Path:
        """Save as plain JSON"""
        filepath = self.save_dir / f"{snapshot_id}.json"
        with open(filepath, 'w') as f:
            json.dump(asdict(snapshot), f, indent=2, default=str)
        return filepath

    def _save_json_compressed(self, snapshot: SimulationSnapshot, snapshot_id: str) -> Path:
        """Save as gzipped JSON"""
        filepath = self.save_dir / f"{snapshot_id}.json.gz"
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(asdict(snapshot), f, default=str)
        return filepath

    def _save_pickle(self, snapshot: SimulationSnapshot, snapshot_id: str) -> Path:
        """Save as pickle (fastest, most complete)"""
        filepath = self.save_dir / f"{snapshot_id}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        return filepath

    def _update_index(self, snapshot: SimulationSnapshot, filepath: Path, description: str):
        """Update the saves index file"""
        index_path = self.save_dir / "index.json"

        # Load existing index
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {"saves": []}

        # Add new entry
        index["saves"].append({
            "snapshot_id": snapshot.snapshot_id,
            "simulation_name": snapshot.simulation_name,
            "tick": snapshot.tick,
            "organisms": len(snapshot.organisms),
            "created_at": snapshot.created_at,
            "filepath": str(filepath),
            "description": description,
            "checksum": snapshot.checksum
        })

        # Keep sorted by date
        index["saves"].sort(key=lambda x: x["created_at"], reverse=True)

        # Save index
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

    def load(self, snapshot_id: str) -> 'SimulationSnapshot':
        """
        Load a simulation snapshot.

        Args:
            snapshot_id: The ID of the snapshot to load

        Returns:
            SimulationSnapshot that can be used to restore simulation
        """
        # Try different formats
        for ext in ['.json.gz', '.json', '.pkl']:
            filepath = self.save_dir / f"{snapshot_id}{ext}"
            if filepath.exists():
                return self._load_file(filepath)

        raise FileNotFoundError(f"No snapshot found with ID: {snapshot_id}")

    def _load_file(self, filepath: Path) -> SimulationSnapshot:
        """Load snapshot from file"""
        suffix = ''.join(filepath.suffixes)

        if suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return SimulationSnapshot(**data)

        elif suffix == '.json.gz':
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            return SimulationSnapshot(**data)

        elif suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)

        raise ValueError(f"Unknown file format: {suffix}")

    def list_saves(self) -> List[Dict[str, Any]]:
        """List all available saves"""
        index_path = self.save_dir / "index.json"

        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
            return index.get("saves", [])

        # Fallback: scan directory
        saves = []
        for filepath in self.save_dir.glob("*.json*"):
            if filepath.name == "index.json":
                continue
            saves.append({
                "snapshot_id": filepath.stem.replace('.json', ''),
                "filepath": str(filepath)
            })
        return saves

    def delete(self, snapshot_id: str):
        """Delete a saved snapshot"""
        # Try different formats
        for ext in ['.json.gz', '.json', '.pkl']:
            filepath = self.save_dir / f"{snapshot_id}{ext}"
            if filepath.exists():
                filepath.unlink()

        # Update index
        index_path = self.save_dir / "index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
            index["saves"] = [
                s for s in index["saves"]
                if s["snapshot_id"] != snapshot_id
            ]
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)

    def verify(self, snapshot_id: str) -> bool:
        """Verify snapshot integrity"""
        snapshot = self.load(snapshot_id)
        expected_checksum = snapshot.checksum
        snapshot.checksum = None
        actual_checksum = self._calculate_checksum(snapshot)
        return expected_checksum == actual_checksum


def restore_simulation(snapshot: SimulationSnapshot):
    """
    Restore a Simulation instance from a snapshot.

    Args:
        snapshot: The snapshot to restore from

    Returns:
        A new Simulation instance in the saved state
    """
    from simulation.runner import Simulation, SimulationConfig, SimulationMetrics, GenerationStats
    from organism.digital_body import DigitalBody, OrganismState
    from organism.lifecycle import LifecycleStage
    from ecosystem.world import World, WorldConfig, Season
    from ecosystem.population import Population

    # Recreate configuration
    config = SimulationConfig(
        name=snapshot.config["name"],
        seed=snapshot.config.get("seed"),
        max_ticks=snapshot.config["max_ticks"],
        initial_population=snapshot.config["initial_population"],
        carrying_capacity=snapshot.config["carrying_capacity"],
        tick_rate=snapshot.config.get("tick_rate", 0.0),
        auto_save_interval=snapshot.config.get("auto_save_interval", 1000),
        output_dir=snapshot.config.get("output_dir", "./simulation_output"),
        verbose=snapshot.config.get("verbose", True)
    )

    # Create simulation without initializing population
    sim = Simulation(config)
    sim.id = snapshot.simulation_id
    sim.tick_count = snapshot.tick
    sim.start_time = snapshot.start_time

    # Clear default population
    sim.population.organisms.clear()

    # Restore organisms
    for org_data in snapshot.organisms:
        organism = DigitalBody(name=org_data["name"])
        organism.id = org_data["id"]
        organism.age = org_data["age"]

        # Restore lifecycle
        if "lifecycle" in org_data:
            lc = org_data["lifecycle"]
            try:
                organism.lifecycle.stage = LifecycleStage(lc["stage"])
            except (ValueError, KeyError):
                pass
            organism.lifecycle.maturity = lc.get("maturity", 0.0)
            organism.lifecycle.age_in_stage = lc.get("age_in_stage", 0)
            organism.lifecycle.health = lc.get("health", 1.0)

        # Restore metabolism
        if "metabolism" in org_data:
            met = org_data["metabolism"]
            organism.metabolism.energy = met.get("energy", 100.0)
            organism.metabolism.max_energy = met.get("max_energy", 100.0)
            organism.metabolism.base_metabolic_rate = met.get("base_metabolic_rate", 1.0)
            organism.metabolism.efficiency = met.get("efficiency", 1.0)

        # Restore homeostasis
        if "homeostasis" in org_data:
            hom = org_data["homeostasis"]
            organism.homeostasis.temperature = hom.get("temperature", 37.0)
            organism.homeostasis.ph_level = hom.get("ph_level", 7.4)
            organism.homeostasis.hydration = hom.get("hydration", 1.0)
            organism.homeostasis.stress = hom.get("stress", 0.0)

        # Restore identity/traits
        if "identity" in org_data:
            ident = org_data["identity"]
            organism.identity.traits = ident.get("traits", {})
            organism.identity.lineage = ident.get("lineage", [])

        # Restore consciousness
        if "consciousness" in org_data:
            cons = org_data["consciousness"]
            organism.consciousness.awareness_level = cons.get("awareness_level", 0.5)
            organism.consciousness.capabilities = cons.get("capabilities", {})

        # Add to population
        sim.population.organisms[organism.id] = organism

    # Restore metrics
    for key, value in snapshot.metrics.items():
        if hasattr(sim.metrics, key):
            setattr(sim.metrics, key, value)

    # Restore history
    sim.population_history = snapshot.population_history.copy()
    sim.fitness_history = snapshot.fitness_history.copy()

    # Restore generation history
    sim.generation_history = [
        GenerationStats(
            generation=g["generation"],
            population=g["population"],
            avg_fitness=g["avg_fitness"],
            best_traits=g.get("best_traits", {}),
            worst_traits=g.get("worst_traits", {}),
            diversity=g.get("diversity", 0.0)
        )
        for g in snapshot.generation_history
    ]

    # Restore world state
    if snapshot.world_state:
        ws = snapshot.world_state
        sim.world.tick = ws.get("tick", 0)
        sim.world.time_of_day = ws.get("time_of_day", 0)
        sim.world.day = ws.get("day", 0)
        sim.world.year = ws.get("year", 0)
        try:
            sim.world.season = Season(ws.get("season", "spring"))
        except (ValueError, KeyError):
            pass

    # Mark as running
    from simulation.runner import SimulationState
    sim.state = SimulationState.RUNNING

    return sim


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulation State Management")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list", help="List saved simulations")
    list_parser.add_argument("--dir", default="./simulation_saves", help="Save directory")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show snapshot details")
    info_parser.add_argument("snapshot_id", help="Snapshot ID to inspect")
    info_parser.add_argument("--dir", default="./simulation_saves", help="Save directory")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify snapshot integrity")
    verify_parser.add_argument("snapshot_id", help="Snapshot ID to verify")
    verify_parser.add_argument("--dir", default="./simulation_saves", help="Save directory")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a snapshot")
    delete_parser.add_argument("snapshot_id", help="Snapshot ID to delete")
    delete_parser.add_argument("--dir", default="./simulation_saves", help="Save directory")

    args = parser.parse_args()

    if args.command == "list":
        persistence = SimulationPersistence(args.dir)
        saves = persistence.list_saves()

        if not saves:
            print("No saved simulations found.")
        else:
            print(f"\nFound {len(saves)} saved simulation(s):\n")
            for save in saves:
                print(f"  ID: {save.get('snapshot_id', 'unknown')}")
                print(f"  Name: {save.get('simulation_name', 'N/A')}")
                print(f"  Tick: {save.get('tick', 'N/A')}")
                print(f"  Organisms: {save.get('organisms', 'N/A')}")
                print(f"  Created: {save.get('created_at', 'N/A')}")
                print()

    elif args.command == "info":
        persistence = SimulationPersistence(args.dir)
        snapshot = persistence.load(args.snapshot_id)

        print(f"\nSnapshot: {snapshot.snapshot_id}")
        print(f"{'='*50}")
        print(f"Simulation: {snapshot.simulation_name}")
        print(f"Version: {snapshot.version}")
        print(f"Created: {snapshot.created_at}")
        print(f"\nState:")
        print(f"  Tick: {snapshot.tick}")
        print(f"  Organisms: {len(snapshot.organisms)}")
        print(f"  Generations: {len(snapshot.generation_history)}")
        print(f"\nMetrics:")
        for key, value in snapshot.metrics.items():
            print(f"  {key}: {value}")

    elif args.command == "verify":
        persistence = SimulationPersistence(args.dir)
        is_valid = persistence.verify(args.snapshot_id)

        if is_valid:
            print(f"Snapshot {args.snapshot_id} is valid.")
        else:
            print(f"WARNING: Snapshot {args.snapshot_id} has been modified!")

    elif args.command == "delete":
        persistence = SimulationPersistence(args.dir)
        persistence.delete(args.snapshot_id)
        print(f"Deleted snapshot: {args.snapshot_id}")

    else:
        parser.print_help()
