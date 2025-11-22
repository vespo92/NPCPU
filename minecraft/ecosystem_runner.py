"""
Minecraft Ecosystem Runner

Main entry point for running NPCPU digital organisms in Minecraft.
Supports single organisms, colonies, and full ecosystem simulations.

Features:
- Single organism mode
- Colony mode (multiple coordinated organisms)
- Full ecosystem mode (populations, evolution)
- Statistics and monitoring
- Memory persistence
"""

import asyncio
import argparse
import signal
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minecraft.bridge import MinecraftConfig, Position
from minecraft.organism_adapter import MinecraftOrganism
from minecraft.colony import Colony, ColonyRole
from minecraft.memory import MemorySystem
from minecraft.behaviors import BehaviorManager


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EcosystemConfig:
    """Configuration for ecosystem runner"""
    # Server
    host: str = "localhost"
    port: int = 25565
    version: str = "1.20.1"
    auth: str = "offline"

    # Mode
    mode: str = "single"              # single, colony, ecosystem
    organism_count: int = 1
    colony_name: str = "NPCPUColony"

    # Simulation
    tick_rate: float = 0.5            # Seconds between ticks
    max_ticks: int = 10000
    auto_respawn: bool = True

    # Persistence
    save_memories: bool = True
    memory_file: str = "minecraft_memories.json"
    stats_file: str = "minecraft_stats.json"

    # Colony roles distribution
    role_distribution: Dict[str, int] = field(default_factory=lambda: {
        "scout": 1,
        "gatherer": 2,
        "guard": 1,
        "builder": 1,
        "leader": 1
    })


# ============================================================================
# Ecosystem Runner
# ============================================================================

class MinecraftEcosystemRunner:
    """
    Runs NPCPU organisms in Minecraft.

    Modes:
    - Single: One organism exploring and surviving
    - Colony: Multiple organisms working together
    - Ecosystem: Full population dynamics

    Example:
        runner = MinecraftEcosystemRunner(config)
        await runner.run()
    """

    def __init__(self, config: EcosystemConfig):
        self.config = config
        self.mc_config = MinecraftConfig(
            host=config.host,
            port=config.port,
            version=config.version,
            auth=config.auth
        )

        # Entities
        self.organism: Optional[MinecraftOrganism] = None
        self.colony: Optional[Colony] = None
        self.organisms: List[MinecraftOrganism] = []

        # State
        self.running = False
        self.tick_count = 0
        self.start_time = 0.0

        # Statistics
        self.stats = {
            "ticks": 0,
            "organisms_spawned": 0,
            "organisms_died": 0,
            "resources_gathered": 0,
            "structures_built": 0,
            "trades_completed": 0,
            "dangers_encountered": 0,
            "peak_population": 0
        }

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print("\nShutdown signal received...")
        self.running = False

    async def run(self):
        """Run the ecosystem"""
        self.running = True
        self.start_time = time.time()

        self._print_banner()

        try:
            if self.config.mode == "single":
                await self._run_single()
            elif self.config.mode == "colony":
                await self._run_colony()
            elif self.config.mode == "ecosystem":
                await self._run_ecosystem()
            else:
                print(f"Unknown mode: {self.config.mode}")
                return

        except Exception as e:
            print(f"Error: {e}")
        finally:
            await self._cleanup()
            self._save_stats()
            self._print_summary()

    async def _run_single(self):
        """Run a single organism"""
        print(f"Starting single organism mode...")
        print(f"Connecting to {self.config.host}:{self.config.port}")

        # Create organism
        self.organism = MinecraftOrganism(
            name="NPCPU_Organism",
            config=self.mc_config
        )

        # Create behavior manager
        memory = MemorySystem()
        if self.config.save_memories:
            self._load_memories(memory)

        behavior_manager = BehaviorManager(self.organism.bridge, memory)

        # Connect
        if not await self.organism.connect():
            print("Failed to connect!")
            return

        self.stats["organisms_spawned"] = 1
        print(f"Connected! Starting simulation...")

        # Main loop
        while self.running and self.tick_count < self.config.max_ticks:
            self.tick_count += 1

            # Run organism tick
            await self.organism.tick()

            # Run behavior manager
            result = await behavior_manager.tick()
            if result:
                print(f"[{self.tick_count:5d}] {result.message}")

            # Check death
            if not self.organism.is_alive:
                self.stats["organisms_died"] += 1
                if self.config.auto_respawn:
                    print("Organism died. Respawning...")
                    # In Minecraft, would wait for respawn
                    await asyncio.sleep(5)
                else:
                    print("Organism died. Ending simulation.")
                    break

            # Print status periodically
            if self.tick_count % 50 == 0:
                self._print_single_status()

            await asyncio.sleep(self.config.tick_rate)

        # Save memories
        if self.config.save_memories:
            self._save_memories(memory)

    async def _run_colony(self):
        """Run a colony of organisms"""
        print(f"Starting colony mode: {self.config.colony_name}")
        print(f"Connecting to {self.config.host}:{self.config.port}")

        # Create colony
        self.colony = Colony(
            self.config.colony_name,
            self.mc_config,
            Position(0, 64, 0)  # Base position
        )

        # Add members based on role distribution
        role_map = {
            "scout": ColonyRole.SCOUT,
            "gatherer": ColonyRole.GATHERER,
            "guard": ColonyRole.GUARD,
            "builder": ColonyRole.BUILDER,
            "leader": ColonyRole.LEADER,
            "worker": ColonyRole.WORKER,
            "healer": ColonyRole.HEALER
        }

        member_id = 0
        for role_name, count in self.config.role_distribution.items():
            role = role_map.get(role_name, ColonyRole.WORKER)
            for i in range(count):
                member_id += 1
                await self.colony.add_member(f"Member{member_id}", role)
                print(f"  Added {role.value}: Member{member_id}")

        self.stats["organisms_spawned"] = len(self.colony.members)
        self.stats["peak_population"] = len(self.colony.members)

        # Load shared memories
        if self.config.save_memories:
            self._load_memories(self.colony.shared_memory)

        # Run colony
        print(f"\nColony '{self.config.colony_name}' ready with {len(self.colony.members)} members")
        print("Starting simulation...\n")

        await self.colony.run(
            max_ticks=self.config.max_ticks,
            tick_delay=self.config.tick_rate
        )

        # Save memories
        if self.config.save_memories:
            self._save_memories(self.colony.shared_memory)

        self.tick_count = self.colony.tick_count

    async def _run_ecosystem(self):
        """Run full ecosystem simulation"""
        print(f"Starting ecosystem mode with {self.config.organism_count} organisms")

        # Create multiple independent organisms
        for i in range(self.config.organism_count):
            org_config = MinecraftConfig(
                host=self.config.host,
                port=self.config.port,
                username=f"NPCPU_{i+1}",
                version=self.config.version,
                auth=self.config.auth
            )
            organism = MinecraftOrganism(f"Organism_{i+1}", org_config)
            self.organisms.append(organism)

        self.stats["organisms_spawned"] = len(self.organisms)
        self.stats["peak_population"] = len(self.organisms)

        # Connect all
        connected = 0
        for organism in self.organisms:
            try:
                if await organism.connect():
                    connected += 1
                    print(f"  Connected: {organism.config.username}")
                await asyncio.sleep(1.0)  # Stagger connections
            except Exception as e:
                print(f"  Failed to connect {organism.config.username}: {e}")

        if connected == 0:
            print("No organisms connected!")
            return

        print(f"\nConnected {connected}/{len(self.organisms)} organisms")
        print("Starting ecosystem simulation...\n")

        # Main loop
        while self.running and self.tick_count < self.config.max_ticks:
            self.tick_count += 1

            alive_count = 0
            for organism in self.organisms:
                try:
                    await organism.tick()
                    if organism.is_alive:
                        alive_count += 1
                except Exception as e:
                    pass

            # Track deaths
            dead = len(self.organisms) - alive_count
            if dead > self.stats["organisms_died"]:
                self.stats["organisms_died"] = dead

            # Status update
            if self.tick_count % 50 == 0:
                print(f"[Tick {self.tick_count:5d}] Alive: {alive_count}/{len(self.organisms)}")

            # End if all dead
            if alive_count == 0 and not self.config.auto_respawn:
                print("All organisms died!")
                break

            await asyncio.sleep(self.config.tick_rate)

    async def _cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")

        if self.organism:
            await self.organism.disconnect()

        if self.colony:
            await self.colony.disconnect_all()

        for organism in self.organisms:
            try:
                await organism.disconnect()
            except Exception:
                pass

    def _load_memories(self, memory: MemorySystem):
        """Load memories from file"""
        path = Path(self.config.memory_file)
        if path.exists():
            try:
                memory.load(str(path))
                print(f"Loaded memories from {path}")
            except Exception as e:
                print(f"Failed to load memories: {e}")

    def _save_memories(self, memory: MemorySystem):
        """Save memories to file"""
        try:
            memory.save(self.config.memory_file)
            print(f"Saved memories to {self.config.memory_file}")
        except Exception as e:
            print(f"Failed to save memories: {e}")

    def _save_stats(self):
        """Save statistics to file"""
        self.stats["ticks"] = self.tick_count
        self.stats["runtime_seconds"] = time.time() - self.start_time

        try:
            with open(self.config.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"Saved stats to {self.config.stats_file}")
        except Exception as e:
            print(f"Failed to save stats: {e}")

    def _print_banner(self):
        """Print startup banner"""
        banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     ███╗   ██╗██████╗  ██████╗██████╗ ██╗   ██╗              ║
    ║     ████╗  ██║██╔══██╗██╔════╝██╔══██╗██║   ██║              ║
    ║     ██╔██╗ ██║██████╔╝██║     ██████╔╝██║   ██║              ║
    ║     ██║╚██╗██║██╔═══╝ ██║     ██╔═══╝ ██║   ██║              ║
    ║     ██║ ╚████║██║     ╚██████╗██║     ╚██████╔╝              ║
    ║     ╚═╝  ╚═══╝╚═╝      ╚═════╝╚═╝      ╚═════╝               ║
    ║                                                               ║
    ║              MINECRAFT ECOSYSTEM RUNNER                       ║
    ║           Digital Life in Virtual Worlds                      ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"    Mode: {self.config.mode.upper()}")
        print(f"    Server: {self.config.host}:{self.config.port}")
        print(f"    Max ticks: {self.config.max_ticks}")
        print()

    def _print_single_status(self):
        """Print status for single organism mode"""
        if not self.organism:
            return

        state = self.organism.bridge.bot_state
        body = self.organism.body

        print(f"\n--- Tick {self.tick_count} ---")
        print(f"  Position: ({int(state.position.x)}, {int(state.position.y)}, {int(state.position.z)})")
        print(f"  Health: {state.health}/20 | Food: {state.food}/20")
        print(f"  Age: {body.lifecycle.age} ticks")
        print(f"  Energy: {body.metabolism.energy:.1f}")

    def _print_summary(self):
        """Print final summary"""
        runtime = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"  Mode: {self.config.mode}")
        print(f"  Runtime: {runtime:.1f} seconds")
        print(f"  Ticks completed: {self.tick_count}")
        print()
        print("Statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        print("=" * 60)


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="NPCPU Minecraft Ecosystem Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single organism mode
  python ecosystem_runner.py --mode single --host localhost --port 25565

  # Colony mode with custom composition
  python ecosystem_runner.py --mode colony --colony-name MyColony --scouts 2 --gatherers 3

  # Ecosystem mode with multiple organisms
  python ecosystem_runner.py --mode ecosystem --organisms 5

  # Save memories for persistence
  python ecosystem_runner.py --mode single --save-memories --memory-file my_memories.json
        """
    )

    # Server options
    parser.add_argument("--host", default="localhost", help="Minecraft server host")
    parser.add_argument("--port", type=int, default=25565, help="Minecraft server port")
    parser.add_argument("--version", default="1.20.1", help="Minecraft version")

    # Mode options
    parser.add_argument(
        "--mode",
        choices=["single", "colony", "ecosystem"],
        default="single",
        help="Simulation mode"
    )
    parser.add_argument("--organisms", type=int, default=1, help="Number of organisms (ecosystem mode)")
    parser.add_argument("--colony-name", default="NPCPUColony", help="Colony name (colony mode)")

    # Colony role distribution
    parser.add_argument("--scouts", type=int, default=1, help="Number of scouts")
    parser.add_argument("--gatherers", type=int, default=2, help="Number of gatherers")
    parser.add_argument("--guards", type=int, default=1, help="Number of guards")
    parser.add_argument("--builders", type=int, default=1, help="Number of builders")
    parser.add_argument("--leader", type=int, default=1, help="Number of leaders")

    # Simulation options
    parser.add_argument("--tick-rate", type=float, default=0.5, help="Seconds between ticks")
    parser.add_argument("--max-ticks", type=int, default=10000, help="Maximum ticks to run")
    parser.add_argument("--no-respawn", action="store_true", help="Disable auto-respawn")

    # Persistence
    parser.add_argument("--save-memories", action="store_true", help="Save memories to file")
    parser.add_argument("--memory-file", default="minecraft_memories.json", help="Memory file path")
    parser.add_argument("--stats-file", default="minecraft_stats.json", help="Stats file path")

    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()

    config = EcosystemConfig(
        host=args.host,
        port=args.port,
        version=args.version,
        mode=args.mode,
        organism_count=args.organisms,
        colony_name=args.colony_name,
        tick_rate=args.tick_rate,
        max_ticks=args.max_ticks,
        auto_respawn=not args.no_respawn,
        save_memories=args.save_memories,
        memory_file=args.memory_file,
        stats_file=args.stats_file,
        role_distribution={
            "scout": args.scouts,
            "gatherer": args.gatherers,
            "guard": args.guards,
            "builder": args.builders,
            "leader": args.leader
        }
    )

    runner = MinecraftEcosystemRunner(config)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
