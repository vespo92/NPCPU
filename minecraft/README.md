# NPCPU Minecraft Integration

Transform Minecraft into a virtual substrate for NPCPU digital organisms.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NPCPU Python                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MinecraftOrganism                                         │   │
│  │   ├── DigitalBody (consciousness, mood, metabolism)      │   │
│  │   ├── MinecraftSensory (world perception)                │   │
│  │   ├── MinecraftMotor (actions → commands)                │   │
│  │   └── MinecraftMetabolism (health/hunger sync)           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                      JSON over stdio                             │
│                             │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MinecraftBridge (Python)                                  │   │
│  │   - Command/response protocol                             │   │
│  │   - State synchronization                                 │   │
│  │   - Event handling                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
                        subprocess
                             │
┌─────────────────────────────────────────────────────────────────┐
│                    Node.js/Mineflayer                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ bot.js                                                    │   │
│  │   - Minecraft server connection                          │   │
│  │   - Pathfinding (mineflayer-pathfinder)                  │   │
│  │   - Combat, mining, building                             │   │
│  │   - World state reporting                                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
                        network
                             │
┌─────────────────────────────────────────────────────────────────┐
│                    Minecraft Server                             │
│   (Java Edition, offline mode for testing)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install Node.js dependencies

```bash
cd minecraft
npm install
```

### 2. Start a Minecraft Server

For testing, use a local server with offline mode:

```bash
# Using Docker
docker run -d -p 25565:25565 -e EULA=TRUE -e ONLINE_MODE=FALSE itzg/minecraft-server

# Or download server.jar from minecraft.net
java -Xmx1G -jar server.jar nogui
```

Edit `server.properties`:
```
online-mode=false
```

### 3. Run NPCPU Organism in Minecraft

```python
import asyncio
from minecraft.organism_adapter import MinecraftOrganism, MinecraftConfig

async def main():
    # Configure connection
    config = MinecraftConfig(
        host="localhost",
        port=25565,
        username="NPCPU_Alpha"
    )

    # Create organism
    organism = MinecraftOrganism(name="Alpha", config=config)

    # Connect to server
    if await organism.connect():
        print("Connected!")

        # Run organism lifecycle
        while organism.is_alive:
            await organism.tick()
            await asyncio.sleep(0.5)

    await organism.disconnect()

asyncio.run(main())
```

## System Mappings

### Sensory System → Minecraft Perception

| NPCPU Sensor | Minecraft Data |
|--------------|----------------|
| THREAT_DETECTOR | Hostile mobs nearby |
| RESOURCE_MONITOR | Ore/wood/food blocks |
| SOCIAL_RECEPTOR | Players nearby |
| SELF_MONITOR | Health, hunger, position |
| TIME_SENSE | Day/night cycle |

### Motor System → Minecraft Actions

| NPCPU Action | Minecraft Command |
|--------------|-------------------|
| MOVE | Pathfinder navigation |
| ACQUIRE | Mining blocks |
| CONSUME | Eating food |
| DEFEND | Attacking entities |
| COMMUNICATE | Chat messages |
| EXPLORE | Random movement |
| REST | Stay still |

### Metabolism → Minecraft Vitals

| NPCPU System | Minecraft Equivalent |
|--------------|---------------------|
| Energy | Food/saturation |
| Health | Health points |
| Resources | Inventory |
| Starvation | Hunger damage |

### World Regions → Biomes

| NPCPU Region | Minecraft Biomes |
|--------------|------------------|
| FERTILE | Plains, Forest |
| BARREN | Desert, Ice |
| FRONTIER | Jungle, Swamp |
| HOSTILE | Nether, Deep Dark |
| SANCTUARY | Mushroom Island |

## Features

### Autonomous Behavior

The organism makes decisions based on:
- **Survival**: Low health → flee/heal
- **Hunger**: Low food → find/eat food
- **Threats**: Hostiles nearby → fight/flee (based on aggression trait)
- **Curiosity**: Explore new areas
- **Resources**: Gather materials

### Emotional System

Minecraft events affect organism mood:
- Finding resources → Dopamine (reward)
- Taking damage → Cortisol (stress)
- Night time → Melatonin (rest)
- Social interactions → Oxytocin (bonding)

### Memory System

Organisms remember and learn from experiences:

```python
from minecraft.memory import MemorySystem, MemoryType, EmotionalValence

memory = MemorySystem()

# Remember resource locations
memory.remember_location(
    Position(100, 64, 200),
    MemoryType.RESOURCE,
    "iron_ore_vein",
    {"block_count": 8},
    EmotionalValence.POSITIVE
)

# Remember dangerous areas
memory.remember_location(
    Position(50, 64, 50),
    MemoryType.DANGER,
    "creeper_spawn",
    {},
    EmotionalValence.NEGATIVE
)

# Recall nearby resources when needed
resources = memory.recall_nearby(current_pos, MemoryType.RESOURCE, radius=100)
```

Memory types:
- **Spatial**: Resource locations, danger zones, shelters
- **Episodic**: Events and their outcomes
- **Social**: Relationships with other entities
- **Procedural**: Learned skills and sequences

### Advanced Behaviors

High-level behavior patterns:

```python
from minecraft.behaviors import BehaviorManager

manager = BehaviorManager(bridge, memory)

# Automatic survival (always running)
# - Monitors health/hunger
# - Flees from danger
# - Seeks shelter at night

# Queue crafting task
manager.crafting.set_target("stone_pickaxe")
manager.queue_behavior(manager.crafting)

# Shelter building when night approaches
if manager.shelter.can_execute():
    await manager.shelter.execute()
```

Behaviors include:
- **ShelterBehavior**: Build emergency shelter before nightfall
- **CraftingBehavior**: Craft tools and items
- **TradingBehavior**: Trade with villagers
- **SurvivalBehavior**: Continuous health/safety monitoring

### Colony System

Run coordinated multi-agent colonies:

```python
from minecraft.colony import Colony, ColonyRole

colony = Colony("AlphaColony", config)

# Add members with specialized roles
await colony.add_member("Scout1", ColonyRole.SCOUT)
await colony.add_member("Guard1", ColonyRole.GUARD)
await colony.add_member("Gatherer1", ColonyRole.GATHERER)
await colony.add_member("Gatherer2", ColonyRole.GATHERER)
await colony.add_member("Leader", ColonyRole.LEADER)

# Run colony
await colony.run(max_ticks=5000)
```

Colony features:
- **Role Specialization**: Scout, Guard, Gatherer, Builder, Leader, Healer
- **Shared Memory**: Colony-wide knowledge of resources and dangers
- **Communication**: Chat-based messaging protocol
- **Task Distribution**: Automatic task assignment based on roles
- **Collective Defense**: Coordinated response to threats

### Ecosystem Runner

Main entry point for simulations:

```bash
# Single organism mode
python minecraft/ecosystem_runner.py --mode single --host localhost

# Colony mode
python minecraft/ecosystem_runner.py --mode colony --colony-name MyColony \
    --scouts 2 --gatherers 3 --guards 1 --leader 1

# Ecosystem mode (multiple independent organisms)
python minecraft/ecosystem_runner.py --mode ecosystem --organisms 5

# With memory persistence
python minecraft/ecosystem_runner.py --mode single --save-memories
```

## Commands Reference

The bot accepts these JSON commands via stdin:

| Command | Parameters | Description |
|---------|------------|-------------|
| `connect` | host, port, username | Connect to server |
| `disconnect` | - | Disconnect |
| `move_to` | x, y, z | Navigate to position |
| `dig` | x, y, z | Mine a block |
| `place` | position, block | Place a block |
| `attack` | entity_id | Attack entity |
| `attack_nearest_hostile` | - | Attack nearest hostile |
| `eat` | - | Eat food from inventory |
| `craft` | recipe, count | Craft an item |
| `chat` | message | Send chat message |
| `find_blocks` | block, radius | Find blocks nearby |
| `find_entities` | type, hostile_only | Find entities |
| `get_state` | - | Get bot state |
| `get_world_state` | - | Get world info |

## Module Structure

```
minecraft/
├── __init__.py           # Package exports
├── bridge.py             # Low-level Mineflayer communication
├── organism_adapter.py   # NPCPU-to-Minecraft system mapping
├── world_adapter.py      # World/biome mapping
├── memory.py             # Spatial/episodic/social memory
├── behaviors.py          # High-level behavior patterns
├── colony.py             # Multi-agent coordination
├── ecosystem_runner.py   # Main entry point
├── bot.js                # Mineflayer bot (Node.js)
├── package.json          # Node dependencies
└── README.md             # This file
```

## Implemented Features

- [x] Building shelter at night
- [x] Tool crafting and use
- [x] Trading with villagers
- [x] Multi-bot communication
- [x] Territory management
- [x] Memory of locations
- [x] Learning from experience
- [x] Colony coordination
- [x] Role-based specialization
- [x] Shared knowledge systems

## Future Plans

- [ ] Breeding/reproduction mechanics
- [ ] Evolution and trait inheritance
- [ ] Cross-world migration
- [ ] Advanced trading economy
- [ ] Structure blueprints
- [ ] Voice/visual communication
