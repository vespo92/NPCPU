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

### Multi-Bot Ecosystem

Run multiple organisms for ecosystem simulation:

```python
organisms = [
    MinecraftOrganism(name="Alpha", config=config),
    MinecraftOrganism(name="Beta", config=config),
    MinecraftOrganism(name="Gamma", config=config)
]

# Connect all
for org in organisms:
    await org.connect()

# Run ecosystem
while any(org.is_alive for org in organisms):
    for org in organisms:
        if org.is_alive:
            await org.tick()
    await asyncio.sleep(0.5)
```

## Commands Reference

The bot accepts these JSON commands via stdin:

| Command | Parameters | Description |
|---------|------------|-------------|
| `connect` | host, port, username | Connect to server |
| `disconnect` | - | Disconnect |
| `move_to` | x, y, z | Navigate to position |
| `dig` | x, y, z | Mine a block |
| `attack` | entity_id | Attack entity |
| `eat` | - | Eat food from inventory |
| `chat` | message | Send chat message |
| `find_blocks` | block, radius | Find blocks nearby |
| `find_entities` | type, hostile_only | Find entities |

## Future Plans

- [ ] Building shelter at night
- [ ] Tool crafting and use
- [ ] Trading with villagers
- [ ] Multi-bot communication
- [ ] Territory marking
- [ ] Breeding/reproduction in Minecraft
- [ ] Memory of locations
- [ ] Learning from experience
