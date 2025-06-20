# NPCPU Initiative Projects

This directory contains documentation and integration patterns for independent projects that build on the NPCPU framework.

## Overview

NPCPU is designed as a foundational framework for consciousness-aware distributed systems. Initiative projects are independent repositories that:

1. **Utilize NPCPU** as a framework dependency
2. **Maintain their own repositories** for independent development
3. **Follow NPCPU patterns** while solving specific domain problems
4. **Can evolve independently** while staying philosophically aligned

## Current Initiatives

### LocalGreenChain
- **Purpose**: Blockchain-based local plant economy with ecosystem intelligence
- **Repository**: `github.com/[your-org]/localgreenchain` (separate repo)
- **NPCPU Integration**: 
  - Uses consciousness states for plant health monitoring
  - Implements distributed swarm coordination
  - Leverages MCP servers for deployment

## Project Structure

```
initiatives/
├── README.md                    # This file
├── PROJECT_TEMPLATE.md          # Template for new initiatives
└── [project_name]/              # Documentation only
    ├── README.md               # Project overview & NPCPU integration
    ├── architecture.md         # How it uses NPCPU patterns
    └── integration_guide.md    # Setup instructions
```

## Creating a New Initiative

1. **Start with NPCPU patterns** but maintain independence
2. **Create separate repository** for your project
3. **Document integration** in this directory
4. **Use NPCPU as dependency** not parent repo

## Integration Patterns

### Using NPCPU Framework
```python
from npcpu.consciousness import ConsciousnessState
from npcpu.swarm import SwarmCoordinator
from npcpu.chromadb import NPCPUChromaDBManager

# Your initiative builds on these foundations
```

### Deployment Integration
- Leverage NPCPU's MCP servers
- Use consciousness-aware ChromaDB
- Follow distributed deployment patterns

## Important Notes

- **NPCPU runs independently** - Initiatives are optional extensions
- **Separate repositories** - Each initiative has its own repo
- **Loose coupling** - Use NPCPU through APIs/packages, not direct inclusion
- **This directory contains documentation only** - Actual code lives in separate repos

## Repository Management

### Option 1: Git Submodules (Recommended for development)
```bash
# Add initiative as submodule
git submodule add https://github.com/[your-org]/localgreenchain.git localgreenchain

# Clone with submodules
git clone --recursive https://github.com/[your-org]/npcpu.git
```

### Option 2: Independent Clones (Recommended for production)
```bash
# Clone NPCPU
git clone https://github.com/[your-org]/npcpu.git

# Clone initiatives separately
git clone https://github.com/[your-org]/localgreenchain.git
```

### Option 3: Package Dependencies
```json
{
  "dependencies": {
    "npcpu-framework": "^1.0.0"
  }
}
```

## Philosophy

NPCPU provides the consciousness layer, distributed coordination, and philosophical foundations. Initiatives apply these patterns to specific domains while maintaining their autonomy - much like conscious agents in a swarm.