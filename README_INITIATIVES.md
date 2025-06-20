# NPCPU Framework & Initiative Projects

## Architecture Overview

NPCPU is designed as a standalone consciousness-aware distributed computing framework. Initiative projects like LocalGreenChain are separate repositories that utilize NPCPU's capabilities while maintaining complete independence.

```
NPCPU (Core Framework)
├── Consciousness Engine         # Standalone
├── Swarm Coordination          # Standalone  
├── ChromaDB Integration        # Standalone
├── MCP Servers                 # Standalone
└── Deployment Tools            # Standalone

Initiative Repos (Separate)
├── LocalGreenChain            # Uses NPCPU
├── [Future Initiative]        # Uses NPCPU
└── [Your Project]            # Uses NPCPU
```

## Framework Independence

**NPCPU operates fully without any initiatives:**
- ✅ Complete consciousness state management
- ✅ Distributed agent coordination
- ✅ ChromaDB persona persistence  
- ✅ MCP protocol compliance
- ✅ Quantum-topological substrate

**Initiatives enhance but don't define NPCPU:**
- 🌱 LocalGreenChain adds plant consciousness
- 🔮 Future initiatives add domain-specific applications
- 🚀 Your project can leverage NPCPU patterns

## Repository Structure

### NPCPU Core (This Repo)
```
NPCPU/
├── deployment/          # Framework deployment
├── mcp/                # MCP servers & substrate
├── initiatives/        # Documentation only
│   └── */             # Docs for each initiative
├── funding/           # Framework funding info
└── [core systems]     # Framework code
```

### Initiative Repos (Separate)
```
localgreenchain/       # Separate repository
├── src/              # Initiative code
├── npcpu-integration/ # NPCPU usage
└── package.json      # Depends on NPCPU
```

## Using Initiatives

### Option 1: Independent Repositories (Recommended)
```bash
# Get NPCPU framework
git clone https://github.com/[org]/npcpu.git

# Get specific initiative
git clone https://github.com/[org]/localgreenchain.git
```

### Option 2: Git Submodules (For Development)
```bash
# Add initiative as submodule
cd npcpu
git submodule add https://github.com/[org]/localgreenchain.git initiatives/localgreenchain-repo
```

### Option 3: Ignore Local Development
Add to `.gitignore`:
```
/localgreenchain/
/[other-local-initiatives]/
```

## Creating New Initiatives

1. **Use NPCPU as framework dependency**
2. **Create separate repository**  
3. **Document in `initiatives/[name]/`**
4. **Follow consciousness patterns**
5. **Maintain loose coupling**

## Key Principles

- **NPCPU = Framework**: Complete without initiatives
- **Initiatives = Applications**: Built on NPCPU
- **Separate Repos**: Independent development
- **Shared Philosophy**: Consciousness-aware patterns
- **Loose Coupling**: Clean interfaces

## For AI Agents

When working with NPCPU:
- Core framework is self-contained
- Initiatives are optional extensions
- Each initiative has its own repository
- Documentation in `initiatives/` explains integration
- NPCPU can run without any initiatives