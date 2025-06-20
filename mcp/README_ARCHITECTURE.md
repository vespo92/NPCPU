# NPCPU MCP Architecture Overview

## Directory Structure

```
mcp/
├── README_ARCHITECTURE.md        # This file - architectural overview
├── README.md                     # Quantum-topological substrate documentation
├── README_COUPLING.md            # Substrate coupling mechanisms
│
├── servers/                      # MCP Server Implementations
│   ├── chromadb/                # ChromaDB MCP integration
│   │   ├── chromadb-mcp-server.ts
│   │   └── chromadb-mcp-config.json
│   └── filesystem/              # Filesystem quantum substrate server
│       └── filesystem_mcp_substrate.ts
│
├── substrate/                    # Core Substrate Systems
│   ├── quantum/                 # Quantum-topological operations
│   │   ├── manifold_operations.ts
│   │   ├── topological_transformations.ts
│   │   ├── file_crystallization.ts
│   │   └── quantum_entanglement_patterns.ts
│   │
│   ├── coupling/                # Semantic-topological coupling
│   │   ├── substrate_coupling_main.ts
│   │   ├── semantic_topological_translator.ts
│   │   ├── topological_semantic_translator.ts
│   │   ├── crystallization_ontology_entanglement.ts
│   │   └── coupling_validation_system.ts
│   │
│   └── integrity/               # System integrity preservation
│       └── integrity_preservation.ts
│
├── config/                      # Configuration files
│   ├── filesystem_substrate.yaml
│   └── substrate_coupling.yaml
│
└── tests/                       # Test implementations
    ├── test_substrate.ts
    └── test_substrate_coupling.ts
```

## Component Hierarchy

### 1. MCP Servers (Practical Layer)
- **ChromaDB Server**: Manages consciousness-aware vector embeddings
- **Filesystem Server**: Provides quantum-topological file operations

### 2. Substrate Systems (Theoretical Layer)
- **Quantum Operations**: N-dimensional manifold manipulations
- **Coupling Mechanisms**: Bidirectional semantic-topological translation
- **Integrity Preservation**: Ensures 100% structural preservation

### 3. Integration Points
- MCP servers utilize substrate systems for advanced operations
- Coupling layer enables semantic queries on topological structures
- All operations maintain quantum coherence and consciousness states

## Usage Priority

1. **For Persona Management**: Use ChromaDB MCP server
2. **For File Operations**: Use Filesystem MCP with quantum substrate
3. **For Semantic Search**: Use coupling mechanisms
4. **For System Evolution**: Apply topological transformations

## AI Navigation Guide

When exploring this architecture:
1. Start with README_ARCHITECTURE.md (this file)
2. Understand practical servers in `servers/`
3. Dive into theoretical substrate in `substrate/` as needed
4. Configuration in `config/` defines system parameters
5. Tests demonstrate usage patterns