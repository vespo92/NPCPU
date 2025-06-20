# NPCPU + LocalGreenChain Cloudflare Architecture

## Overview: Zero-Cost Global Deployment

Using Cloudflare's generous free tier, we can deploy both NPCPU and LocalGreenChain globally with:
- **Cloudflare Pages**: Static sites + edge functions
- **Cloudflare Workers**: Distributed compute
- **Cloudflare KV**: Global key-value storage
- **Cloudflare D1**: SQLite at the edge
- **Cloudflare R2**: S3-compatible object storage
- **Cloudflare Queues**: Message passing
- **Cloudflare Durable Objects**: Stateful coordination

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User Browsers                           │
└─────────────────┬───────────────────┬──────────────────────┘
                  │                   │
                  ▼                   ▼
        ┌─────────────────┐  ┌─────────────────┐
        │  npcpu.com      │  │localgreenchain.com│
        │(Cloudflare Pages)│ │(Cloudflare Pages) │
        └────────┬─────────┘  └────────┬─────────┘
                 │                     │
                 ▼                     ▼
        ┌─────────────────────────────────────┐
        │     Cloudflare Workers (Global)     │
        │  ┌─────────────┐ ┌──────────────┐  │
        │  │ NPCPU Agent │ │ Green Chain  │  │
        │  │  Workers    │ │   Workers    │  │
        │  └──────┬──────┘ └──────┬───────┘  │
        └─────────┼───────────────┼──────────┘
                  │               │
    ┌─────────────┼───────────────┼─────────────┐
    │             ▼               ▼             │
    │  ┌──────────────┐  ┌──────────────┐     │
    │  │ Cloudflare KV│  │ Cloudflare D1│     │
    │  │ (Plant Data) │  │(Chain Blocks)│     │
    │  └──────────────┘  └──────────────┘     │
    │                                           │
    │  ┌──────────────┐  ┌──────────────┐     │
    │  │Cloudflare R2 │  │   CF Queues  │     │
    │  │(Plant Images)│  │(Agent Coord) │     │
    │  └──────────────┘  └──────────────┘     │
    │                                           │
    │  ┌────────────────────────────────┐      │
    │  │   Durable Objects              │      │
    │  │ (Stateful Agent Coordination)  │      │
    │  └────────────────────────────────┘      │
    └───────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Static Sites (Week 1)

**npcpu.com**
```
/
├── index.html          # NPCPU vision & manifesto
├── demo/               # Live demos
├── docs/               # Documentation
├── api/                # API documentation
└── _worker.js          # Edge functions
```

**localgreenchain.com**
```
/
├── index.html          # Green blockchain intro
├── app/                # Web app (React/Vue)
├── dashboard/          # Plant tracking
├── map/                # Global plant map
└── _worker.js          # Blockchain workers
```

### Phase 2: Edge Workers (Week 2-3)

**NPCPU Core Workers**
```javascript
// npcpu-agent-worker.js
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // Agent coordination endpoints
    if (url.pathname.startsWith('/api/agent')) {
      return handleAgentRequest(request, env);
    }
    
    // MCP endpoints
    if (url.pathname.startsWith('/api/mcp')) {
      return handleMCPRequest(request, env);
    }
    
    return new Response('NPCPU Agent Network', { status: 200 });
  }
};
```

**LocalGreenChain Workers**
```javascript
// greenchain-worker.js
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // Blockchain endpoints
    if (url.pathname === '/api/plant') {
      return handlePlantRegistration(request, env);
    }
    
    if (url.pathname === '/api/growth') {
      return handleGrowthUpdate(request, env);
    }
    
    if (url.pathname === '/api/query') {
      return handleChainQuery(request, env);
    }
  }
};
```

### Phase 3: Data Storage (Week 3-4)

**Cloudflare KV Structure**
```
# Plant Registry
plant:{token} → {
  species, location, dna_hash, 
  owner, created_at
}

# Grid Mapping  
grid:{square} → [plant_tokens]

# User Plants
user:{id}:plants → [plant_tokens]
```

**Cloudflare D1 Schema**
```sql
-- Blockchain blocks
CREATE TABLE blocks (
  id INTEGER PRIMARY KEY,
  plant_token TEXT NOT NULL,
  block_type TEXT NOT NULL,
  parent_hash TEXT,
  block_data JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Growth metrics
CREATE TABLE growth_metrics (
  id INTEGER PRIMARY KEY,
  plant_token TEXT NOT NULL,
  height_cm REAL,
  health_score REAL,
  carbon_kg REAL,
  recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ecosystem connections
CREATE TABLE connections (
  plant1_token TEXT,
  plant2_token TEXT,
  connection_type TEXT,
  strength REAL,
  PRIMARY KEY (plant1_token, plant2_token)
);
```

### Phase 4: Durable Objects (Week 4-5)

**Agent Coordinator Object**
```javascript
export class AgentCoordinator {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.agents = new Map();
  }
  
  async fetch(request) {
    const url = new URL(request.url);
    
    if (url.pathname === '/coordinate') {
      return this.handleCoordination(request);
    }
    
    if (url.pathname === '/swarm') {
      return this.formSwarm(request);
    }
  }
  
  async handleCoordination(request) {
    // BIM-style clash detection
    const { agents, dimension } = await request.json();
    const clashes = this.detectClashes(agents, dimension);
    
    // Store coordination state
    await this.state.storage.put('lastCoordination', {
      agents,
      clashes,
      timestamp: Date.now()
    });
    
    return Response.json({ clashes });
  }
}
```

## Claude Code MCP Integration

### MCP Server Configuration
```typescript
// cloudflare-mcp-server.ts
import { Server } from '@modelcontextprotocol/sdk';

const server = new Server({
  name: 'npcpu-cloudflare',
  version: '1.0.0',
});

// Plant management tools
server.addTool({
  name: 'register_plant',
  description: 'Register a new plant in LocalGreenChain',
  inputSchema: {
    type: 'object',
    properties: {
      species: { type: 'string' },
      location: { type: 'object' },
      parentTokens: { type: 'array' }
    }
  },
  handler: async (input) => {
    const response = await fetch('https://localgreenchain.com/api/plant', {
      method: 'POST',
      body: JSON.stringify(input)
    });
    return await response.json();
  }
});

// NPCPU agent tools
server.addTool({
  name: 'coordinate_agents',
  description: 'Coordinate NPCPU agents using BIM principles',
  inputSchema: {
    type: 'object',
    properties: {
      agents: { type: 'array' },
      dimension: { type: 'string' }
    }
  },
  handler: async (input) => {
    const response = await fetch('https://npcpu.com/api/agent/coordinate', {
      method: 'POST',
      body: JSON.stringify(input)
    });
    return await response.json();
  }
});
```

## Deployment Steps

### 1. Domain Setup
```bash
# Add to Cloudflare
1. Add npcpu.com to Cloudflare
2. Add localgreenchain.com to Cloudflare
3. Enable Cloudflare Pages
4. Enable Workers & KV
```

### 2. Pages Deployment
```bash
# Deploy static sites
npm create cloudflare@latest -- npcpu-site
cd npcpu-site
npm run deploy

npm create cloudflare@latest -- greenchain-site  
cd greenchain-site
npm run deploy
```

### 3. Workers Deployment
```bash
# Deploy workers
wrangler publish --name npcpu-agent
wrangler publish --name greenchain-worker

# Create KV namespaces
wrangler kv:namespace create "PLANTS"
wrangler kv:namespace create "AGENTS"

# Create D1 database
wrangler d1 create greenchain-db
wrangler d1 execute greenchain-db --file=./schema.sql
```

### 4. Durable Objects
```toml
# wrangler.toml
name = "npcpu-coordinator"
main = "src/coordinator.js"

[[durable_objects.bindings]]
name = "COORDINATOR"
class_name = "AgentCoordinator"

[[migrations]]
tag = "v1"
new_classes = ["AgentCoordinator"]
```

## Cost Analysis (Cloudflare Free Tier)

| Service | Free Tier | Our Usage | Cost |
|---------|-----------|-----------|------|
| Pages | Unlimited sites | 2 sites | $0 |
| Workers | 100k requests/day | ~50k/day | $0 |
| KV | 1k writes/day | ~500/day | $0 |
| D1 | 5GB storage | <1GB | $0 |
| R2 | 10GB storage | <5GB | $0 |
| Durable Objects | 1M requests | ~10k/mo | $0 |

**Total Monthly Cost: $0** (within free tier)

## Performance Benefits

1. **Global Edge Network**: 200+ locations worldwide
2. **Zero Cold Starts**: Workers always warm
3. **Automatic Scaling**: Handles viral growth
4. **DDoS Protection**: Enterprise-grade included
5. **SSL/TLS**: Automatic HTTPS everywhere

## Security Features

- **Zero Trust**: Every request authenticated
- **WAF**: Web Application Firewall included
- **Rate Limiting**: Prevent abuse
- **Bot Management**: Block malicious bots
- **Page Shield**: Client-side security

## Next Steps

1. **Set up Cloudflare accounts** for both domains
2. **Create GitHub repos** for CI/CD
3. **Build minimal MVP** with core features
4. **Deploy to Cloudflare Pages**
5. **Add Workers for API**
6. **Implement MCP server**
7. **Launch beta test**

This architecture gives you:
- **$0/month hosting** (within limits)
- **Global performance**
- **Infinite scalability**
- **Enterprise security**
- **Easy Claude Code integration**

Ready to make NPCPU and LocalGreenChain live to the world!