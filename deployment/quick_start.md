# Quick Start: Deploy NPCPU + LocalGreenChain to Cloudflare

## Prerequisites
- Cloudflare account (free)
- GitHub account (for CI/CD)
- Node.js installed locally
- Claude Code with MCP extension

## Step 1: Domain Setup (10 minutes)

### Add Domains to Cloudflare
1. Log into Cloudflare Dashboard
2. Add Site → Enter `npcpu.com`
3. Select Free plan
4. Update nameservers at your registrar
5. Repeat for `localgreenchain.com`

### Enable Services
```bash
# In Cloudflare Dashboard for each domain:
1. Speed → Optimization → Enable all
2. SSL/TLS → Set to "Full (strict)"
3. Security → Enable Bot Fight Mode
4. Workers → Enable Workers subdomain
```

## Step 2: Create GitHub Repos (5 minutes)

```bash
# Create repos
gh repo create npcpu-site --public
gh repo create localgreenchain-site --public

# Clone locally
git clone https://github.com/YOUR_USERNAME/npcpu-site
git clone https://github.com/YOUR_USERNAME/localgreenchain-site
```

## Step 3: Initialize NPCPU Site (20 minutes)

```bash
cd npcpu-site

# Create with Cloudflare template
npm create cloudflare@latest . -- --template=pages

# Install dependencies
npm install

# Create site structure
mkdir -p public/{demo,docs,api}
```

### Create Landing Page
```html
<!-- public/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NPCPU - Distributed Consciousness Infrastructure</title>
    <link rel="stylesheet" href="/styles.css">
</head>
<body>
    <header>
        <h1>NPCPU</h1>
        <p>The Neural Network That Grows With Humanity</p>
    </header>
    
    <main>
        <section class="hero">
            <h2>Democratizing AI Through Distributed Consciousness</h2>
            <p>Join the world's first truly distributed, transparent, and collectively-owned artificial consciousness.</p>
            <a href="/demo" class="cta">Try Live Demo</a>
            <a href="https://github.com/YOUR_USERNAME/NPCPU" class="cta secondary">View on GitHub</a>
        </section>
        
        <section class="features">
            <div class="feature">
                <h3>🧠 Self-Modifying</h3>
                <p>Improves its own algorithms through recursive bootstrapping</p>
            </div>
            <div class="feature">
                <h3>🌐 Distributed</h3>
                <p>Runs across voluntary nodes worldwide</p>
            </div>
            <div class="feature">
                <h3>🔍 Transparent</h3>
                <p>Every decision auditable and explainable</p>
            </div>
        </section>
        
        <section class="initiative">
            <h2>First Initiative: LocalGreenChain</h2>
            <p>See NPCPU in action with our carbon-negative blockchain</p>
            <a href="https://localgreenchain.com" class="cta">Explore LocalGreenChain</a>
        </section>
    </main>
    
    <script src="/app.js"></script>
</body>
</html>
```

### Create Worker for API
```javascript
// functions/api/[[route]].js
export async function onRequest(context) {
  const { request, env } = context;
  const url = new URL(request.url);
  
  // Handle different API routes
  if (url.pathname === '/api/status') {
    return Response.json({
      status: 'online',
      version: '1.0.0',
      nodes: await getNodeCount(env)
    });
  }
  
  if (url.pathname === '/api/agent/coordinate') {
    const data = await request.json();
    return handleAgentCoordination(data, env);
  }
  
  return new Response('Not Found', { status: 404 });
}

async function getNodeCount(env) {
  // Get from KV or return mock data
  return env.KV?.get('node_count') || 5;
}

async function handleAgentCoordination(data, env) {
  // Simple coordination logic
  const { agents, dimension } = data;
  
  // Detect clashes (simplified)
  const clashes = [];
  for (let i = 0; i < agents.length; i++) {
    for (let j = i + 1; j < agents.length; j++) {
      if (agents[i].dimension === agents[j].dimension) {
        clashes.push({
          agents: [agents[i].id, agents[j].id],
          dimension,
          severity: 0.5
        });
      }
    }
  }
  
  return Response.json({ clashes });
}
```

## Step 4: Initialize LocalGreenChain Site (20 minutes)

```bash
cd ../localgreenchain-site

# Create with Cloudflare template
npm create cloudflare@latest . -- --template=pages

# Install dependencies
npm install

# Create app structure
mkdir -p public/{app,dashboard,map}
```

### Create Green Chain Landing
```html
<!-- public/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LocalGreenChain - The Living Blockchain</title>
    <link rel="stylesheet" href="/styles.css">
</head>
<body>
    <header>
        <h1>🌱 LocalGreenChain</h1>
        <p>Every Plant a Node, Every Garden a Network</p>
    </header>
    
    <main>
        <section class="hero">
            <h2>The World's First Carbon-Negative Blockchain</h2>
            <p>Track plant genealogy, monitor ecosystems, and earn rewards for growing.</p>
            <a href="/app" class="cta">Start Planting</a>
            <a href="/map" class="cta secondary">View Global Forest</a>
        </section>
        
        <section class="stats">
            <div class="stat">
                <h3 id="plant-count">0</h3>
                <p>Plants Tracked</p>
            </div>
            <div class="stat">
                <h3 id="carbon-count">0 kg</h3>
                <p>Carbon Sequestered</p>
            </div>
            <div class="stat">
                <h3 id="gardener-count">0</h3>
                <p>Active Gardeners</p>
            </div>
        </section>
        
        <section class="how-it-works">
            <h2>How It Works</h2>
            <ol>
                <li>🌰 Plant a seed and register it</li>
                <li>📸 Track growth with photos</li>
                <li>🔗 Connect with nearby plants</li>
                <li>🪙 Earn LEAF tokens for care</li>
                <li>🌍 Contribute to global reforestation</li>
            </ol>
        </section>
    </main>
    
    <script src="/app.js"></script>
</body>
</html>
```

### Create Blockchain Worker
```javascript
// functions/api/[[route]].js
export async function onRequest(context) {
  const { request, env } = context;
  const url = new URL(request.url);
  
  switch (url.pathname) {
    case '/api/plant':
      return handlePlantRegistration(request, env);
    case '/api/growth':
      return handleGrowthUpdate(request, env);
    case '/api/stats':
      return getGlobalStats(env);
    default:
      return new Response('Not Found', { status: 404 });
  }
}

async function handlePlantRegistration(request, env) {
  const data = await request.json();
  const { species, location, parentTokens } = data;
  
  // Generate plant token
  const token = `${species.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}`;
  
  // Store in KV
  await env.PLANTS.put(token, JSON.stringify({
    species,
    location: anonymizeLocation(location),
    parentTokens: parentTokens || [],
    createdAt: new Date().toISOString()
  }));
  
  // Update grid mapping
  const gridKey = `grid:${location.grid}`;
  const existing = await env.PLANTS.get(gridKey);
  const plants = existing ? JSON.parse(existing) : [];
  plants.push(token);
  await env.PLANTS.put(gridKey, JSON.stringify(plants));
  
  // Update stats
  await incrementStat(env, 'total_plants');
  
  return Response.json({ 
    success: true, 
    token,
    message: 'Plant registered successfully!'
  });
}

async function handleGrowthUpdate(request, env) {
  const data = await request.json();
  const { token, metrics, imageHash } = data;
  
  // Get existing plant
  const plantData = await env.PLANTS.get(token);
  if (!plantData) {
    return Response.json({ error: 'Plant not found' }, { status: 404 });
  }
  
  // Calculate carbon sequestered
  const carbonKg = metrics.biomassKg * 0.5 * 0.4;
  
  // Store growth record
  const growthKey = `growth:${token}:${Date.now()}`;
  await env.PLANTS.put(growthKey, JSON.stringify({
    metrics,
    carbonKg,
    imageHash,
    timestamp: new Date().toISOString()
  }));
  
  // Update carbon total
  await incrementStat(env, 'total_carbon', carbonKg);
  
  return Response.json({
    success: true,
    carbonSequestered: carbonKg,
    reward: carbonKg * 1.0 // 1 LEAF per kg carbon
  });
}

async function getGlobalStats(env) {
  const stats = {
    totalPlants: parseInt(await env.PLANTS.get('stat:total_plants') || '0'),
    totalCarbon: parseFloat(await env.PLANTS.get('stat:total_carbon') || '0'),
    totalGardeners: parseInt(await env.PLANTS.get('stat:total_gardeners') || '0')
  };
  
  return Response.json(stats);
}

function anonymizeLocation(location) {
  // Round to 100m grid square
  return {
    grid: `${Math.round(location.lat * 100) / 100}_${Math.round(location.lon * 100) / 100}_100m`,
    climate: location.climate
  };
}

async function incrementStat(env, stat, amount = 1) {
  const key = `stat:${stat}`;
  const current = parseFloat(await env.PLANTS.get(key) || '0');
  await env.PLANTS.put(key, String(current + amount));
}
```

## Step 5: Deploy to Cloudflare (10 minutes)

### Deploy NPCPU
```bash
cd npcpu-site

# Login to Cloudflare
npx wrangler login

# Deploy
npm run deploy

# It will give you a URL like: https://npcpu.pages.dev
# Connect custom domain in Cloudflare Dashboard:
# Pages → npcpu → Custom domains → Add npcpu.com
```

### Deploy LocalGreenChain
```bash
cd ../localgreenchain-site

# Deploy
npm run deploy

# Connect domain:
# Pages → localgreenchain → Custom domains → Add localgreenchain.com
```

## Step 6: Create KV Namespaces (5 minutes)

```bash
# For NPCPU
wrangler kv:namespace create "AGENTS"
wrangler kv:namespace create "COORDINATION"

# For LocalGreenChain
wrangler kv:namespace create "PLANTS"
wrangler kv:namespace create "GRIDS"

# Add bindings to wrangler.toml
```

## Step 7: Setup MCP Integration (15 minutes)

### Create MCP Config
```json
// claude-mcp-config.json
{
  "servers": {
    "npcpu": {
      "url": "https://npcpu.com/api/mcp",
      "description": "NPCPU distributed consciousness"
    },
    "greenchain": {
      "url": "https://localgreenchain.com/api/mcp",
      "description": "LocalGreenChain plant tracking"
    }
  }
}
```

### Add MCP Endpoints
```javascript
// Add to both workers
if (url.pathname === '/api/mcp') {
  return handleMCPRequest(request, env);
}

async function handleMCPRequest(request, env) {
  const { method, params } = await request.json();
  
  switch (method) {
    case 'tools':
      return Response.json({
        tools: [{
          name: 'register_plant',
          description: 'Register a new plant',
          inputSchema: {
            type: 'object',
            properties: {
              species: { type: 'string' },
              location: { type: 'object' }
            }
          }
        }]
      });
      
    case 'invoke':
      const { tool, input } = params;
      if (tool === 'register_plant') {
        return handlePlantRegistration(
          new Request('', { 
            method: 'POST', 
            body: JSON.stringify(input) 
          }), 
          env
        );
      }
      break;
  }
  
  return Response.json({ error: 'Unknown method' });
}
```

## Step 8: Test Everything (10 minutes)

### Test NPCPU
```bash
# API Status
curl https://npcpu.com/api/status

# Agent coordination
curl -X POST https://npcpu.com/api/agent/coordinate \
  -H "Content-Type: application/json" \
  -d '{"agents":[{"id":"a1","dimension":"semantic"},{"id":"a2","dimension":"semantic"}],"dimension":"semantic"}'
```

### Test LocalGreenChain
```bash
# Register plant
curl -X POST https://localgreenchain.com/api/plant \
  -H "Content-Type: application/json" \
  -d '{"species":"Oak Tree","location":{"lat":37.7749,"lon":-122.4194,"climate":"temperate"}}'

# Get stats
curl https://localgreenchain.com/api/stats
```

### Test with Claude Code
```
1. Open Claude Code
2. Add MCP server URLs
3. Try: "Register an oak tree in San Francisco"
4. Claude should use the LocalGreenChain API
```

## Done! 🎉

You now have:
- ✅ NPCPU.com live globally
- ✅ LocalGreenChain.com live globally  
- ✅ Zero hosting costs
- ✅ Automatic scaling
- ✅ MCP integration
- ✅ Enterprise security

### Next Steps
1. Add more interactive features
2. Implement D1 database for complex queries
3. Add Durable Objects for stateful coordination
4. Create mobile apps
5. Launch marketing campaign

Total time: ~90 minutes
Total cost: $0/month (within free tier)
Global reach: ♾️