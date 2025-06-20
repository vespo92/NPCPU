# Claude MCP Setup for Direct API Access

## Install These MCP Servers in Claude Desktop

### 1. Cloudflare MCP Server
Gives Claude direct access to all Cloudflare APIs:

```json
{
  "mcpServers": {
    "cloudflare": {
      "command": "npx",
      "args": ["-y", "@cloudflare/mcp-server-cloudflare"],
      "env": {
        "CLOUDFLARE_ACCOUNT_ID": "your-account-id",
        "CLOUDFLARE_API_TOKEN": "your-api-token"
      }
    }
  }
}
```

### 2. GitHub MCP Server
Direct GitHub API access:

```json
{
  "github": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-token"
    }
  }
}
```

### 3. Postgres MCP Server
Direct database access:

```json
{
  "postgres": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-postgres"],
    "env": {
      "DATABASE_URL": "postgresql://user:pass@host:5432/db"
    }
  }
}
```

### 4. Fetch/HTTP MCP Server
General HTTP API access:

```json
{
  "fetch": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-fetch"],
    "env": {}
  }
}
```

## With These Installed, Claude Can:

Instead of:
```bash
# Old way - shell scripts
curl -X POST https://api.cloudflare.com/...
```

Claude can do:
```
# New way - direct API access
cloudflare.deployPages("localgreenchain", "./public")
cloudflare.createKVNamespace("PLANTS")
github.createRepository("new-project")
postgres.query("SELECT * FROM plants")
fetch.get("https://api.example.com/data")
```

## Get Your Tokens:

### Cloudflare:
1. Log into Cloudflare Dashboard
2. My Profile → API Tokens
3. Create Token → Use template "Edit Cloudflare Workers"
4. Account ID is in the right sidebar of dashboard

### GitHub:
1. GitHub Settings → Developer settings
2. Personal access tokens → Tokens (classic)
3. Generate new token with repo, workflow permissions

## Full Claude Config Example:

Edit your Claude Desktop config (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "cloudflare": {
      "command": "npx",
      "args": ["-y", "@cloudflare/mcp-server-cloudflare"],
      "env": {
        "CLOUDFLARE_ACCOUNT_ID": "abc123",
        "CLOUDFLARE_API_TOKEN": "xyz789"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"
      }
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "env": {}
    }
  }
}
```

## Now Claude Can Deploy Everything Via APIs!

- Deploy to Cloudflare Pages
- Create/manage Workers
- Create KV namespaces  
- Manage GitHub repos
- Make any HTTP request
- No more shell scripts!