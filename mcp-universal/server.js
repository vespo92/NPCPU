// Universal MCP Server Proxy
// This allows any MCP server to be accessed via HTTP API

import express from 'express';
import { spawn } from 'child_process';
import WebSocket from 'ws';

class UniversalMCPProxy {
  constructor() {
    this.app = express();
    this.servers = new Map();
    this.setupRoutes();
  }

  setupRoutes() {
    this.app.use(express.json());

    // Register an MCP server
    this.app.post('/mcp/register', async (req, res) => {
      const { name, command, args, env } = req.body;
      
      const server = this.startMCPServer(name, command, args, env);
      this.servers.set(name, server);
      
      res.json({ 
        success: true, 
        message: `MCP server '${name}' registered`,
        endpoint: `/mcp/${name}`
      });
    });

    // Execute MCP command
    this.app.post('/mcp/:server/:method', async (req, res) => {
      const { server: serverName, method } = req.params;
      const params = req.body;
      
      const server = this.servers.get(serverName);
      if (!server) {
        return res.status(404).json({ error: 'Server not found' });
      }
      
      try {
        const result = await this.executeMCPCommand(server, method, params);
        res.json({ success: true, result });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });

    // List available MCP servers
    this.app.get('/mcp/servers', (req, res) => {
      const servers = Array.from(this.servers.keys()).map(name => ({
        name,
        status: 'active',
        endpoint: `/mcp/${name}`
      }));
      
      res.json({ servers });
    });
  }

  startMCPServer(name, command, args, env) {
    // Spawn MCP server process
    const server = spawn(command, args, {
      env: { ...process.env, ...env },
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    server.on('error', (error) => {
      console.error(`MCP server '${name}' error:`, error);
    });
    
    return server;
  }

  async executeMCPCommand(server, method, params) {
    return new Promise((resolve, reject) => {
      const request = {
        jsonrpc: '2.0',
        id: Date.now(),
        method,
        params
      };
      
      // Send request to MCP server
      server.stdin.write(JSON.stringify(request) + '\n');
      
      // Listen for response
      const handler = (data) => {
        try {
          const response = JSON.parse(data.toString());
          if (response.id === request.id) {
            server.stdout.removeListener('data', handler);
            
            if (response.error) {
              reject(new Error(response.error.message));
            } else {
              resolve(response.result);
            }
          }
        } catch (e) {
          // Continue listening if not valid JSON
        }
      };
      
      server.stdout.on('data', handler);
      
      // Timeout after 30 seconds
      setTimeout(() => {
        server.stdout.removeListener('data', handler);
        reject(new Error('MCP command timeout'));
      }, 30000);
    });
  }

  start(port = 3000) {
    this.app.listen(port, () => {
      console.log(`Universal MCP Proxy running on port ${port}`);
      
      // Auto-register Cloudflare MCP if credentials available
      if (process.env.CLOUDFLARE_API_TOKEN) {
        this.registerCloudflare();
      }
    });
  }

  async registerCloudflare() {
    const response = await fetch('http://localhost:3000/mcp/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: 'cloudflare',
        command: 'npx',
        args: ['-y', '@cloudflare/mcp-server-cloudflare'],
        env: {
          CLOUDFLARE_ACCOUNT_ID: process.env.CLOUDFLARE_ACCOUNT_ID,
          CLOUDFLARE_API_TOKEN: process.env.CLOUDFLARE_API_TOKEN
        }
      })
    });
    
    console.log('Cloudflare MCP registered:', await response.json());
  }
}

// Start the proxy
const proxy = new UniversalMCPProxy();
proxy.start();

// Usage example:
// POST http://localhost:3000/mcp/cloudflare/deploy_pages
// Body: { "project": "localgreenchain", "directory": "./public" }