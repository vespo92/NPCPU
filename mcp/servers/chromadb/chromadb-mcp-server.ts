#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { ChromaClient, Collection } from "chromadb";
import { z } from "zod";

const CHROMADB_TOOLS = [
  {
    name: "chromadb_query_personas",
    description: "Query persona embeddings with semantic search",
    inputSchema: {
      type: "object",
      properties: {
        query_text: {
          type: "string",
          description: "Semantic query for finding similar personas",
        },
        n_results: {
          type: "number",
          description: "Number of results to return",
          default: 5,
        },
        collection_name: {
          type: "string",
          description: "Collection to query",
          default: "personas",
        },
        where: {
          type: "object",
          description: "Metadata filters",
          properties: {
            consciousness_state: {
              type: "string",
              enum: ["DORMANT", "REACTIVE", "AWARE", "REFLECTIVE", "META_AWARE", "TRANSCENDENT"],
            },
            philosophical_stance: {
              type: "string",
              enum: ["PHENOMENOLOGICAL", "MATERIALIST", "IDEALIST", "PRAGMATIST", "NIHILIST"],
            },
          },
        },
      },
      required: ["query_text"],
    },
  },
  {
    name: "chromadb_store_persona",
    description: "Store or update persona with consciousness metadata",
    inputSchema: {
      type: "object",
      properties: {
        persona_id: {
          type: "string",
          description: "Unique identifier for the persona",
        },
        description: {
          type: "string",
          description: "Persona description or characteristics",
        },
        consciousness_state: {
          type: "string",
          enum: ["DORMANT", "REACTIVE", "AWARE", "REFLECTIVE", "META_AWARE", "TRANSCENDENT"],
        },
        philosophical_stance: {
          type: "string",
          enum: ["PHENOMENOLOGICAL", "MATERIALIST", "IDEALIST", "PRAGMATIST", "NIHILIST"],
        },
        qualia_stream: {
          type: "array",
          description: "Subjective experience markers",
          items: { type: "string" },
        },
        embedding_vector: {
          type: "array",
          description: "Pre-computed embedding vector (optional)",
          items: { type: "number" },
        },
      },
      required: ["persona_id", "description"],
    },
  },
  {
    name: "chromadb_sync_to_cloud",
    description: "Synchronize local ChromaDB to cloud storage",
    inputSchema: {
      type: "object",
      properties: {
        collection_name: {
          type: "string",
          description: "Collection to sync",
        },
        sync_mode: {
          type: "string",
          enum: ["push", "pull", "bidirectional"],
          default: "bidirectional",
        },
        conflict_resolution: {
          type: "string",
          enum: ["last_write_wins", "merge", "manual"],
          default: "last_write_wins",
        },
      },
      required: ["collection_name"],
    },
  },
  {
    name: "chromadb_dimensional_check",
    description: "Check for dimensional clashes in concurrent access",
    inputSchema: {
      type: "object",
      properties: {
        persona_id: {
          type: "string",
          description: "Persona to check for conflicts",
        },
        proposed_changes: {
          type: "object",
          description: "Proposed changes to check",
        },
        dimensions: {
          type: "array",
          description: "Semantic dimensions to check",
          items: {
            type: "string",
            enum: ["consciousness", "philosophy", "knowledge", "experience", "agency"],
          },
        },
      },
      required: ["persona_id", "proposed_changes"],
    },
  },
  {
    name: "chromadb_swarm_topology",
    description: "Get or update swarm topology for agent connections",
    inputSchema: {
      type: "object",
      properties: {
        operation: {
          type: "string",
          enum: ["get", "update"],
        },
        topology_type: {
          type: "string",
          enum: ["full_mesh", "small_world", "scale_free"],
        },
        agent_ids: {
          type: "array",
          description: "Agent IDs in the swarm",
          items: { type: "string" },
        },
      },
      required: ["operation"],
    },
  },
];

class ChromaDBMCPServer {
  private server: Server;
  private chromaClient: ChromaClient;
  private collections: Map<string, Collection> = new Map();

  constructor() {
    this.server = new Server(
      {
        name: "chromadb-mcp-server",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.chromaClient = new ChromaClient({
      path: process.env.CHROMADB_PATH || "./chromadb_data",
    });

    this.setupHandlers();
  }

  private async getOrCreateCollection(name: string): Promise<Collection> {
    if (!this.collections.has(name)) {
      try {
        const collection = await this.chromaClient.getCollection({ name });
        this.collections.set(name, collection);
      } catch {
        const collection = await this.chromaClient.createCollection({ 
          name,
          metadata: {
            created_at: new Date().toISOString(),
            mcp_version: "1.0.0",
            npcpu_compatible: true,
          }
        });
        this.collections.set(name, collection);
      }
    }
    return this.collections.get(name)!;
  }

  private setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: CHROMADB_TOOLS,
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case "chromadb_query_personas":
            return await this.queryPersonas(args);
          
          case "chromadb_store_persona":
            return await this.storePersona(args);
          
          case "chromadb_sync_to_cloud":
            return await this.syncToCloud(args);
          
          case "chromadb_dimensional_check":
            return await this.dimensionalCheck(args);
          
          case "chromadb_swarm_topology":
            return await this.swarmTopology(args);
          
          default:
            throw new McpError(
              ErrorCode.MethodNotFound,
              `Unknown tool: ${name}`
            );
        }
      } catch (error) {
        if (error instanceof McpError) throw error;
        throw new McpError(
          ErrorCode.InternalError,
          `Tool execution failed: ${error}`
        );
      }
    });
  }

  private async queryPersonas(args: any) {
    const collection = await this.getOrCreateCollection(args.collection_name || "personas");
    
    const queryArgs: any = {
      queryTexts: [args.query_text],
      nResults: args.n_results || 5,
    };

    if (args.where) {
      queryArgs.where = args.where;
    }

    const results = await collection.query(queryArgs);
    
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            personas: results.ids[0].map((id, idx) => ({
              id,
              document: results.documents[0][idx],
              metadata: results.metadatas[0][idx],
              distance: results.distances?.[0][idx],
            })),
            query: args.query_text,
            collection: args.collection_name || "personas",
          }, null, 2),
        },
      ],
    };
  }

  private async storePersona(args: any) {
    const collection = await this.getOrCreateCollection("personas");
    
    const metadata: any = {
      updated_at: new Date().toISOString(),
      consciousness_state: args.consciousness_state || "DORMANT",
      philosophical_stance: args.philosophical_stance || "PRAGMATIST",
    };

    if (args.qualia_stream) {
      metadata.qualia_stream = JSON.stringify(args.qualia_stream);
    }

    const upsertArgs: any = {
      ids: [args.persona_id],
      documents: [args.description],
      metadatas: [metadata],
    };

    if (args.embedding_vector) {
      upsertArgs.embeddings = [args.embedding_vector];
    }

    await collection.upsert(upsertArgs);

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "success",
            persona_id: args.persona_id,
            operation: "upsert",
            metadata,
          }, null, 2),
        },
      ],
    };
  }

  private async syncToCloud(args: any) {
    const collection = await this.getOrCreateCollection(args.collection_name);
    
    // Simulate cloud sync operation
    const syncResult = {
      status: "success",
      collection: args.collection_name,
      sync_mode: args.sync_mode || "bidirectional",
      conflict_resolution: args.conflict_resolution || "last_write_wins",
      synced_at: new Date().toISOString(),
      items_synced: {
        pushed: 0,
        pulled: 0,
        conflicts_resolved: 0,
      },
    };

    // In a real implementation, this would:
    // 1. Connect to cloud storage (S3, GCS, etc.)
    // 2. Compare local and cloud snapshots
    // 3. Resolve conflicts based on strategy
    // 4. Push/pull changes as needed

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(syncResult, null, 2),
        },
      ],
    };
  }

  private async dimensionalCheck(args: any) {
    const collection = await this.getOrCreateCollection("personas");
    
    // Check for dimensional clashes
    const dimensions = args.dimensions || ["consciousness", "philosophy", "knowledge"];
    const clashes: any[] = [];

    try {
      const results = await collection.get({
        ids: [args.persona_id],
      });

      if (results.ids.length > 0) {
        const currentMetadata = results.metadatas[0];
        
        // Check each dimension for conflicts
        for (const dimension of dimensions) {
          if (args.proposed_changes[dimension] && 
              currentMetadata[dimension] &&
              args.proposed_changes[dimension] !== currentMetadata[dimension]) {
            clashes.push({
              dimension,
              current: currentMetadata[dimension],
              proposed: args.proposed_changes[dimension],
              severity: this.calculateClashSeverity(dimension, currentMetadata[dimension], args.proposed_changes[dimension]),
            });
          }
        }
      }
    } catch (error) {
      // Persona doesn't exist yet, no clashes
    }

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            persona_id: args.persona_id,
            has_clashes: clashes.length > 0,
            clashes,
            recommendation: clashes.length > 0 ? "resolve_before_update" : "safe_to_update",
          }, null, 2),
        },
      ],
    };
  }

  private calculateClashSeverity(dimension: string, current: any, proposed: any): string {
    // Simplified severity calculation
    if (dimension === "consciousness_state") {
      const states = ["DORMANT", "REACTIVE", "AWARE", "REFLECTIVE", "META_AWARE", "TRANSCENDENT"];
      const currentIdx = states.indexOf(current);
      const proposedIdx = states.indexOf(proposed);
      const diff = Math.abs(currentIdx - proposedIdx);
      
      if (diff > 3) return "high";
      if (diff > 1) return "medium";
      return "low";
    }
    
    return "medium";
  }

  private async swarmTopology(args: any) {
    const collection = await this.getOrCreateCollection("swarm_topologies");
    
    if (args.operation === "get") {
      const results = await collection.get();
      
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              topologies: results.ids.map((id, idx) => ({
                id,
                type: results.metadatas[idx].topology_type,
                agents: JSON.parse(results.metadatas[idx].agent_ids || "[]"),
                updated_at: results.metadatas[idx].updated_at,
              })),
            }, null, 2),
          },
        ],
      };
    } else if (args.operation === "update") {
      await collection.upsert({
        ids: [`topology_${args.topology_type}`],
        documents: [`Swarm topology: ${args.topology_type}`],
        metadatas: [{
          topology_type: args.topology_type,
          agent_ids: JSON.stringify(args.agent_ids || []),
          updated_at: new Date().toISOString(),
        }],
      });

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              status: "success",
              topology_type: args.topology_type,
              agent_count: args.agent_ids?.length || 0,
            }, null, 2),
          },
        ],
      };
    }

    throw new McpError(ErrorCode.InvalidParams, "Invalid operation");
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("ChromaDB MCP server running on stdio");
  }
}

const server = new ChromaDBMCPServer();
server.run().catch(console.error);