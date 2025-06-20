// Topological Transformations
// Implements 19 discrete transformation events for filesystem manifolds

import { DirectoryManifold, ManifoldNode, ManifoldEdge } from './manifold_operations';
import { EventEmitter } from 'events';

interface TransformationEvent {
  id: number;
  name: string;
  type: TransformationType;
  description: string;
  preservesInvariants: boolean;
  energyRequired: number;
  apply: (manifold: DirectoryManifold, params?: any) => Promise<DirectoryManifold>;
  validate: (manifold: DirectoryManifold) => boolean;
}

interface TransformationResult {
  success: boolean;
  manifold: DirectoryManifold;
  invariantsPreserved: boolean;
  energyConsumed: number;
  transformationMatrix?: number[][];
}

type TransformationType = 
  | 'manifold_expansion'
  | 'dimensional_contraction'
  | 'topological_inversion'
  | 'crystalline_phase_transition'
  | 'quantum_tunneling'
  | 'manifold_bifurcation'
  | 'dimensional_folding'
  | 'topological_knot_formation'
  | 'crystalline_fusion'
  | 'manifold_splitting'
  | 'dimensional_rotation'
  | 'topological_homeomorphism'
  | 'crystalline_sublimation'
  | 'manifold_intersection'
  | 'dimensional_projection'
  | 'topological_embedding'
  | 'crystalline_precipitation'
  | 'manifold_extrusion'
  | 'dimensional_translation';

export class TopologicalTransformationSystem extends EventEmitter {
  private transformations: Map<number, TransformationEvent>;
  private transformationHistory: TransformationResult[];
  
  constructor() {
    super();
    this.transformations = new Map();
    this.transformationHistory = [];
    this.initializeTransformations();
  }
  
  private initializeTransformations(): void {
    // Initialize all 19 discrete transformation events
    const events: TransformationEvent[] = [
      {
        id: 1,
        name: 'Manifold Expansion',
        type: 'manifold_expansion',
        description: 'Expands the manifold by increasing dimensional bounds',
        preservesInvariants: true,
        energyRequired: 0.2,
        apply: this.manifoldExpansion.bind(this),
        validate: (m) => m.dimensions.length > 0
      },
      {
        id: 2,
        name: 'Dimensional Contraction',
        type: 'dimensional_contraction',
        description: 'Contracts manifold by reducing dimensional ranges',
        preservesInvariants: true,
        energyRequired: 0.3,
        apply: this.dimensionalContraction.bind(this),
        validate: (m) => m.dimensions.some(d => d.range[1] - d.range[0] > 1)
      },
      {
        id: 3,
        name: 'Topological Inversion',
        type: 'topological_inversion',
        description: 'Inverts the manifold topology inside-out',
        preservesInvariants: true,
        energyRequired: 0.5,
        apply: this.topologicalInversion.bind(this),
        validate: (m) => m.topology !== 'projective_plane'
      },
      {
        id: 4,
        name: 'Crystalline Phase Transition',
        type: 'crystalline_phase_transition',
        description: 'Transitions between different crystalline structures',
        preservesInvariants: false,
        energyRequired: 0.8,
        apply: this.crystallinePhaseTransition.bind(this),
        validate: (m) => m.nodes.length > 8
      },
      {
        id: 5,
        name: 'Quantum Tunneling',
        type: 'quantum_tunneling',
        description: 'Allows nodes to tunnel through dimensional barriers',
        preservesInvariants: true,
        energyRequired: 0.7,
        apply: this.quantumTunneling.bind(this),
        validate: (m) => m.dimensions.length >= 3
      },
      {
        id: 6,
        name: 'Manifold Bifurcation',
        type: 'manifold_bifurcation',
        description: 'Splits manifold into two connected components',
        preservesInvariants: false,
        energyRequired: 0.9,
        apply: this.manifoldBifurcation.bind(this),
        validate: (m) => m.nodes.length >= 16
      },
      {
        id: 7,
        name: 'Dimensional Folding',
        type: 'dimensional_folding',
        description: 'Folds manifold along a dimensional axis',
        preservesInvariants: true,
        energyRequired: 0.4,
        apply: this.dimensionalFolding.bind(this),
        validate: (m) => m.dimensions.length >= 2
      },
      {
        id: 8,
        name: 'Topological Knot Formation',
        type: 'topological_knot_formation',
        description: 'Creates topological knots in the manifold structure',
        preservesInvariants: true,
        energyRequired: 0.6,
        apply: this.topologicalKnotFormation.bind(this),
        validate: (m) => m.edges.length >= 12
      },
      {
        id: 9,
        name: 'Crystalline Fusion',
        type: 'crystalline_fusion',
        description: 'Fuses multiple crystalline structures together',
        preservesInvariants: false,
        energyRequired: 0.8,
        apply: this.crystallineFusion.bind(this),
        validate: (m) => m.nodes.filter(n => n.state === 'crystallizing').length >= 2
      },
      {
        id: 10,
        name: 'Manifold Splitting',
        type: 'manifold_splitting',
        description: 'Splits manifold into independent regions',
        preservesInvariants: false,
        energyRequired: 0.7,
        apply: this.manifoldSplitting.bind(this),
        validate: (m) => m.nodes.length >= 8
      },
      {
        id: 11,
        name: 'Dimensional Rotation',
        type: 'dimensional_rotation',
        description: 'Rotates manifold in higher dimensional space',
        preservesInvariants: true,
        energyRequired: 0.3,
        apply: this.dimensionalRotation.bind(this),
        validate: (m) => m.dimensions.length >= 3
      },
      {
        id: 12,
        name: 'Topological Homeomorphism',
        type: 'topological_homeomorphism',
        description: 'Continuous deformation preserving topology',
        preservesInvariants: true,
        energyRequired: 0.4,
        apply: this.topologicalHomeomorphism.bind(this),
        validate: (m) => true
      },
      {
        id: 13,
        name: 'Crystalline Sublimation',
        type: 'crystalline_sublimation',
        description: 'Direct phase transition skipping intermediate states',
        preservesInvariants: false,
        energyRequired: 0.9,
        apply: this.crystallineSublimation.bind(this),
        validate: (m) => m.nodes.some(n => n.state === 'stable')
      },
      {
        id: 14,
        name: 'Manifold Intersection',
        type: 'manifold_intersection',
        description: 'Creates self-intersecting manifold regions',
        preservesInvariants: false,
        energyRequired: 0.6,
        apply: this.manifoldIntersection.bind(this),
        validate: (m) => m.dimensions.length >= 4
      },
      {
        id: 15,
        name: 'Dimensional Projection',
        type: 'dimensional_projection',
        description: 'Projects manifold to lower dimensional subspace',
        preservesInvariants: false,
        energyRequired: 0.5,
        apply: this.dimensionalProjection.bind(this),
        validate: (m) => m.dimensions.length > 1
      },
      {
        id: 16,
        name: 'Topological Embedding',
        type: 'topological_embedding',
        description: 'Embeds manifold in higher dimensional space',
        preservesInvariants: true,
        energyRequired: 0.4,
        apply: this.topologicalEmbedding.bind(this),
        validate: (m) => m.dimensions.length < 11
      },
      {
        id: 17,
        name: 'Crystalline Precipitation',
        type: 'crystalline_precipitation',
        description: 'Forms crystalline structures from quantum flux',
        preservesInvariants: false,
        energyRequired: 0.7,
        apply: this.crystallinePrecipitation.bind(this),
        validate: (m) => m.nodes.some(n => n.state === 'quantum_flux')
      },
      {
        id: 18,
        name: 'Manifold Extrusion',
        type: 'manifold_extrusion',
        description: 'Extrudes manifold along a dimensional vector',
        preservesInvariants: true,
        energyRequired: 0.5,
        apply: this.manifoldExtrusion.bind(this),
        validate: (m) => m.dimensions.length >= 2
      },
      {
        id: 19,
        name: 'Dimensional Translation',
        type: 'dimensional_translation',
        description: 'Translates manifold through dimensional space',
        preservesInvariants: true,
        energyRequired: 0.2,
        apply: this.dimensionalTranslation.bind(this),
        validate: (m) => true
      }
    ];
    
    events.forEach(event => {
      this.transformations.set(event.id, event);
    });
  }
  
  async applyTransformation(
    manifold: DirectoryManifold,
    transformationId: number,
    params?: any
  ): Promise<TransformationResult> {
    const transformation = this.transformations.get(transformationId);
    
    if (!transformation) {
      throw new Error(`Unknown transformation ID: ${transformationId}`);
    }
    
    if (!transformation.validate(manifold)) {
      throw new Error(`Manifold does not meet requirements for ${transformation.name}`);
    }
    
    this.emit('transformation:start', { manifold, transformation });
    
    // Store initial invariants
    const initialInvariants = this.captureInvariants(manifold);
    
    // Apply transformation
    const transformedManifold = await transformation.apply(manifold, params);
    
    // Check invariant preservation
    const finalInvariants = this.captureInvariants(transformedManifold);
    const invariantsPreserved = this.compareInvariants(initialInvariants, finalInvariants);
    
    const result: TransformationResult = {
      success: true,
      manifold: transformedManifold,
      invariantsPreserved,
      energyConsumed: transformation.energyRequired
    };
    
    this.transformationHistory.push(result);
    this.emit('transformation:complete', result);
    
    return result;
  }
  
  // Transformation implementations
  
  private async manifoldExpansion(manifold: DirectoryManifold, params?: { factor?: number }): Promise<DirectoryManifold> {
    const factor = params?.factor || 2;
    const expanded = { ...manifold };
    
    // Expand dimensional ranges
    expanded.dimensions = expanded.dimensions.map(dim => ({
      ...dim,
      range: [dim.range[0], dim.range[1] * factor]
    }));
    
    // Scale node coordinates
    expanded.nodes = expanded.nodes.map(node => ({
      ...node,
      coordinates: node.coordinates.map((coord, i) => {
        const dim = expanded.dimensions[i];
        const normalized = (coord - dim.range[0]) / (dim.range[1] - dim.range[0]);
        return dim.range[0] + normalized * (dim.range[1] * factor - dim.range[0]);
      })
    }));
    
    return expanded;
  }
  
  private async dimensionalContraction(manifold: DirectoryManifold, params?: { factor?: number }): Promise<DirectoryManifold> {
    const factor = params?.factor || 0.5;
    const contracted = { ...manifold };
    
    // Contract dimensional ranges
    contracted.dimensions = contracted.dimensions.map(dim => ({
      ...dim,
      range: [dim.range[0], dim.range[0] + (dim.range[1] - dim.range[0]) * factor]
    }));
    
    // Adjust node coordinates
    contracted.nodes = contracted.nodes.map(node => ({
      ...node,
      coordinates: node.coordinates.map((coord, i) => {
        const dim = contracted.dimensions[i];
        return Math.min(coord, dim.range[1]);
      })
    }));
    
    return contracted;
  }
  
  private async topologicalInversion(manifold: DirectoryManifold): Promise<DirectoryManifold> {
    const inverted = { ...manifold };
    
    // Invert node coordinates relative to center
    const center = inverted.dimensions.map(dim => 
      (dim.range[0] + dim.range[1]) / 2
    );
    
    inverted.nodes = inverted.nodes.map(node => ({
      ...node,
      coordinates: node.coordinates.map((coord, i) => {
        const distance = coord - center[i];
        return center[i] - distance;
      })
    }));
    
    // Reverse edge directions
    inverted.edges = inverted.edges.map(edge => ({
      ...edge,
      source: edge.target,
      target: edge.source
    }));
    
    return inverted;
  }
  
  private async crystallinePhaseTransition(manifold: DirectoryManifold, params?: { targetPhase?: string }): Promise<DirectoryManifold> {
    const transitioned = { ...manifold };
    const phases = ['cubic', 'hexagonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic'];
    const targetPhase = params?.targetPhase || phases[Math.floor(Math.random() * phases.length)];
    
    // Change topology type based on phase
    transitioned.topology = targetPhase as any;
    
    // Update node states
    transitioned.nodes = transitioned.nodes.map(node => ({
      ...node,
      state: 'transitioning',
      metadata: {
        ...node.metadata,
        phase: targetPhase
      }
    }));
    
    return transitioned;
  }
  
  private async quantumTunneling(manifold: DirectoryManifold, params?: { probability?: number }): Promise<DirectoryManifold> {
    const tunneled = { ...manifold };
    const probability = params?.probability || 0.1;
    
    // Allow random nodes to tunnel to new positions
    tunneled.nodes = tunneled.nodes.map(node => {
      if (Math.random() < probability) {
        // Quantum tunnel to new position
        const newCoordinates = node.coordinates.map((coord, i) => {
          const dim = tunneled.dimensions[i];
          const range = dim.range[1] - dim.range[0];
          const tunnel = (Math.random() - 0.5) * range * 0.5;
          
          let newCoord = coord + tunnel;
          
          // Handle periodic boundaries
          if (dim.periodic) {
            if (newCoord < dim.range[0]) newCoord += range;
            if (newCoord > dim.range[1]) newCoord -= range;
          } else {
            newCoord = Math.max(dim.range[0], Math.min(dim.range[1], newCoord));
          }
          
          return newCoord;
        });
        
        return {
          ...node,
          coordinates: newCoordinates,
          state: 'quantum_flux' as const
        };
      }
      
      return node;
    });
    
    return tunneled;
  }
  
  private async manifoldBifurcation(manifold: DirectoryManifold): Promise<DirectoryManifold> {
    const bifurcated = { ...manifold };
    
    // Find bifurcation plane
    const splitDimension = 0; // Split along first dimension
    const splitValue = (bifurcated.dimensions[splitDimension].range[0] + 
                       bifurcated.dimensions[splitDimension].range[1]) / 2;
    
    // Duplicate nodes on opposite sides
    const newNodes: ManifoldNode[] = [];
    
    bifurcated.nodes.forEach(node => {
      if (Math.abs(node.coordinates[splitDimension] - splitValue) < 0.1) {
        // Node is near bifurcation plane, duplicate it
        const duplicate = {
          ...node,
          id: `${node.id}_bifurcated`,
          coordinates: [...node.coordinates]
        };
        duplicate.coordinates[splitDimension] += 0.2;
        newNodes.push(duplicate);
      }
    });
    
    bifurcated.nodes.push(...newNodes);
    
    // Add connecting edges between bifurcated parts
    newNodes.forEach(newNode => {
      const originalId = newNode.id.replace('_bifurcated', '');
      bifurcated.edges.push({
        source: originalId,
        target: newNode.id,
        type: 'quantum_entangled',
        weight: 0.5,
        bidirectional: true
      });
    });
    
    return bifurcated;
  }
  
  private async dimensionalFolding(manifold: DirectoryManifold, params?: { foldAxis?: number }): Promise<DirectoryManifold> {
    const folded = { ...manifold };
    const foldAxis = params?.foldAxis || 0;
    const foldPoint = (folded.dimensions[foldAxis].range[0] + 
                      folded.dimensions[foldAxis].range[1]) / 2;
    
    // Fold nodes across the axis
    folded.nodes = folded.nodes.map(node => {
      const newCoordinates = [...node.coordinates];
      
      if (newCoordinates[foldAxis] > foldPoint) {
        // Fold this coordinate
        newCoordinates[foldAxis] = foldPoint - (newCoordinates[foldAxis] - foldPoint);
      }
      
      return {
        ...node,
        coordinates: newCoordinates
      };
    });
    
    return folded;
  }
  
  private async topologicalKnotFormation(manifold: DirectoryManifold): Promise<DirectoryManifold> {
    const knotted = { ...manifold };
    
    // Create a trefoil knot pattern in the edges
    const centerNode = knotted.nodes[Math.floor(knotted.nodes.length / 2)];
    const radius = 0.3;
    
    // Add knot nodes
    const knotNodes: ManifoldNode[] = [];
    for (let t = 0; t < 2 * Math.PI; t += Math.PI / 6) {
      const x = centerNode.coordinates[0] + radius * (Math.sin(t) + 2 * Math.sin(2 * t));
      const y = centerNode.coordinates[1] + radius * (Math.cos(t) - 2 * Math.cos(2 * t));
      const z = centerNode.coordinates[2] || 0 + radius * (-Math.sin(3 * t));
      
      knotNodes.push({
        id: `knot_${t}`,
        coordinates: [x, y, z],
        type: 'symlink',
        state: 'stable',
        metadata: { knot: true }
      });
    }
    
    knotted.nodes.push(...knotNodes);
    
    // Connect knot nodes
    for (let i = 0; i < knotNodes.length; i++) {
      const next = (i + 1) % knotNodes.length;
      knotted.edges.push({
        source: knotNodes[i].id,
        target: knotNodes[next].id,
        type: 'spatial',
        weight: 1,
        bidirectional: true
      });
    }
    
    return knotted;
  }
  
  private async crystallineFusion(manifold: DirectoryManifold): Promise<DirectoryManifold> {
    const fused = { ...manifold };
    
    // Find crystallizing nodes
    const crystallizingNodes = fused.nodes.filter(n => n.state === 'crystallizing');
    
    if (crystallizingNodes.length >= 2) {
      // Fuse first two crystallizing regions
      const node1 = crystallizingNodes[0];
      const node2 = crystallizingNodes[1];
      
      // Create fusion center
      const fusionCenter = {
        id: `fusion_${node1.id}_${node2.id}`,
        coordinates: node1.coordinates.map((c, i) => (c + node2.coordinates[i]) / 2),
        type: 'directory' as const,
        state: 'crystallizing' as const,
        metadata: { fusion: true }
      };
      
      fused.nodes.push(fusionCenter);
      
      // Connect to fusion center
      fused.edges.push(
        {
          source: node1.id,
          target: fusionCenter.id,
          type: 'quantum_entangled',
          weight: 2,
          bidirectional: true
        },
        {
          source: node2.id,
          target: fusionCenter.id,
          type: 'quantum_entangled',
          weight: 2,
          bidirectional: true
        }
      );
    }
    
    return fused;
  }
  
  private async manifoldSplitting(manifold: DirectoryManifold): Promise<DirectoryManifold> {
    const split = { ...manifold };
    
    // Remove some edges to create disconnected components
    const edgesToRemove = Math.floor(split.edges.length * 0.2);
    
    for (let i = 0; i < edgesToRemove; i++) {
      const randomIndex = Math.floor(Math.random() * split.edges.length);
      split.edges.splice(randomIndex, 1);
    }
    
    return split;
  }
  
  private async dimensionalRotation(manifold: DirectoryManifold, params?: { angle?: number, plane?: [number, number] }): Promise<DirectoryManifold> {
    const rotated = { ...manifold };
    const angle = params?.angle || Math.PI / 4;
    const plane = params?.plane || [0, 1];
    
    // Rotate nodes in the specified plane
    rotated.nodes = rotated.nodes.map(node => {
      const newCoordinates = [...node.coordinates];
      const x = newCoordinates[plane[0]];
      const y = newCoordinates[plane[1]];
      
      newCoordinates[plane[0]] = x * Math.cos(angle) - y * Math.sin(angle);
      newCoordinates[plane[1]] = x * Math.sin(angle) + y * Math.cos(angle);
      
      return {
        ...node,
        coordinates: newCoordinates
      };
    });
    
    return rotated;
  }
  
  private async topologicalHomeomorphism(manifold: DirectoryManifold): Promise<DirectoryManifold> {
    const deformed = { ...manifold };
    
    // Apply continuous deformation
    deformed.nodes = deformed.nodes.map(node => {
      const newCoordinates = node.coordinates.map((coord, i) => {
        const dim = deformed.dimensions[i];
        const normalized = (coord - dim.range[0]) / (dim.range[1] - dim.range[0]);
        
        // Apply smooth deformation function
        const deformed = normalized + 0.1 * Math.sin(normalized * Math.PI * 2);
        
        return dim.range[0] + deformed * (dim.range[1] - dim.range[0]);
      });
      
      return {
        ...node,
        coordinates: newCoordinates
      };
    });
    
    return deformed;
  }
  
  private async crystallineSublimation(manifold: DirectoryManifold): Promise<DirectoryManifold> {
    const sublimated = { ...manifold };
    
    // Direct transition from stable to quantum flux
    sublimated.nodes = sublimated.nodes.map(node => {
      if (node.state === 'stable') {
        return {
          ...node,
          state: 'quantum_flux',
          metadata: {
            ...node.metadata,
            sublimated: true
          }
        };
      }
      return node;
    });
    
    return sublimated;
  }
  
  private async manifoldIntersection(manifold: DirectoryManifold): Promise<DirectoryManifold> {
    const intersected = { ...manifold };
    
    // Create self-intersection by connecting distant nodes
    const nodeCount = intersected.nodes.length;
    
    for (let i = 0; i < Math.min(5, nodeCount / 4); i++) {
      const node1 = intersected.nodes[Math.floor(Math.random() * nodeCount)];
      const node2 = intersected.nodes[Math.floor(Math.random() * nodeCount)];
      
      if (node1.id !== node2.id) {
        intersected.edges.push({
          source: node1.id,
          target: node2.id,
          type: 'quantum_entangled',
          weight: 0.3,
          bidirectional: true
        });
      }
    }
    
    return intersected;
  }
  
  private async dimensionalProjection(manifold: DirectoryManifold, params?: { targetDimensions?: number }): Promise<DirectoryManifold> {
    const projected = { ...manifold };
    const targetDims = params?.targetDimensions || Math.max(1, manifold.dimensions.length - 1);
    
    // Project to lower dimensions
    projected.dimensions = projected.dimensions.slice(0, targetDims);
    projected.nodes = projected.nodes.map(node => ({
      ...node,
      coordinates: node.coordinates.slice(0, targetDims)
    }));
    
    return projected;
  }
  
  private async topologicalEmbedding(manifold: DirectoryManifold, params?: { targetDimensions?: number }): Promise<DirectoryManifold> {
    const embedded = { ...manifold };
    const targetDims = params?.targetDimensions || manifold.dimensions.length + 1;
    
    // Add new dimensions
    while (embedded.dimensions.length < targetDims) {
      embedded.dimensions.push({
        axis: `d${embedded.dimensions.length}`,
        range: [0, 1],
        periodic: false,
        connectivity: 'nearest_neighbor' as const
      });
    }
    
    // Extend node coordinates
    embedded.nodes = embedded.nodes.map(node => ({
      ...node,
      coordinates: [
        ...node.coordinates,
        ...new Array(targetDims - node.coordinates.length).fill(0.5)
      ]
    }));
    
    return embedded;
  }
  
  private async crystallinePrecipitation(manifold: DirectoryManifold): Promise<DirectoryManifold> {
    const precipitated = { ...manifold };
    
    // Convert quantum flux nodes to crystallizing
    precipitated.nodes = precipitated.nodes.map(node => {
      if (node.state === 'quantum_flux') {
        return {
          ...node,
          state: 'crystallizing',
          metadata: {
            ...node.metadata,
            precipitated: true
          }
        };
      }
      return node;
    });
    
    return precipitated;
  }
  
  private async manifoldExtrusion(manifold: DirectoryManifold, params?: { extrusionVector?: number[] }): Promise<DirectoryManifold> {
    const extruded = { ...manifold };
    const vector = params?.extrusionVector || [0, 0, 0.5];
    
    // Duplicate nodes with extrusion offset
    const extrudedNodes: ManifoldNode[] = [];
    
    extruded.nodes.forEach(node => {
      const newNode = {
        ...node,
        id: `${node.id}_extruded`,
        coordinates: node.coordinates.map((c, i) => c + (vector[i] || 0))
      };
      extrudedNodes.push(newNode);
      
      // Connect original to extruded
      extruded.edges.push({
        source: node.id,
        target: newNode.id,
        type: 'spatial',
        weight: 1,
        bidirectional: true
      });
    });
    
    extruded.nodes.push(...extrudedNodes);
    
    return extruded;
  }
  
  private async dimensionalTranslation(manifold: DirectoryManifold, params?: { translationVector?: number[] }): Promise<DirectoryManifold> {
    const translated = { ...manifold };
    const vector = params?.translationVector || manifold.dimensions.map(() => 0.1);
    
    // Translate all nodes
    translated.nodes = translated.nodes.map(node => ({
      ...node,
      coordinates: node.coordinates.map((coord, i) => {
        const dim = translated.dimensions[i];
        let newCoord = coord + (vector[i] || 0);
        
        // Handle boundaries
        if (dim.periodic) {
          const range = dim.range[1] - dim.range[0];
          if (newCoord < dim.range[0]) newCoord += range;
          if (newCoord > dim.range[1]) newCoord -= range;
        } else {
          newCoord = Math.max(dim.range[0], Math.min(dim.range[1], newCoord));
        }
        
        return newCoord;
      })
    }));
    
    return translated;
  }
  
  // Helper methods
  
  private captureInvariants(manifold: DirectoryManifold): Map<string, any> {
    const invariants = new Map<string, any>();
    
    manifold.invariants.forEach(inv => {
      invariants.set(inv.name, inv.value);
    });
    
    // Add structural invariants
    invariants.set('node_count', manifold.nodes.length);
    invariants.set('edge_count', manifold.edges.length);
    invariants.set('dimension_count', manifold.dimensions.length);
    
    return invariants;
  }
  
  private compareInvariants(initial: Map<string, any>, final: Map<string, any>): boolean {
    // Check critical invariants
    const criticalInvariants = ['euler_characteristic', 'fundamental_group'];
    
    for (const key of criticalInvariants) {
      if (initial.has(key) && final.has(key)) {
        if (initial.get(key) !== final.get(key)) {
          return false;
        }
      }
    }
    
    return true;
  }
  
  getTransformationList(): TransformationEvent[] {
    return Array.from(this.transformations.values());
  }
  
  getTransformationHistory(): TransformationResult[] {
    return [...this.transformationHistory];
  }
}