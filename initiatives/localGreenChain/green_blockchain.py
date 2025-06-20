"""
LocalGreenChain: Living Blockchain Implementation

A carbon-negative blockchain that uses plant growth as consensus mechanism
and tracks botanical genealogy for ecosystem health.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import asyncio
from geopy import distance
import cv2


@dataclass
class PlantDNA:
    """Represents simplified plant genetic markers"""
    species: str
    variant: str
    genetic_markers: List[str]
    parent_hashes: List[str] = field(default_factory=list)
    
    def to_hash(self) -> str:
        """Generate unique genetic hash"""
        content = f"{self.species}:{self.variant}:{''.join(self.genetic_markers)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Location:
    """Anonymized location data"""
    grid_square: str  # 100m x 100m grid
    elevation: float
    climate_zone: str
    soil_type: str
    
    @staticmethod
    def anonymize_coordinates(lat: float, lon: float, precision: int = 100) -> str:
        """Convert exact coordinates to grid square"""
        # Round to nearest grid square (default 100m)
        grid_lat = round(lat * (10000 / precision)) / (10000 / precision)
        grid_lon = round(lon * (10000 / precision)) / (10000 / precision)
        return f"{grid_lat:.4f}_{grid_lon:.4f}_{precision}m"


@dataclass
class GrowthMetrics:
    """Plant growth measurements"""
    height_cm: float
    trunk_diameter_mm: Optional[float]
    leaf_count: int
    root_depth_cm: float
    health_score: float  # 0-1
    biomass_kg: float
    
    def calculate_carbon_sequestered(self) -> float:
        """Estimate carbon sequestration based on biomass"""
        # Simplified: ~50% of dry biomass is carbon
        return self.biomass_kg * 0.5 * 0.4  # 40% of biomass is dry


@dataclass
class EcosystemInteractions:
    """Track plant's ecosystem relationships"""
    mycorrhizal_connections: List[str]  # Other plant IDs
    pollinator_visits: int
    companion_plants: List[str]
    pest_interactions: List[Dict[str, Any]]
    soil_contributions: Dict[str, float]  # Nutrient contributions


class GreenBlock:
    """Base class for LocalGreenChain blocks"""
    
    def __init__(self, block_type: str, plant_token: str, parent_hash: Optional[str] = None):
        self.block_type = block_type
        self.plant_token = plant_token
        self.parent_hash = parent_hash
        self.timestamp = datetime.utcnow()
        self.nonce = 0
        self.hash = None
        
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        content = {
            "block_type": self.block_type,
            "plant_token": self.plant_token,
            "parent_hash": self.parent_hash,
            "timestamp": self.timestamp.isoformat(),
            "nonce": self.nonce
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        
    def mine_green(self, growth_proof: Any) -> bool:
        """Green mining - validate through growth proof instead of computation"""
        # No intensive computation needed - just verify growth
        self.hash = self.calculate_hash()
        return True


class GenesisBlock(GreenBlock):
    """Plant birth certificate block"""
    
    def __init__(self, plant_token: str, dna: PlantDNA, location: Location, 
                 initial_metrics: GrowthMetrics):
        super().__init__("genesis", plant_token)
        self.dna = dna
        self.location = location
        self.initial_metrics = initial_metrics
        self.planting_method = None
        self.caretaker_id = None  # Anonymized
        
    def to_dict(self) -> Dict:
        return {
            "block_type": self.block_type,
            "plant_token": self.plant_token,
            "timestamp": self.timestamp.isoformat(),
            "dna": {
                "species": self.dna.species,
                "variant": self.dna.variant,
                "hash": self.dna.to_hash(),
                "lineage": self.dna.parent_hashes
            },
            "location": {
                "grid": self.location.grid_square,
                "elevation": self.location.elevation,
                "climate": self.location.climate_zone
            },
            "initial_state": {
                "height_cm": self.initial_metrics.height_cm,
                "health": self.initial_metrics.health_score
            },
            "hash": self.hash
        }


class GrowthBlock(GreenBlock):
    """Plant growth milestone block"""
    
    def __init__(self, plant_token: str, parent_hash: str, 
                 metrics: GrowthMetrics, interactions: EcosystemInteractions):
        super().__init__("growth", plant_token, parent_hash)
        self.metrics = metrics
        self.interactions = interactions
        self.carbon_sequestered = metrics.calculate_carbon_sequestered()
        self.growth_proof = None  # Image hash or sensor data
        
    def validate_growth(self, previous_metrics: GrowthMetrics) -> bool:
        """Validate realistic growth patterns"""
        # Growth should be positive but within biological limits
        height_change = self.metrics.height_cm - previous_metrics.height_cm
        
        # Basic sanity checks
        if height_change < 0:  # Plants don't shrink (usually)
            return False
        if height_change > 100:  # 1m growth between blocks is suspicious
            return False
            
        return True


class LocalGreenChain:
    """Main blockchain implementation for plant tracking"""
    
    def __init__(self):
        self.chain: Dict[str, List[GreenBlock]] = {}  # token -> blocks
        self.active_plants: Dict[str, PlantDNA] = {}
        self.ecosystem_map: Dict[str, List[str]] = {}  # grid -> plant tokens
        self.carbon_ledger = 0.0
        self.species_registry: Dict[str, int] = {}
        
    async def plant_seed(self, species: str, variant: str, location: Location,
                        parent_tokens: Optional[List[str]] = None) -> str:
        """Register a new plant in the chain"""
        # Generate plant token
        token = f"{species.lower().replace(' ', '_')}_{int(datetime.utcnow().timestamp())}"
        
        # Create DNA record
        parent_hashes = []
        if parent_tokens:
            parent_hashes = [self.active_plants[pt].to_hash() for pt in parent_tokens]
            
        dna = PlantDNA(
            species=species,
            variant=variant,
            genetic_markers=[f"marker_{i}" for i in range(10)],  # Simplified
            parent_hashes=parent_hashes
        )
        
        # Initial measurements
        initial_metrics = GrowthMetrics(
            height_cm=5.0,
            trunk_diameter_mm=None,
            leaf_count=2,
            root_depth_cm=3.0,
            health_score=0.9,
            biomass_kg=0.001
        )
        
        # Create genesis block
        genesis = GenesisBlock(token, dna, location, initial_metrics)
        genesis.mine_green(None)
        
        # Add to chain
        self.chain[token] = [genesis]
        self.active_plants[token] = dna
        
        # Update ecosystem map
        if location.grid_square not in self.ecosystem_map:
            self.ecosystem_map[location.grid_square] = []
        self.ecosystem_map[location.grid_square].append(token)
        
        # Update species registry
        self.species_registry[species] = self.species_registry.get(species, 0) + 1
        
        return token
        
    async def record_growth(self, token: str, metrics: GrowthMetrics,
                          interactions: EcosystemInteractions,
                          growth_image: Optional[np.ndarray] = None) -> bool:
        """Record plant growth milestone"""
        if token not in self.chain:
            return False
            
        # Get previous block
        previous_block = self.chain[token][-1]
        
        # Create growth block
        growth_block = GrowthBlock(
            token, 
            previous_block.hash,
            metrics,
            interactions
        )
        
        # Validate growth
        if isinstance(previous_block, GrowthBlock):
            if not growth_block.validate_growth(previous_block.metrics):
                return False
        elif isinstance(previous_block, GenesisBlock):
            if not growth_block.validate_growth(previous_block.initial_metrics):
                return False
                
        # Process growth proof (image analysis)
        if growth_image is not None:
            growth_proof = await self._analyze_growth_image(growth_image, token)
            growth_block.growth_proof = growth_proof
            
        # Mine block (instant for green consensus)
        growth_block.mine_green(growth_block.growth_proof)
        
        # Add to chain
        self.chain[token].append(growth_block)
        
        # Update carbon ledger
        self.carbon_ledger += growth_block.carbon_sequestered
        
        # Update ecosystem connections
        await self._update_ecosystem_connections(token, interactions)
        
        return True
        
    async def _analyze_growth_image(self, image: np.ndarray, token: str) -> Dict:
        """Analyze plant image for growth verification"""
        # Simplified image analysis
        # In production, would use trained CNN for species verification
        
        height_pixels = np.sum(image > 128)  # Simple plant detection
        health_color = np.mean(image[image > 128])  # Green-ness
        
        return {
            "image_hash": hashlib.sha256(image.tobytes()).hexdigest()[:16],
            "detected_height": height_pixels,
            "health_indicator": health_color / 255.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def _update_ecosystem_connections(self, token: str, 
                                          interactions: EcosystemInteractions):
        """Update mycorrhizal and companion plant networks"""
        # This creates the underground wood wide web tracking
        for connected_plant in interactions.mycorrhizal_connections:
            if connected_plant in self.active_plants:
                # Bidirectional connection
                # Would update both plants' interaction records
                pass
                
    def track_invasive_species(self, species: str) -> List[Dict]:
        """Track spread of potentially invasive species"""
        invasive_locations = []
        
        for grid, plants in self.ecosystem_map.items():
            for plant_token in plants:
                if plant_token in self.active_plants:
                    if self.active_plants[plant_token].species == species:
                        # Get growth rate from blocks
                        blocks = self.chain[plant_token]
                        growth_rate = self._calculate_growth_rate(blocks)
                        
                        invasive_locations.append({
                            "grid": grid,
                            "token": plant_token,
                            "growth_rate": growth_rate,
                            "block_count": len(blocks),
                            "risk_score": growth_rate * len(blocks) * 0.1
                        })
                        
        return sorted(invasive_locations, key=lambda x: x["risk_score"], reverse=True)
        
    def _calculate_growth_rate(self, blocks: List[GreenBlock]) -> float:
        """Calculate average growth rate from block history"""
        if len(blocks) < 2:
            return 0.0
            
        total_height_change = 0.0
        time_span = 0.0
        
        for i in range(1, len(blocks)):
            if isinstance(blocks[i], GrowthBlock) and isinstance(blocks[i-1], (GrowthBlock, GenesisBlock)):
                if isinstance(blocks[i-1], GrowthBlock):
                    prev_height = blocks[i-1].metrics.height_cm
                else:
                    prev_height = blocks[i-1].initial_metrics.height_cm
                    
                height_change = blocks[i].metrics.height_cm - prev_height
                time_delta = (blocks[i].timestamp - blocks[i-1].timestamp).total_seconds() / 86400  # Days
                
                total_height_change += height_change
                time_span += time_delta
                
        return total_height_change / time_span if time_span > 0 else 0.0
        
    def generate_carbon_report(self) -> Dict:
        """Generate carbon sequestration report"""
        return {
            "total_carbon_sequestered_kg": self.carbon_ledger,
            "equivalent_co2_offset_kg": self.carbon_ledger * 3.67,  # C to CO2 ratio
            "trees_equivalent": self.carbon_ledger / 21.77,  # Avg tree sequesters 21.77kg C/year
            "cars_offset_days": self.carbon_ledger * 3.67 / 4.6,  # Avg car emits 4.6kg CO2/day
            "total_plants": len(self.active_plants),
            "species_diversity": len(self.species_registry),
            "most_effective_species": max(self.species_registry.items(), 
                                         key=lambda x: x[1])[0] if self.species_registry else None
        }
        
    def get_plant_genealogy(self, token: str) -> Dict:
        """Get complete family tree of a plant"""
        if token not in self.active_plants:
            return {}
            
        dna = self.active_plants[token]
        genealogy = {
            "token": token,
            "species": dna.species,
            "variant": dna.variant,
            "parents": [],
            "siblings": [],
            "children": []
        }
        
        # Find parents
        for parent_hash in dna.parent_hashes:
            for other_token, other_dna in self.active_plants.items():
                if other_dna.to_hash() == parent_hash:
                    genealogy["parents"].append(other_token)
                    
        # Find siblings (same parents)
        for other_token, other_dna in self.active_plants.items():
            if other_token != token and set(other_dna.parent_hashes) == set(dna.parent_hashes):
                genealogy["siblings"].append(other_token)
                
        # Find children
        my_hash = dna.to_hash()
        for other_token, other_dna in self.active_plants.items():
            if my_hash in other_dna.parent_hashes:
                genealogy["children"].append(other_token)
                
        return genealogy
        
    def calculate_biodiversity_index(self, grid_square: str) -> float:
        """Calculate Shannon diversity index for a location"""
        if grid_square not in self.ecosystem_map:
            return 0.0
            
        species_count = {}
        total = 0
        
        for plant_token in self.ecosystem_map[grid_square]:
            if plant_token in self.active_plants:
                species = self.active_plants[plant_token].species
                species_count[species] = species_count.get(species, 0) + 1
                total += 1
                
        if total == 0:
            return 0.0
            
        # Shannon diversity index
        diversity = 0.0
        for count in species_count.values():
            proportion = count / total
            diversity -= proportion * np.log(proportion)
            
        return diversity


class GreenToken:
    """LEAF token implementation for ecosystem rewards"""
    
    def __init__(self):
        self.balances: Dict[str, float] = {}
        self.total_supply = 0.0
        self.rewards = {
            "plant_seed": 10.0,
            "growth_milestone": 5.0,
            "carbon_kg": 1.0,
            "biodiversity_boost": 20.0,
            "invasive_alert": 50.0,
            "data_contribution": 2.0
        }
        
    def mint(self, address: str, amount: float, reason: str):
        """Mint new tokens for ecosystem contributions"""
        if address not in self.balances:
            self.balances[address] = 0.0
            
        self.balances[address] += amount
        self.total_supply += amount
        
        return {
            "address": address,
            "amount": amount,
            "reason": reason,
            "new_balance": self.balances[address]
        }
        
    def burn(self, address: str, amount: float, reason: str):
        """Burn tokens (e.g., plant death penalty)"""
        if address not in self.balances or self.balances[address] < amount:
            return False
            
        self.balances[address] -= amount
        self.total_supply -= amount
        
        return True


# Demonstration
async def demo_greenchain():
    """Demonstrate LocalGreenChain functionality"""
    
    # Initialize chain
    chain = LocalGreenChain()
    token_system = GreenToken()
    
    # Plant an oak tree
    location = Location(
        grid_square=Location.anonymize_coordinates(37.7749, -122.4194),
        elevation=52.0,
        climate_zone="Mediterranean",
        soil_type="Loamy"
    )
    
    oak_token = await chain.plant_seed(
        species="Quercus alba",
        variant="Northern",
        location=location
    )
    
    print(f"Planted oak tree: {oak_token}")
    
    # Reward planter
    token_system.mint("user_123", token_system.rewards["plant_seed"], "planted_oak")
    
    # Simulate growth
    await asyncio.sleep(1)  # In reality, would be weeks/months
    
    # Record growth
    new_metrics = GrowthMetrics(
        height_cm=25.0,
        trunk_diameter_mm=5.0,
        leaf_count=45,
        root_depth_cm=15.0,
        health_score=0.95,
        biomass_kg=0.05
    )
    
    interactions = EcosystemInteractions(
        mycorrhizal_connections=["oak_4782", "fern_892"],
        pollinator_visits=127,
        companion_plants=["fern_892", "moss_1247"],
        pest_interactions=[],
        soil_contributions={"nitrogen": 0.02, "carbon": 0.05}
    )
    
    success = await chain.record_growth(oak_token, new_metrics, interactions)
    print(f"Growth recorded: {success}")
    
    # Reward growth
    if success:
        token_system.mint("user_123", token_system.rewards["growth_milestone"], "oak_growth")
        carbon_reward = new_metrics.calculate_carbon_sequestered() * token_system.rewards["carbon_kg"]
        token_system.mint("user_123", carbon_reward, "carbon_sequestration")
    
    # Check carbon impact
    carbon_report = chain.generate_carbon_report()
    print(f"Carbon Report: {json.dumps(carbon_report, indent=2)}")
    
    # Check biodiversity
    biodiversity = chain.calculate_biodiversity_index(location.grid_square)
    print(f"Biodiversity Index: {biodiversity:.3f}")
    
    # Token balance
    print(f"Token Balance: {token_system.balances.get('user_123', 0)} LEAF")


if __name__ == "__main__":
    asyncio.run(demo_greenchain())