"""
Ecosystem Intelligence Layer for LocalGreenChain

Uses NPCPU's distributed consciousness to understand and optimize
plant ecosystems at scale.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN
import asyncio
from datetime import datetime, timedelta


@dataclass
class MycelialNetwork:
    """Represents underground fungal networks connecting plants"""
    network_id: str
    connected_plants: Set[str]
    nutrient_flow: Dict[Tuple[str, str], float]  # (from, to) -> flow rate
    health_score: float
    dominant_species: List[str]
    
    def calculate_network_resilience(self) -> float:
        """Calculate how resilient the network is to plant loss"""
        if len(self.connected_plants) < 2:
            return 0.0
            
        # Create network graph
        G = nx.Graph()
        for (plant1, plant2), flow in self.nutrient_flow.items():
            G.add_edge(plant1, plant2, weight=flow)
            
        # Calculate resilience metrics
        connectivity = nx.node_connectivity(G)
        avg_degree = sum(dict(G.degree()).values()) / len(G)
        
        resilience = (connectivity * 0.5 + avg_degree * 0.1) * self.health_score
        return min(resilience, 1.0)


class EcosystemIntelligence:
    """NPCPU-powered ecosystem analysis and optimization"""
    
    def __init__(self, greenchain):
        self.greenchain = greenchain
        self.mycelial_networks: Dict[str, MycelialNetwork] = {}
        self.species_interactions = nx.DiGraph()  # Directed graph of species relationships
        self.pollinator_networks = nx.Graph()
        self.invasion_predictions = {}
        
    async def analyze_mycelial_connections(self, grid_square: str) -> MycelialNetwork:
        """Analyze underground fungal networks in an area"""
        if grid_square not in self.greenchain.ecosystem_map:
            return None
            
        plants = self.greenchain.ecosystem_map[grid_square]
        network_id = f"mycelial_{grid_square}_{int(datetime.now().timestamp())}"
        
        # Build connection map from plant interactions
        connected_plants = set()
        nutrient_flow = {}
        
        for plant_token in plants:
            if plant_token not in self.greenchain.chain:
                continue
                
            # Get latest growth block
            blocks = self.greenchain.chain[plant_token]
            for block in reversed(blocks):
                if hasattr(block, 'interactions'):
                    connected_plants.add(plant_token)
                    
                    # Map nutrient flows
                    for connected in block.interactions.mycorrhizal_connections:
                        if connected in plants:
                            # Estimate nutrient flow based on plant sizes
                            flow = self._estimate_nutrient_flow(plant_token, connected)
                            nutrient_flow[(plant_token, connected)] = flow
                    break
                    
        # Calculate network health
        health_score = len(connected_plants) / max(len(plants), 1)
        
        # Find dominant species
        species_count = {}
        for plant in connected_plants:
            if plant in self.greenchain.active_plants:
                species = self.greenchain.active_plants[plant].species
                species_count[species] = species_count.get(species, 0) + 1
                
        dominant_species = sorted(species_count.items(), key=lambda x: x[1], reverse=True)
        dominant_species = [species for species, _ in dominant_species[:3]]
        
        network = MycelialNetwork(
            network_id=network_id,
            connected_plants=connected_plants,
            nutrient_flow=nutrient_flow,
            health_score=health_score,
            dominant_species=dominant_species
        )
        
        self.mycelial_networks[network_id] = network
        return network
        
    def _estimate_nutrient_flow(self, plant1: str, plant2: str) -> float:
        """Estimate nutrient flow between plants"""
        # Get plant sizes
        size1 = self._get_plant_biomass(plant1)
        size2 = self._get_plant_biomass(plant2)
        
        # Larger plants typically share more with smaller ones
        size_diff = abs(size1 - size2)
        flow = size_diff * 0.1  # Simplified model
        
        return min(flow, 1.0)
        
    def _get_plant_biomass(self, plant_token: str) -> float:
        """Get current biomass of a plant"""
        if plant_token not in self.greenchain.chain:
            return 0.0
            
        blocks = self.greenchain.chain[plant_token]
        for block in reversed(blocks):
            if hasattr(block, 'metrics'):
                return block.metrics.biomass_kg
            elif hasattr(block, 'initial_metrics'):
                return block.initial_metrics.biomass_kg
                
        return 0.0
        
    async def predict_invasive_spread(self, species: str, 
                                    time_horizon_days: int = 365) -> Dict[str, float]:
        """Predict spread of potentially invasive species"""
        current_locations = self.greenchain.track_invasive_species(species)
        
        if not current_locations:
            return {}
            
        # Build spatial model
        coordinates = []
        growth_rates = []
        
        for location in current_locations:
            # Convert grid to approximate coordinates
            grid_parts = location['grid'].split('_')
            lat = float(grid_parts[0])
            lon = float(grid_parts[1])
            coordinates.append([lat, lon])
            growth_rates.append(location['growth_rate'])
            
        coordinates = np.array(coordinates)
        growth_rates = np.array(growth_rates)
        
        # Predict spread using cellular automaton model
        predictions = {}
        
        # Find neighboring grids
        for coord, growth_rate in zip(coordinates, growth_rates):
            # Calculate spread radius based on growth rate
            daily_spread_m = growth_rate * 0.1  # 10cm per unit growth rate
            total_spread_m = daily_spread_m * time_horizon_days
            
            # Find all grids within spread radius
            spread_grids = self._find_grids_within_radius(
                coord[0], coord[1], total_spread_m
            )
            
            for grid in spread_grids:
                # Calculate invasion probability
                distance_m = self._calculate_grid_distance(coord, grid)
                prob = np.exp(-distance_m / total_spread_m)
                
                if grid in predictions:
                    predictions[grid] = max(predictions[grid], prob)
                else:
                    predictions[grid] = prob
                    
        # Filter low probability predictions
        predictions = {k: v for k, v in predictions.items() if v > 0.1}
        
        self.invasion_predictions[species] = {
            "predictions": predictions,
            "time_horizon": time_horizon_days,
            "calculated_at": datetime.now()
        }
        
        return predictions
        
    def _find_grids_within_radius(self, lat: float, lon: float, 
                                 radius_m: float) -> List[str]:
        """Find all grid squares within radius"""
        grids = []
        
        # Convert radius to approximate degrees
        radius_deg = radius_m / 111000  # Rough conversion
        
        # Check grid squares in bounding box
        for dlat in np.arange(-radius_deg, radius_deg, 0.001):  # 100m grids
            for dlon in np.arange(-radius_deg, radius_deg, 0.001):
                check_lat = lat + dlat
                check_lon = lon + dlon
                
                # Calculate actual distance
                dist_m = self._haversine_distance(lat, lon, check_lat, check_lon)
                
                if dist_m <= radius_m:
                    grid = f"{check_lat:.4f}_{check_lon:.4f}_100m"
                    grids.append(grid)
                    
        return grids
        
    def _calculate_grid_distance(self, coord: List[float], grid: str) -> float:
        """Calculate distance between coordinate and grid center"""
        grid_parts = grid.split('_')
        grid_lat = float(grid_parts[0])
        grid_lon = float(grid_parts[1])
        
        return self._haversine_distance(coord[0], coord[1], grid_lat, grid_lon)
        
    def _haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters"""
        R = 6371000  # Earth radius in meters
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
            np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
        
    async def optimize_companion_planting(self, grid_square: str) -> List[Dict]:
        """Suggest optimal companion plants for the area"""
        if grid_square not in self.greenchain.ecosystem_map:
            return []
            
        current_plants = self.greenchain.ecosystem_map[grid_square]
        current_species = set()
        
        for plant in current_plants:
            if plant in self.greenchain.active_plants:
                current_species.add(self.greenchain.active_plants[plant].species)
                
        # Build species interaction knowledge
        beneficial_pairs = {
            ("Solanum lycopersicum", "Ocimum basilicum"),  # Tomato + Basil
            ("Zea mays", "Phaseolus vulgaris"),  # Corn + Beans
            ("Brassica oleracea", "Allium sativum"),  # Cabbage + Garlic
            ("Rosa", "Allium sativum"),  # Roses + Garlic
            ("Cucurbita", "Tropaeolum majus"),  # Squash + Nasturtium
        }
        
        suggestions = []
        
        for species in current_species:
            for pair in beneficial_pairs:
                if species in pair:
                    companion = pair[0] if pair[1] == species else pair[1]
                    if companion not in current_species:
                        suggestions.append({
                            "current_species": species,
                            "suggested_companion": companion,
                            "benefit": "mutual_growth_enhancement",
                            "confidence": 0.85
                        })
                        
        # Add biodiversity suggestions
        if len(current_species) < 5:
            suggestions.append({
                "current_species": "all",
                "suggested_companion": "native_wildflowers",
                "benefit": "increased_biodiversity",
                "confidence": 0.9
            })
            
        return suggestions
        
    async def calculate_ecosystem_health(self, grid_square: str) -> Dict[str, float]:
        """Calculate comprehensive ecosystem health metrics"""
        if grid_square not in self.greenchain.ecosystem_map:
            return {"overall_health": 0.0}
            
        # Get mycelial network health
        mycelial_network = await self.analyze_mycelial_connections(grid_square)
        mycelial_health = mycelial_network.calculate_network_resilience() if mycelial_network else 0.0
        
        # Calculate biodiversity
        biodiversity = self.greenchain.calculate_biodiversity_index(grid_square)
        
        # Calculate carbon sequestration rate
        carbon_rate = self._calculate_area_carbon_rate(grid_square)
        
        # Check for invasive species
        invasive_risk = 0.0
        for plant in self.greenchain.ecosystem_map[grid_square]:
            if plant in self.greenchain.active_plants:
                species = self.greenchain.active_plants[plant].species
                if species in self.invasion_predictions:
                    invasive_risk = max(invasive_risk, 0.5)  # Presence of invasive
                    
        # Calculate overall health
        overall_health = (
            mycelial_health * 0.3 +
            biodiversity * 0.3 +
            carbon_rate * 0.2 +
            (1 - invasive_risk) * 0.2
        )
        
        return {
            "overall_health": overall_health,
            "mycelial_health": mycelial_health,
            "biodiversity": biodiversity,
            "carbon_efficiency": carbon_rate,
            "invasive_risk": invasive_risk,
            "recommendations": self._generate_health_recommendations(
                overall_health, mycelial_health, biodiversity, invasive_risk
            )
        }
        
    def _calculate_area_carbon_rate(self, grid_square: str) -> float:
        """Calculate carbon sequestration rate for an area"""
        if grid_square not in self.greenchain.ecosystem_map:
            return 0.0
            
        total_carbon = 0.0
        plant_count = 0
        
        for plant in self.greenchain.ecosystem_map[grid_square]:
            if plant in self.greenchain.chain:
                blocks = self.greenchain.chain[plant]
                if len(blocks) > 1:
                    # Get carbon from latest block
                    for block in reversed(blocks):
                        if hasattr(block, 'carbon_sequestered'):
                            total_carbon += block.carbon_sequestered
                            plant_count += 1
                            break
                            
        # Normalize by area (100m²) and plant density
        area_carbon_rate = (total_carbon / 100) * (plant_count / 10)  # per m² adjusted
        return min(area_carbon_rate, 1.0)
        
    def _generate_health_recommendations(self, overall: float, mycelial: float,
                                       biodiversity: float, invasive: float) -> List[str]:
        """Generate actionable recommendations based on health metrics"""
        recommendations = []
        
        if overall < 0.5:
            recommendations.append("Critical: Ecosystem needs immediate intervention")
            
        if mycelial < 0.3:
            recommendations.append("Plant more perennials to strengthen fungal networks")
            
        if biodiversity < 0.5:
            recommendations.append("Increase species diversity with native plants")
            
        if invasive > 0.3:
            recommendations.append("Monitor and control invasive species spread")
            
        if overall > 0.8:
            recommendations.append("Excellent ecosystem health - maintain current practices")
            
        return recommendations
        
    async def generate_planting_schedule(self, grid_square: str, 
                                       season: str = "spring") -> List[Dict]:
        """Generate optimal planting schedule based on ecosystem needs"""
        health_metrics = await self.calculate_ecosystem_health(grid_square)
        companion_suggestions = await self.optimize_companion_planting(grid_square)
        
        schedule = []
        
        # Priority 1: Address health issues
        if health_metrics["biodiversity"] < 0.5:
            schedule.append({
                "week": 1,
                "action": "plant_native_wildflowers",
                "species": ["Echinacea", "Rudbeckia", "Solidago"],
                "reason": "increase_biodiversity",
                "quantity": 20
            })
            
        # Priority 2: Strengthen mycelial networks
        if health_metrics["mycelial_health"] < 0.5:
            schedule.append({
                "week": 2,
                "action": "plant_fungal_hosts",
                "species": ["Quercus", "Acer", "Pinus"],
                "reason": "strengthen_mycelial_network",
                "quantity": 3
            })
            
        # Priority 3: Companion planting
        for i, suggestion in enumerate(companion_suggestions[:3]):
            schedule.append({
                "week": 3 + i,
                "action": "companion_planting",
                "species": [suggestion["suggested_companion"]],
                "reason": suggestion["benefit"],
                "quantity": 5
            })
            
        # Priority 4: Carbon optimization
        if health_metrics["carbon_efficiency"] < 0.5:
            schedule.append({
                "week": 6,
                "action": "plant_fast_growers",
                "species": ["Populus", "Salix", "Bambusa"],
                "reason": "increase_carbon_sequestration",
                "quantity": 5
            })
            
        return schedule


class CollectiveGardenIntelligence:
    """Aggregates intelligence from multiple gardens globally"""
    
    def __init__(self):
        self.garden_networks: Dict[str, EcosystemIntelligence] = {}
        self.global_patterns = {}
        self.climate_adaptations = {}
        
    async def learn_from_garden(self, location: str, 
                              ecosystem: EcosystemIntelligence):
        """Learn patterns from a specific garden"""
        self.garden_networks[location] = ecosystem
        
        # Extract successful patterns
        health_metrics = {}
        for grid in ecosystem.greenchain.ecosystem_map:
            health = await ecosystem.calculate_ecosystem_health(grid)
            if health["overall_health"] > 0.8:
                # This is a successful ecosystem
                species_list = []
                for plant in ecosystem.greenchain.ecosystem_map[grid]:
                    if plant in ecosystem.greenchain.active_plants:
                        species_list.append(
                            ecosystem.greenchain.active_plants[plant].species
                        )
                        
                pattern_key = tuple(sorted(species_list))
                if pattern_key not in self.global_patterns:
                    self.global_patterns[pattern_key] = {
                        "locations": [],
                        "avg_health": 0.0,
                        "occurrences": 0
                    }
                    
                self.global_patterns[pattern_key]["locations"].append(location)
                self.global_patterns[pattern_key]["occurrences"] += 1
                self.global_patterns[pattern_key]["avg_health"] = (
                    (self.global_patterns[pattern_key]["avg_health"] * 
                     (self.global_patterns[pattern_key]["occurrences"] - 1) +
                     health["overall_health"]) / 
                    self.global_patterns[pattern_key]["occurrences"]
                )
                
    def recommend_for_climate(self, climate_zone: str) -> List[Tuple[str, ...]]:
        """Recommend plant combinations for a climate zone"""
        climate_patterns = []
        
        for pattern, data in self.global_patterns.items():
            # Check if pattern succeeds in similar climates
            # This is simplified - would use actual climate data
            if data["avg_health"] > 0.8 and data["occurrences"] > 3:
                climate_patterns.append(pattern)
                
        # Sort by success rate
        climate_patterns.sort(
            key=lambda p: self.global_patterns[p]["avg_health"], 
            reverse=True
        )
        
        return climate_patterns[:5]  # Top 5 combinations


# Example usage
async def demo_ecosystem_intelligence():
    """Demonstrate ecosystem intelligence capabilities"""
    from green_blockchain import LocalGreenChain, Location
    
    # Initialize systems
    chain = LocalGreenChain()
    eco_intel = EcosystemIntelligence(chain)
    
    # Create a sample ecosystem
    location = Location(
        grid_square="37.7749_-122.4194_100m",
        elevation=52.0,
        climate_zone="Mediterranean",
        soil_type="Loamy"
    )
    
    # Plant diverse species
    species = [
        ("Quercus alba", "oak"),
        ("Acer rubrum", "maple"),
        ("Pinus strobus", "pine"),
        ("Echinacea purpurea", "coneflower"),
        ("Rudbeckia hirta", "blackeyed_susan")
    ]
    
    tokens = []
    for sp_name, sp_var in species:
        token = await chain.plant_seed(sp_name, sp_var, location)
        tokens.append(token)
        
    # Analyze ecosystem
    mycelial = await eco_intel.analyze_mycelial_connections(location.grid_square)
    print(f"Mycelial Network Resilience: {mycelial.calculate_network_resilience():.2f}")
    
    # Get health report
    health = await eco_intel.calculate_ecosystem_health(location.grid_square)
    print(f"Ecosystem Health Report: {json.dumps(health, indent=2)}")
    
    # Get planting schedule
    schedule = await eco_intel.generate_planting_schedule(location.grid_square)
    print(f"Recommended Planting Schedule:")
    for item in schedule:
        print(f"  Week {item['week']}: {item['action']} - {item['species']}")


if __name__ == "__main__":
    import json
    asyncio.run(demo_ecosystem_intelligence())