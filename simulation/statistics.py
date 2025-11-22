"""
Simulation Statistics and Visualization

Tools for analyzing and visualizing simulation results:
- Statistical analysis of population dynamics
- Fitness evolution tracking
- Trait distribution analysis
- Export to various formats (CSV, JSON, HTML reports)
- Optional matplotlib visualizations
"""

import json
import csv
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class PopulationStats:
    """Statistics for population analysis"""
    min_population: int
    max_population: int
    mean_population: float
    std_population: float
    final_population: int
    growth_rate: float  # Average change per tick


@dataclass
class FitnessStats:
    """Statistics for fitness analysis"""
    min_fitness: float
    max_fitness: float
    mean_fitness: float
    std_fitness: float
    final_fitness: float
    improvement_rate: float  # Average change per generation


@dataclass
class TraitAnalysis:
    """Analysis of trait distributions"""
    trait_name: str
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    trend: str  # "increasing", "decreasing", "stable"


@dataclass
class SimulationReport:
    """Complete simulation analysis report"""
    simulation_name: str
    simulation_id: str
    analysis_time: str
    duration_ticks: int
    duration_seconds: float

    # Population stats
    population_stats: PopulationStats

    # Fitness stats
    fitness_stats: FitnessStats

    # Trait analysis
    trait_analyses: List[TraitAnalysis]

    # Event summary
    total_births: int
    total_deaths: int
    survival_rate: float
    world_events: int

    # Generation summary
    generations_recorded: int
    final_diversity: float


class SimulationAnalyzer:
    """
    Analyzes simulation results and generates reports.

    Example:
        # After running simulation
        analyzer = SimulationAnalyzer(simulation)
        report = analyzer.analyze()

        # Export results
        analyzer.export_csv("results.csv")
        analyzer.export_json("results.json")
        analyzer.generate_html_report("report.html")

        # Create visualizations
        analyzer.plot_population()
        analyzer.plot_fitness()
    """

    def __init__(self, simulation=None, results: Dict[str, Any] = None):
        """
        Initialize analyzer with simulation or results.

        Args:
            simulation: A Simulation instance
            results: Results dict (from sim.get_results())
        """
        self.simulation = simulation
        self.results = results or (simulation.get_results() if simulation else {})
        self.report: Optional[SimulationReport] = None

    @classmethod
    def from_file(cls, filepath: str) -> 'SimulationAnalyzer':
        """Load results from a saved file"""
        with open(filepath, 'r') as f:
            results = json.load(f)
        return cls(results=results)

    def analyze(self) -> SimulationReport:
        """Perform complete analysis and generate report"""
        if not self.results:
            raise ValueError("No simulation results to analyze")

        # Analyze population
        pop_stats = self._analyze_population()

        # Analyze fitness
        fitness_stats = self._analyze_fitness()

        # Analyze traits
        trait_analyses = self._analyze_traits()

        # Build report
        self.report = SimulationReport(
            simulation_name=self.results.get("config", {}).get("name", "Unknown"),
            simulation_id=self.results.get("simulation_id", "unknown"),
            analysis_time=datetime.now().isoformat(),
            duration_ticks=self.results.get("metrics", {}).get("total_ticks", 0),
            duration_seconds=self.results.get("duration", 0),
            population_stats=pop_stats,
            fitness_stats=fitness_stats,
            trait_analyses=trait_analyses,
            total_births=self.results.get("metrics", {}).get("total_organisms_created", 0),
            total_deaths=self.results.get("metrics", {}).get("total_deaths", 0),
            survival_rate=self._calculate_survival_rate(),
            world_events=self.results.get("metrics", {}).get("world_events", 0),
            generations_recorded=len(self.results.get("generations", [])),
            final_diversity=self.results.get("metrics", {}).get("genetic_diversity", 0)
        )

        return self.report

    def _analyze_population(self) -> PopulationStats:
        """Analyze population dynamics"""
        history = self.results.get("population_history", [])

        if not history:
            return PopulationStats(0, 0, 0, 0, 0, 0)

        if HAS_NUMPY:
            arr = np.array(history)
            return PopulationStats(
                min_population=int(arr.min()),
                max_population=int(arr.max()),
                mean_population=float(arr.mean()),
                std_population=float(arr.std()),
                final_population=history[-1],
                growth_rate=float(np.mean(np.diff(arr))) if len(arr) > 1 else 0
            )
        else:
            # Pure Python fallback
            return PopulationStats(
                min_population=min(history),
                max_population=max(history),
                mean_population=sum(history) / len(history),
                std_population=self._std(history),
                final_population=history[-1],
                growth_rate=(history[-1] - history[0]) / len(history) if len(history) > 1 else 0
            )

    def _analyze_fitness(self) -> FitnessStats:
        """Analyze fitness evolution"""
        history = self.results.get("fitness_history", [])

        if not history:
            return FitnessStats(0, 0, 0, 0, 0, 0)

        if HAS_NUMPY:
            arr = np.array(history)
            return FitnessStats(
                min_fitness=float(arr.min()),
                max_fitness=float(arr.max()),
                mean_fitness=float(arr.mean()),
                std_fitness=float(arr.std()),
                final_fitness=history[-1],
                improvement_rate=float(np.mean(np.diff(arr))) if len(arr) > 1 else 0
            )
        else:
            return FitnessStats(
                min_fitness=min(history),
                max_fitness=max(history),
                mean_fitness=sum(history) / len(history),
                std_fitness=self._std(history),
                final_fitness=history[-1],
                improvement_rate=(history[-1] - history[0]) / len(history) if len(history) > 1 else 0
            )

    def _analyze_traits(self) -> List[TraitAnalysis]:
        """Analyze trait distributions across generations"""
        generations = self.results.get("generations", [])

        if not generations:
            return []

        # Collect trait data
        trait_data: Dict[str, List[float]] = {}
        for gen in generations:
            best_traits = gen.get("best_traits", {})
            for trait, value in best_traits.items():
                if trait not in trait_data:
                    trait_data[trait] = []
                trait_data[trait].append(value)

        # Analyze each trait
        analyses = []
        for trait, values in trait_data.items():
            if not values:
                continue

            # Determine trend
            if len(values) > 1:
                diff = values[-1] - values[0]
                if diff > 0.1:
                    trend = "increasing"
                elif diff < -0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            if HAS_NUMPY:
                arr = np.array(values)
                analyses.append(TraitAnalysis(
                    trait_name=trait,
                    min_value=float(arr.min()),
                    max_value=float(arr.max()),
                    mean_value=float(arr.mean()),
                    std_value=float(arr.std()),
                    trend=trend
                ))
            else:
                analyses.append(TraitAnalysis(
                    trait_name=trait,
                    min_value=min(values),
                    max_value=max(values),
                    mean_value=sum(values) / len(values),
                    std_value=self._std(values),
                    trend=trend
                ))

        return analyses

    def _calculate_survival_rate(self) -> float:
        """Calculate survival rate"""
        births = self.results.get("metrics", {}).get("total_organisms_created", 0)
        deaths = self.results.get("metrics", {}).get("total_deaths", 0)
        if births == 0:
            return 0
        return (births - deaths) / births

    def _std(self, values: List[float]) -> float:
        """Pure Python standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    # ========================================================================
    # Export Methods
    # ========================================================================

    def export_csv(self, filepath: str):
        """Export population and fitness history to CSV"""
        pop_history = self.results.get("population_history", [])
        fit_history = self.results.get("fitness_history", [])

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["tick", "population", "fitness"])

            for i in range(max(len(pop_history), len(fit_history))):
                pop = pop_history[i] if i < len(pop_history) else ""
                fit = fit_history[i] if i < len(fit_history) else ""
                writer.writerow([i, pop, fit])

        print(f"Exported CSV to: {filepath}")

    def export_json(self, filepath: str):
        """Export complete analysis to JSON"""
        if not self.report:
            self.analyze()

        output = {
            "report": asdict(self.report),
            "population_history": self.results.get("population_history", []),
            "fitness_history": self.results.get("fitness_history", []),
            "generations": self.results.get("generations", [])
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Exported JSON to: {filepath}")

    def generate_html_report(self, filepath: str):
        """Generate an HTML report with embedded charts"""
        if not self.report:
            self.analyze()

        pop_history = self.results.get("population_history", [])
        fit_history = self.results.get("fitness_history", [])

        # Sample data for charts (every 10th point)
        pop_sampled = pop_history[::10] if len(pop_history) > 100 else pop_history
        fit_sampled = fit_history[::10] if len(fit_history) > 100 else fit_history

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>NPCPU Simulation Report - {self.report.simulation_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .stat {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #4CAF50; }}
        .stat-label {{ color: #666; }}
        .chart-container {{ height: 300px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f9f9f9; }}
        .trend-up {{ color: #4CAF50; }}
        .trend-down {{ color: #f44336; }}
        .trend-stable {{ color: #9e9e9e; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>NPCPU Simulation Report</h1>
            <p><strong>Simulation:</strong> {self.report.simulation_name}</p>
            <p><strong>Analysis Time:</strong> {self.report.analysis_time}</p>
            <p><strong>Duration:</strong> {self.report.duration_ticks} ticks ({self.report.duration_seconds:.2f} seconds)</p>
        </div>

        <div class="card">
            <h2>Key Metrics</h2>
            <div class="stat">
                <div class="stat-value">{self.report.population_stats.max_population}</div>
                <div class="stat-label">Peak Population</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.report.population_stats.final_population}</div>
                <div class="stat-label">Final Population</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.report.fitness_stats.max_fitness:.3f}</div>
                <div class="stat-label">Peak Fitness</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.report.total_births}</div>
                <div class="stat-label">Total Births</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.report.survival_rate*100:.1f}%</div>
                <div class="stat-label">Survival Rate</div>
            </div>
        </div>

        <div class="card">
            <h2>Population Over Time</h2>
            <div class="chart-container">
                <canvas id="populationChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Fitness Evolution</h2>
            <div class="chart-container">
                <canvas id="fitnessChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Trait Analysis</h2>
            <table>
                <tr>
                    <th>Trait</th>
                    <th>Mean</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Trend</th>
                </tr>
                {"".join(f'''
                <tr>
                    <td>{t.trait_name}</td>
                    <td>{t.mean_value:.3f}</td>
                    <td>{t.min_value:.3f}</td>
                    <td>{t.max_value:.3f}</td>
                    <td class="trend-{t.trend.split()[0]}">{t.trend}</td>
                </tr>
                ''' for t in self.report.trait_analyses)}
            </table>
        </div>

        <div class="card">
            <h2>Detailed Statistics</h2>
            <h3>Population</h3>
            <table>
                <tr><td>Mean</td><td>{self.report.population_stats.mean_population:.2f}</td></tr>
                <tr><td>Std Dev</td><td>{self.report.population_stats.std_population:.2f}</td></tr>
                <tr><td>Growth Rate</td><td>{self.report.population_stats.growth_rate:.4f} per tick</td></tr>
            </table>
            <h3>Fitness</h3>
            <table>
                <tr><td>Mean</td><td>{self.report.fitness_stats.mean_fitness:.4f}</td></tr>
                <tr><td>Std Dev</td><td>{self.report.fitness_stats.std_fitness:.4f}</td></tr>
                <tr><td>Improvement Rate</td><td>{self.report.fitness_stats.improvement_rate:.6f} per tick</td></tr>
            </table>
        </div>
    </div>

    <script>
        // Population Chart
        new Chart(document.getElementById('populationChart'), {{
            type: 'line',
            data: {{
                labels: {list(range(0, len(pop_sampled) * 10, 10))},
                datasets: [{{
                    label: 'Population',
                    data: {pop_sampled},
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: true,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});

        // Fitness Chart
        new Chart(document.getElementById('fitnessChart'), {{
            type: 'line',
            data: {{
                labels: {list(range(0, len(fit_sampled) * 10, 10))},
                datasets: [{{
                    label: 'Average Fitness',
                    data: {fit_sampled},
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    fill: true,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
    </script>
</body>
</html>"""

        with open(filepath, 'w') as f:
            f.write(html)

        print(f"Generated HTML report: {filepath}")

    # ========================================================================
    # Matplotlib Visualizations
    # ========================================================================

    def plot_population(self, save_path: Optional[str] = None, show: bool = True):
        """Plot population over time"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        history = self.results.get("population_history", [])
        if not history:
            print("No population history to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(history, color='#4CAF50', linewidth=1.5)
        ax.fill_between(range(len(history)), history, alpha=0.3, color='#4CAF50')

        ax.set_xlabel('Tick')
        ax.set_ylabel('Population')
        ax.set_title(f"Population Over Time - {self.results.get('config', {}).get('name', 'Simulation')}")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved population plot: {save_path}")

        if show:
            plt.show()

        plt.close()

    def plot_fitness(self, save_path: Optional[str] = None, show: bool = True):
        """Plot fitness evolution"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        history = self.results.get("fitness_history", [])
        if not history:
            print("No fitness history to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(history, color='#2196F3', linewidth=1.5)
        ax.fill_between(range(len(history)), history, alpha=0.3, color='#2196F3')

        ax.set_xlabel('Tick')
        ax.set_ylabel('Average Fitness')
        ax.set_title(f"Fitness Evolution - {self.results.get('config', {}).get('name', 'Simulation')}")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved fitness plot: {save_path}")

        if show:
            plt.show()

        plt.close()

    def plot_combined(self, save_path: Optional[str] = None, show: bool = True):
        """Plot population and fitness together"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        pop_history = self.results.get("population_history", [])
        fit_history = self.results.get("fitness_history", [])

        if not pop_history and not fit_history:
            print("No data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Population
        if pop_history:
            ax1.plot(pop_history, color='#4CAF50', linewidth=1.5)
            ax1.fill_between(range(len(pop_history)), pop_history, alpha=0.3, color='#4CAF50')
            ax1.set_ylabel('Population')
            ax1.set_title(f"Simulation Analysis - {self.results.get('config', {}).get('name', 'Simulation')}")
            ax1.grid(True, alpha=0.3)

        # Fitness
        if fit_history:
            ax2.plot(fit_history, color='#2196F3', linewidth=1.5)
            ax2.fill_between(range(len(fit_history)), fit_history, alpha=0.3, color='#2196F3')
            ax2.set_ylabel('Average Fitness')
            ax2.set_xlabel('Tick')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved combined plot: {save_path}")

        if show:
            plt.show()

        plt.close()

    def plot_generations(self, save_path: Optional[str] = None, show: bool = True):
        """Plot generation statistics"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        generations = self.results.get("generations", [])
        if not generations:
            print("No generation data to plot")
            return

        gen_nums = [g["generation"] for g in generations]
        populations = [g["population"] for g in generations]
        fitness = [g["avg_fitness"] for g in generations]
        diversity = [g.get("diversity", 0) for g in generations]

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        axes[0].bar(gen_nums, populations, color='#4CAF50', alpha=0.7)
        axes[0].set_ylabel('Population')
        axes[0].set_title('Generation Statistics')

        axes[1].plot(gen_nums, fitness, 'o-', color='#2196F3')
        axes[1].set_ylabel('Avg Fitness')

        axes[2].plot(gen_nums, diversity, 's-', color='#FF9800')
        axes[2].set_ylabel('Diversity')
        axes[2].set_xlabel('Generation')

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved generation plot: {save_path}")

        if show:
            plt.show()

        plt.close()


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--csv", help="Export to CSV file")
    parser.add_argument("--json", help="Export analysis to JSON file")
    parser.add_argument("--html", help="Generate HTML report")
    parser.add_argument("--plot", choices=["population", "fitness", "combined", "generations", "all"],
                       help="Generate plots")
    parser.add_argument("--save-plots", help="Directory to save plots")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")

    args = parser.parse_args()

    # Load and analyze
    analyzer = SimulationAnalyzer.from_file(args.results_file)
    report = analyzer.analyze()

    # Print summary
    print(f"\n{'='*60}")
    print(f"SIMULATION ANALYSIS: {report.simulation_name}")
    print(f"{'='*60}")
    print(f"\nDuration: {report.duration_ticks} ticks ({report.duration_seconds:.2f}s)")
    print(f"\nPopulation:")
    print(f"  Peak: {report.population_stats.max_population}")
    print(f"  Final: {report.population_stats.final_population}")
    print(f"  Mean: {report.population_stats.mean_population:.2f}")
    print(f"\nFitness:")
    print(f"  Peak: {report.fitness_stats.max_fitness:.4f}")
    print(f"  Final: {report.fitness_stats.final_fitness:.4f}")
    print(f"  Mean: {report.fitness_stats.mean_fitness:.4f}")
    print(f"\nTraits analyzed: {len(report.trait_analyses)}")
    print(f"Survival rate: {report.survival_rate*100:.1f}%")

    # Exports
    if args.csv:
        analyzer.export_csv(args.csv)

    if args.json:
        analyzer.export_json(args.json)

    if args.html:
        analyzer.generate_html_report(args.html)

    # Plots
    if args.plot:
        show = not args.no_show
        save_dir = Path(args.save_plots) if args.save_plots else None

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        if args.plot in ["population", "all"]:
            save_path = str(save_dir / "population.png") if save_dir else None
            analyzer.plot_population(save_path, show)

        if args.plot in ["fitness", "all"]:
            save_path = str(save_dir / "fitness.png") if save_dir else None
            analyzer.plot_fitness(save_path, show)

        if args.plot in ["combined", "all"]:
            save_path = str(save_dir / "combined.png") if save_dir else None
            analyzer.plot_combined(save_path, show)

        if args.plot in ["generations", "all"]:
            save_path = str(save_dir / "generations.png") if save_dir else None
            analyzer.plot_generations(save_path, show)


if __name__ == "__main__":
    main()
