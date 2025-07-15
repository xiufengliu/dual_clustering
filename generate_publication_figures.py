#!/usr/bin/env python3
"""
Enhanced figure generation script with PDF support for publication-quality outputs.
This script generates both PNG and PDF versions of all experimental figures.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.visualization.experiment_plots import ExperimentVisualizer
from src.utils.logger import setup_logger

# Set matplotlib backend for PDF generation
matplotlib.use('Agg')  # Use non-interactive backend

class PublicationFigureGenerator:
    """Enhanced figure generator with PDF support for publication-quality outputs."""
    
    def __init__(self, output_dir: str = "results/figures", paper_images_dir: str = "paper/images"):
        """Initialize the publication figure generator.
        
        Args:
            output_dir: Directory to save figures
            paper_images_dir: Directory for paper images (will copy figures here)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paper images directory
        self.paper_images_dir = Path(paper_images_dir)
        self.paper_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.png_dir = self.output_dir / "png"
        self.pdf_dir = self.output_dir / "pdf"
        self.png_dir.mkdir(exist_ok=True)
        self.pdf_dir.mkdir(exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = ExperimentVisualizer()
        
        # Setup logger
        self.logger = setup_logger("publication_figures")
        
        # Configure matplotlib for publication quality
        self._configure_matplotlib()
    
    def _configure_matplotlib(self):
        """Configure matplotlib for publication-quality figures."""
        # Set publication-quality parameters
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5,
            'axes.linewidth': 1.0,
        })
    
    def generate_all_figures(self, results_file: str, formats: List[str] = ['png', 'pdf']):
        """Generate all publication figures from results file.
        
        Args:
            results_file: Path to JSON results file
            formats: List of formats to save ('png', 'pdf', or both)
        """
        self.logger.info(f"Loading results from: {results_file}")
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        self.logger.info("Generating publication-quality figures...")
        
        figures_generated = []
        
        # 1. Model Comparison Figure
        if 'main_results' in results:
            self.logger.info("Generating model comparison figure...")
            fig = self.visualizer.plot_model_comparison(results['main_results'])
            self._save_figure(fig, 'model_comparison', formats)
            figures_generated.append('model_comparison')
            plt.close(fig)
        
        # 2. Statistical Significance Heatmap
        if 'main_results' in results:
            self.logger.info("Generating statistical significance heatmap...")
            statistical_data = self._extract_statistical_data(results['main_results'])
            if statistical_data:
                fig = self.visualizer.plot_statistical_significance_heatmap(statistical_data)
                self._save_figure(fig, 'statistical_significance', formats)
                figures_generated.append('statistical_significance')
                plt.close(fig)
        
        # 3. Ablation Study
        if 'ablation_studies' in results:
            self.logger.info("Generating ablation study figure...")
            fig = self.visualizer.plot_ablation_study(results['ablation_studies'])
            self._save_figure(fig, 'ablation_study', formats)
            figures_generated.append('ablation_study')
            plt.close(fig)
        
        # 4. Sensitivity Analysis
        if 'sensitivity_analysis' in results:
            self.logger.info("Generating sensitivity analysis figure...")
            fig = self.visualizer.plot_sensitivity_analysis(results['sensitivity_analysis'])
            self._save_figure(fig, 'sensitivity_analysis', formats)
            figures_generated.append('sensitivity_analysis')
            plt.close(fig)
        
        # 5. Computational Analysis
        if 'computational_analysis' in results:
            self.logger.info("Generating computational analysis figure...")
            fig = self.visualizer.plot_computational_analysis(results['computational_analysis'])
            self._save_figure(fig, 'computational_analysis', formats)
            figures_generated.append('computational_analysis')
            plt.close(fig)
        
        # 6. Robustness Analysis
        if 'robustness_analysis' in results:
            self.logger.info("Generating robustness analysis figure...")
            fig = self.visualizer.plot_robustness_analysis(results['robustness_analysis'])
            self._save_figure(fig, 'robustness_analysis', formats)
            figures_generated.append('robustness_analysis')
            plt.close(fig)
        
        # 7. Generate Summary Report (PDF only)
        if 'pdf' in formats:
            self.logger.info("Generating comprehensive summary report...")
            self._generate_comprehensive_report(results, figures_generated)
        
        # 8. Generate Results Table (CSV and LaTeX)
        self._generate_results_table(results)
        
        self.logger.info(f"Generated {len(figures_generated)} figures: {', '.join(figures_generated)}")
        self.logger.info(f"Figures saved to:")
        if 'png' in formats:
            self.logger.info(f"  PNG: {self.png_dir}")
        if 'pdf' in formats:
            self.logger.info(f"  PDF: {self.pdf_dir}")
    
    def _save_figure(self, fig: plt.Figure, name: str, formats: List[str]):
        """Save figure in specified formats and copy to paper images directory.
        
        Args:
            fig: Matplotlib figure
            name: Base filename
            formats: List of formats to save
        """
        import shutil
        
        if 'png' in formats:
            png_path = self.png_dir / f"{name}.png"
            fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
            
            # Copy to paper images directory
            paper_png_path = self.paper_images_dir / f"{name}.png"
            shutil.copy2(png_path, paper_png_path)
            
        if 'pdf' in formats:
            pdf_path = self.pdf_dir / f"{name}.pdf"
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
            
            # Copy to paper images directory
            paper_pdf_path = self.paper_images_dir / f"{name}.pdf"
            shutil.copy2(pdf_path, paper_pdf_path)
    
    def _extract_statistical_data(self, main_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract statistical test data from main results."""
        statistical_data = {}
        for dataset, dataset_results in main_results.items():
            if 'statistical_tests' in dataset_results:
                statistical_data[dataset] = dataset_results['statistical_tests']
        return statistical_data
    
    def _generate_comprehensive_report(self, results: Dict[str, Any], figures_generated: List[str]):
        """Generate a comprehensive PDF report with all results and figures.
        
        Args:
            results: Complete experimental results
            figures_generated: List of generated figure names
        """
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            
            pdf_path = self.pdf_dir / "comprehensive_experiment_report.pdf"
            
            with PdfPages(pdf_path) as pdf:
                # Title page
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.text(0.5, 0.8, 'Neutrosophic Dual Clustering Random Forest\nComprehensive Experiment Report', 
                       ha='center', va='center', fontsize=18, fontweight='bold')
                ax.text(0.5, 0.65, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                       ha='center', va='center', fontsize=12)
                ax.text(0.5, 0.55, f'Figures Generated: {len(figures_generated)}', 
                       ha='center', va='center', fontsize=12)
                ax.text(0.5, 0.45, f'Datasets Analyzed: {len(results.get("main_results", {}))}', 
                       ha='center', va='center', fontsize=12)
                
                # Add experiment info if available
                if 'experiment_info' in results:
                    exp_info = results['experiment_info']
                    y_pos = 0.35
                    for key, value in exp_info.items():
                        ax.text(0.5, y_pos, f'{key}: {value}', 
                               ha='center', va='center', fontsize=10)
                        y_pos -= 0.05
                
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # Results summary pages
                self._add_results_summary_pages(pdf, results)
                
                # Performance metrics page
                if 'performance_metrics' in results:
                    self._add_performance_metrics_page(pdf, results['performance_metrics'])
            
            self.logger.info(f"Comprehensive report saved to: {pdf_path}")
            
        except ImportError:
            self.logger.warning("PdfPages not available, skipping comprehensive report generation")
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
    
    def _add_results_summary_pages(self, pdf, results: Dict[str, Any]):
        """Add results summary pages to PDF."""
        if 'main_results' not in results:
            return
            
        for dataset, dataset_results in results['main_results'].items():
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.95, f'Results: {dataset.replace("_", " ").title()}', 
                   ha='center', va='top', fontsize=16, fontweight='bold')
            
            y_pos = 0.85
            
            if 'model_results' in dataset_results:
                ax.text(0.1, y_pos, 'Model Performance:', fontsize=14, fontweight='bold')
                y_pos -= 0.05
                
                # Sort models by RMSE for better presentation
                model_items = list(dataset_results['model_results'].items())
                model_items.sort(key=lambda x: x[1].get('point_metrics', {}).get('rmse', float('inf')))
                
                for model_name, model_results in model_items:
                    if 'point_metrics' in model_results and not model_results.get('failed', False):
                        metrics = model_results['point_metrics']
                        
                        # Highlight our method
                        if 'NDC-RF' in model_name or 'neutrosophic' in model_name.lower():
                            ax.text(0.15, y_pos, f"→ {model_name} (Proposed Method)", 
                                   fontsize=11, fontweight='bold', color='red')
                        else:
                            ax.text(0.15, y_pos, f"  {model_name}", fontsize=11)
                        y_pos -= 0.03
                        
                        # Add key metrics
                        for metric in ['rmse', 'mae', 'mape', 'r2']:
                            if metric in metrics:
                                ax.text(0.2, y_pos, f"{metric.upper()}: {metrics[metric]:.4f}", 
                                       fontsize=9)
                                y_pos -= 0.025
                        y_pos -= 0.02
            
            # Add statistical test results if available
            if 'statistical_tests' in dataset_results:
                ax.text(0.1, y_pos, 'Statistical Significance:', fontsize=14, fontweight='bold')
                y_pos -= 0.05
                
                stat_tests = dataset_results['statistical_tests']
                for test_name, test_result in stat_tests.items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        significance = "Significant" if test_result['p_value'] < 0.05 else "Not Significant"
                        ax.text(0.15, y_pos, f"{test_name}: p={test_result['p_value']:.4f} ({significance})", 
                               fontsize=10)
                        y_pos -= 0.03
            
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def _add_performance_metrics_page(self, pdf, performance_metrics: Dict[str, Any]):
        """Add performance metrics page to PDF."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 0.95, 'Experiment Performance Metrics', 
               ha='center', va='top', fontsize=16, fontweight='bold')
        
        y_pos = 0.85
        
        for key, value in performance_metrics.items():
            if isinstance(value, dict):
                ax.text(0.1, y_pos, f"{key.replace('_', ' ').title()}:", 
                       fontsize=12, fontweight='bold')
                y_pos -= 0.05
                
                for sub_key, sub_value in value.items():
                    ax.text(0.15, y_pos, f"{sub_key}: {sub_value}", fontsize=10)
                    y_pos -= 0.03
            else:
                ax.text(0.1, y_pos, f"{key.replace('_', ' ').title()}: {value}", 
                       fontsize=12)
                y_pos -= 0.05
        
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_results_table(self, results: Dict[str, Any]):
        """Generate results table in CSV and LaTeX formats.
        
        Args:
            results: Experimental results
        """
        if 'main_results' not in results:
            return
        
        # Prepare data for table
        table_data = []
        
        for dataset, dataset_results in results['main_results'].items():
            if 'model_results' in dataset_results:
                for model_name, model_results in dataset_results['model_results'].items():
                    if 'point_metrics' in model_results and not model_results.get('failed', False):
                        metrics = model_results['point_metrics']
                        row = {
                            'Dataset': dataset.replace('_', ' ').title(),
                            'Model': model_name,
                            'RMSE': metrics.get('rmse', 'N/A'),
                            'MAE': metrics.get('mae', 'N/A'),
                            'MAPE': metrics.get('mape', 'N/A'),
                            'R²': metrics.get('r2', 'N/A')
                        }
                        table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            
            # Save as CSV
            csv_path = self.output_dir / "results_table.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Results table saved to: {csv_path}")
            
            # Save as LaTeX
            latex_path = self.output_dir / "results_table.tex"
            latex_table = df.to_latex(index=False, float_format="%.4f", escape=False)
            with open(latex_path, 'w') as f:
                f.write(latex_table)
            self.logger.info(f"LaTeX table saved to: {latex_path}")


def main():
    """Main function to generate publication figures."""
    parser = argparse.ArgumentParser(description="Generate publication-quality figures from experiment results")
    parser.add_argument("results_file", help="Path to JSON results file")
    parser.add_argument("--output-dir", default="results/figures", help="Output directory for figures")
    parser.add_argument("--formats", nargs="+", choices=['png', 'pdf'], default=['png', 'pdf'],
                       help="Output formats (default: both png and pdf)")
    parser.add_argument("--find-latest", action="store_true", 
                       help="Automatically find the latest results file")
    
    args = parser.parse_args()
    
    # Find latest results file if requested
    if args.find_latest:
        results_dir = Path("results")
        result_files = []
        for subdir in ["comprehensive", "optimized"]:
            subdir_path = results_dir / subdir
            if subdir_path.exists():
                result_files.extend(subdir_path.glob("*.json"))
        
        if result_files:
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            args.results_file = str(latest_file)
            print(f"Using latest results file: {args.results_file}")
        else:
            print("No results files found!")
            return 1
    
    # Check if results file exists
    if not Path(args.results_file).exists():
        print(f"Results file not found: {args.results_file}")
        return 1
    
    # Generate figures
    generator = PublicationFigureGenerator(args.output_dir)
    
    try:
        generator.generate_all_figures(args.results_file, args.formats)
        print(f"\n🎉 Publication figures generated successfully!")
        print(f"Output directory: {args.output_dir}")
        if 'png' in args.formats:
            print(f"PNG files: {Path(args.output_dir) / 'png'}")
        if 'pdf' in args.formats:
            print(f"PDF files: {Path(args.output_dir) / 'pdf'}")
        return 0
    except Exception as e:
        print(f"❌ Failed to generate figures: {e}")
        return 1


if __name__ == "__main__":
    exit(main())