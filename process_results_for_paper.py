#!/usr/bin/env python3
"""
Process experimental results for paper publication.
This script will:
1. Find the latest experimental results
2. Generate publication-quality figures and save to paper/images/
3. Create LaTeX tables for the paper
4. Generate a summary of results for the experiment section
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent))

from generate_publication_figures import PublicationFigureGenerator

def find_latest_results():
    """Find the most recent experimental results file."""
    results_dirs = ["results/optimized", "results/comprehensive"]
    latest_file = None
    latest_time = 0
    
    for results_dir in results_dirs:
        results_path = Path(results_dir)
        if results_path.exists():
            for json_file in results_path.glob("*.json"):
                file_time = json_file.stat().st_mtime
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = json_file
    
    return latest_file

def extract_results_summary(results_file):
    """Extract key results for the paper."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    summary = {
        'datasets': [],
        'models': [],
        'performance': {},
        'statistical_tests': {},
        'experiment_info': results.get('experiment_info', {}),
        'performance_metrics': results.get('performance_metrics', {})
    }
    
    # Extract main results
    if 'main_results' in results:
        for dataset, dataset_results in results['main_results'].items():
            summary['datasets'].append(dataset)
            
            if 'model_results' in dataset_results:
                dataset_performance = {}
                for model_name, model_results in dataset_results['model_results'].items():
                    if model_name not in summary['models']:
                        summary['models'].append(model_name)
                    
                    if 'point_metrics' in model_results and not model_results.get('failed', False):
                        dataset_performance[model_name] = model_results['point_metrics']
                
                summary['performance'][dataset] = dataset_performance
            
            # Extract statistical test results
            if 'statistical_tests' in dataset_results:
                summary['statistical_tests'][dataset] = dataset_results['statistical_tests']
    
    return summary

def generate_latex_table(summary):
    """Generate LaTeX table for the paper."""
    latex_tables = {}
    
    for dataset, performance in summary['performance'].items():
        if not performance:
            continue
            
        # Create DataFrame for this dataset
        rows = []
        for model_name, metrics in performance.items():
            row = {'Model': model_name}
            row.update(metrics)
            rows.append(row)
        
        if not rows:
            continue
            
        df = pd.DataFrame(rows)
        
        # Sort by RMSE (best first)
        if 'rmse' in df.columns:
            df = df.sort_values('rmse')
        
        # Generate LaTeX table
        latex_table = df.to_latex(
            index=False,
            float_format="%.4f",
            escape=False,
            caption=f"Performance comparison for {dataset.replace('_', ' ').title()} dataset",
            label=f"tab:{dataset}_results"
        )
        
        latex_tables[dataset] = latex_table
    
    return latex_tables

def generate_results_text(summary):
    """Generate text summary for the experiment section."""
    text_summary = []
    
    text_summary.append("## Experimental Results Summary")
    text_summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    text_summary.append("")
    
    # Dataset information
    text_summary.append(f"### Datasets Analyzed: {len(summary['datasets'])}")
    for dataset in summary['datasets']:
        text_summary.append(f"- {dataset.replace('_', ' ').title()}")
    text_summary.append("")
    
    # Model information
    text_summary.append(f"### Models Compared: {len(summary['models'])}")
    for model in summary['models']:
        text_summary.append(f"- {model}")
    text_summary.append("")
    
    # Performance summary
    text_summary.append("### Performance Results")
    for dataset, performance in summary['performance'].items():
        if not performance:
            continue
            
        text_summary.append(f"\n#### {dataset.replace('_', ' ').title()} Dataset")
        
        # Find best and proposed model performance
        best_rmse = float('inf')
        best_model = None
        proposed_rmse = None
        
        for model_name, metrics in performance.items():
            if 'rmse' in metrics:
                rmse = metrics['rmse']
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_name
                
                if 'NDC-RF' in model_name or 'neutrosophic' in model_name.lower():
                    proposed_rmse = rmse
        
        text_summary.append(f"- Best performing model: {best_model} (RMSE: {best_rmse:.4f})")
        
        if proposed_rmse is not None:
            if proposed_rmse == best_rmse:
                text_summary.append(f"- Proposed method achieved best performance (RMSE: {proposed_rmse:.4f})")
            else:
                improvement = ((best_rmse - proposed_rmse) / best_rmse) * 100
                text_summary.append(f"- Proposed method RMSE: {proposed_rmse:.4f}")
                if improvement > 0:
                    text_summary.append(f"- Improvement over best baseline: {improvement:.2f}%")
                else:
                    text_summary.append(f"- Performance gap: {-improvement:.2f}%")
    
    # Performance metrics
    if 'performance_metrics' in summary and summary['performance_metrics']:
        text_summary.append("\n### Computational Performance")
        perf_metrics = summary['performance_metrics']
        if 'total_execution_time' in perf_metrics:
            total_time = perf_metrics['total_execution_time']
            text_summary.append(f"- Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
        if 'datasets_processed' in perf_metrics:
            text_summary.append(f"- Datasets processed: {perf_metrics['datasets_processed']}")
        
        if 'average_time_per_dataset' in perf_metrics:
            avg_time = perf_metrics['average_time_per_dataset']
            text_summary.append(f"- Average time per dataset: {avg_time:.2f} seconds")
    
    return "\n".join(text_summary)

def main():
    """Main function to process results for paper."""
    parser = argparse.ArgumentParser(description="Process experimental results for paper")
    parser.add_argument("--results-file", help="Specific results file to process")
    parser.add_argument("--output-dir", default="paper_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Find results file
    if args.results_file:
        results_file = Path(args.results_file)
    else:
        results_file = find_latest_results()
    
    if not results_file or not results_file.exists():
        print("❌ No results file found!")
        print("Make sure experiments have completed successfully.")
        return 1
    
    print(f"📊 Processing results from: {results_file}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract results summary
    summary = extract_results_summary(results_file)
    
    # Generate figures and save to paper/images/
    print("🎨 Generating publication figures...")
    figure_generator = PublicationFigureGenerator(
        output_dir="results/figures",
        paper_images_dir="paper/images"
    )
    figure_generator.generate_all_figures(str(results_file), formats=['png', 'pdf'])
    
    # Generate LaTeX tables
    print("📋 Generating LaTeX tables...")
    latex_tables = generate_latex_table(summary)
    
    # Save LaTeX tables
    for dataset, latex_table in latex_tables.items():
        table_file = output_dir / f"{dataset}_table.tex"
        with open(table_file, 'w') as f:
            f.write(latex_table)
        print(f"   Saved: {table_file}")
    
    # Generate results summary text
    print("📝 Generating results summary...")
    results_text = generate_results_text(summary)
    
    # Save results summary
    summary_file = output_dir / "results_summary.md"
    with open(summary_file, 'w') as f:
        f.write(results_text)
    print(f"   Saved: {summary_file}")
    
    # Save raw summary data
    summary_json = output_dir / "results_summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved: {summary_json}")
    
    print(f"\n🎉 Results processing completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖼️  Figures saved to: paper/images/")
    print(f"📊 Tables and summaries in: {output_dir}")
    
    # Print quick summary
    print(f"\n📈 Quick Summary:")
    print(f"   Datasets: {len(summary['datasets'])}")
    print(f"   Models: {len(summary['models'])}")
    print(f"   Figures generated: Available in paper/images/")
    
    return 0

if __name__ == "__main__":
    exit(main())