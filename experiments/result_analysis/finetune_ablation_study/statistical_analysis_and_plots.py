#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(results_folder: Path):
    """
    Load all CSV result files and prepare data for analysis.
    
    Args:
        results_folder: Path to the folder containing result CSV files
        
    Returns:
        Dictionary with dataset names as keys and prepared data as values
    """
    all_datasets = {}
    
    # Find all CSV files in the results folder
    csv_files = list(results_folder.glob("*_hyperparameter_ablation_results.csv"))
    
    for csv_file in csv_files:
        # Extract dataset name from filename
        dataset_name = csv_file.stem.replace("_hyperparameter_ablation_results", "")
        
        # Load the CSV
        df = pd.read_csv(csv_file)
        
        # Rename method names to have spaces instead of underscores
        df['method'] = df['method'].replace({
            'Optimized_Hyperparameters': 'Optimized Hyperparameters',
            'Fixed_Hyperparameters': 'Fixed Hyperparameters'
        })
        
        # Prepare data for each metric
        prepared_data = {}
        metrics = df['metric_name'].unique()
        
        for metric in metrics:
            # Filter data for current metric
            metric_data = df[df['metric_name'] == metric]
            
            # Check if we have both methods
            methods = metric_data['method'].unique()
            if len(methods) < 2:
                print(f"Skipping {dataset_name} - {metric}: only has {methods}")
                continue
            
            # Pivot the data to have methods as columns and folds as rows
            pivot_data = metric_data.pivot(
                index='fold',
                columns='method',
                values='value'
            )
            
            prepared_data[metric] = {
                'data': pivot_data,
                'metric_name': metric
            }
        
        if prepared_data:
            all_datasets[dataset_name] = prepared_data
    
    return all_datasets

def find_best_model(data: pd.DataFrame, metric_name: str) -> str:
    """
    Find the best performing model based on the metric values.
    For MAE and MSE, lower is better. For R2 and rho, higher is better.
    
    Args:
        data: DataFrame with models as columns and metric values as rows
        metric_name: Name of the metric being evaluated
        
    Returns:
        Name of the best performing model
    """
    # Calculate mean performance for each model
    model_means = data.mean()
    
    if metric_name in ['mae', 'mse']:
        # For MAE and MSE, lower is better
        best_model = model_means.idxmin()
    else:
        # For R2 and rho, higher is better
        best_model = model_means.idxmax()
    
    return best_model

def repeated_measures_anova(data: pd.DataFrame) -> dict:
    """
    Perform repeated measures ANOVA on the data.
    
    Args:
        data: DataFrame with models as columns and folds as rows
        
    Returns:
        Dictionary containing ANOVA results
    """
    # Reshape data for ANOVA
    data_melted = data.reset_index().melt(
        id_vars=['fold'], 
        var_name='method', 
        value_name='value'
    )
    
    # Perform repeated measures ANOVA
    anova_rm = AnovaRM(data_melted, 'value', 'fold', within=['method'])
    anova_results = anova_rm.fit()
    
    # Extract p-value
    p_value = anova_results.anova_table['Pr > F']['method']
    f_stat = anova_results.anova_table['F Value']['method']
    
    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'anova_table': anova_results.anova_table
    }

def tukey_hsd(data: pd.DataFrame) -> dict:
    """
    Perform Tukey's HSD test on the data.
    
    Args:
        data: DataFrame with models as columns and folds as rows
        
    Returns:
        Dictionary containing Tukey HSD results
    """
    # Reshape data for Tukey HSD
    data_melted = data.reset_index().melt(
        id_vars=['fold'], 
        var_name='method', 
        value_name='value'
    )
    
    # Perform Tukey HSD test
    tukey_results = pairwise_tukeyhsd(
        endog=data_melted['value'],
        groups=data_melted['method'],
        alpha=0.05
    )
    
    return {
        'results': tukey_results,
        'summary': tukey_results.summary()
    }

def create_analysis_plots(all_datasets: dict, output_folder: Path):
    """
    Create analysis plots in the style of the user's notebook.
    
    Args:
        all_datasets: Dictionary with dataset data
        output_folder: Path to save plots
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("=" * 80)
    
    # Process each dataset
    for dataset_name, dataset_data in all_datasets.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 50)
        
        # Process each metric
        for metric, metric_data in dataset_data.items():
            data = metric_data['data']
            metric_name = metric_data['metric_name']
            
            print(f"\nMetric: {metric_name}")
            
            # Calculate descriptive statistics
            for method in data.columns:
                mean_val = data[method].mean()
                std_val = data[method].std()
                print(f"  {method}: {mean_val:.4f} ± {std_val:.4f}")
            
            # Perform repeated measures ANOVA
            anova_results = repeated_measures_anova(data)
            print(f"  Repeated Measures ANOVA: F={anova_results['f_stat']:.4f}, p={anova_results['p_value']:.4f}")
            
            # Determine significance
            if anova_results['p_value'] < 0.001:
                significance = "***"
            elif anova_results['p_value'] < 0.01:
                significance = "**"
            elif anova_results['p_value'] < 0.05:
                significance = "*"
            else:
                significance = "ns"
            
            print(f"  Significance: {significance}")
            
            # Find the best performing model
            best_model = find_best_model(data, metric_name)
            print(f"  Best performing method: {best_model}")
            
            # Always perform Tukey's HSD test and create plot (like in the notebook)
            tukey_results = tukey_hsd(data)
            
            # Create and save the Tukey HSD plot with improved aesthetics
            fig, ax = plt.subplots(figsize=(7.5, 2.5))
            tukey_results['results'].plot_simultaneous(comparison_name=best_model, ax=ax)
            
            # Apply aesthetic styling to match existing plots
            ax.set_title('HCLint', fontsize=10, color='#666666')
            ax.set_xlabel(metric_name.upper(), fontsize=10, color='#666666')
            ax.set_ylabel('Method', fontsize=10, color='#666666')
            
            # Set tick parameters
            ax.tick_params(axis='x', labelsize=8, colors='#666666')
            ax.tick_params(axis='y', labelsize=8, colors='#333333')
            
            # Make borders light grey
            for spine in ax.spines.values():
                spine.set_color('#CCCCCC')
                spine.set_linewidth(0.8)
            
            # Remove lowest and highest y-axis ticks if there are more than 2
            yticks = ax.get_yticks()
            if len(yticks) > 2:
                new_yticks = yticks[1:-1]
                ax.set_yticks(new_yticks)
            
            plt.tight_layout()
            
            # Save the plot in both PNG and PDF formats
            base_filename = f"{dataset_name}_{metric_name}_tukey_hsd"
            
            # Save as PNG
            png_filename = f"{base_filename}.png"
            plt.savefig(output_folder / png_filename, dpi=300, bbox_inches='tight')
            print(f"  Tukey HSD plot saved: {png_filename}")
            
            # Save as PDF
            pdf_filename = f"{base_filename}.pdf"
            plt.savefig(output_folder / pdf_filename, bbox_inches='tight')
            print(f"  Tukey HSD plot saved: {pdf_filename}")
            
            plt.close()
            
            # Print Tukey HSD summary
            print(f"  Tukey HSD Summary:")
            print(f"    {tukey_results['summary']}")
            
            # Check if the ANOVA result is significant (p < 0.05)
            if anova_results['p_value'] < 0.05:
                print(f"  Significant differences found for {metric_name}")
            else:
                print(f"  No significant differences found for {metric_name}")

def create_summary_table(all_datasets: dict, output_folder: Path):
    """
    Create a summary table of all statistical results.
    
    Args:
        all_datasets: Dictionary with dataset data
        output_folder: Path to save the table
    """
    summary_data = []
    
    for dataset_name, dataset_data in all_datasets.items():
        for metric, metric_data in dataset_data.items():
            data = metric_data['data']
            metric_name = metric_data['metric_name']
            
            # Calculate statistics for each method
            method_stats = {}
            for method in data.columns:
                method_stats[method] = {
                    'mean': data[method].mean(),
                    'std': data[method].std()
                }
            
            # Perform ANOVA
            anova_results = repeated_measures_anova(data)
            
            # Find best method
            best_method = find_best_model(data, metric_name)
            
            # Determine significance
            if anova_results['p_value'] < 0.001:
                significance = "***"
            elif anova_results['p_value'] < 0.01:
                significance = "**"
            elif anova_results['p_value'] < 0.05:
                significance = "*"
            else:
                significance = "ns"
            
            # Add to summary
            for method in data.columns:
                summary_data.append({
                    'Dataset': dataset_name,
                    'Metric': metric_name.upper(),
                    'Method': method,
                    'Mean': f"{method_stats[method]['mean']:.4f}",
                    'Std': f"{method_stats[method]['std']:.4f}",
                    'F_Statistic': f"{anova_results['f_stat']:.4f}",
                    'P_Value': f"{anova_results['p_value']:.4f}",
                    'Significance': significance,
                    'Best_Method': best_method
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_df.to_csv(output_folder / 'statistical_summary.csv', index=False)
    print(f"\nSummary table saved to: {output_folder / 'statistical_summary.csv'}")

def main():
    """Main function to run the complete statistical analysis and plotting."""
    
    # Set up paths
    results_folder = Path("results")
    plots_folder = Path("/Users/fabian/Code/smilesencoder/plots/hparam_insensivity/finetune")
    
    print("Loading and preparing data...")
    try:
        all_datasets = load_and_prepare_data(results_folder)
        
        if not all_datasets:
            print("No datasets with complete data found!")
            return
        
        print(f"Loaded {len(all_datasets)} datasets:")
        for dataset_name, dataset_data in all_datasets.items():
            print(f"  - {dataset_name}: {len(dataset_data)} metrics")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("\nPerforming statistical analysis and creating plots...")
    try:
        create_analysis_plots(all_datasets, plots_folder)
    except Exception as e:
        print(f"Error in analysis: {e}")
        return
    
    print("\nCreating summary table...")
    try:
        create_summary_table(all_datasets, plots_folder)
    except Exception as e:
        print(f"Error creating summary table: {e}")
        return
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"All outputs saved to: {plots_folder.absolute()}")

if __name__ == "__main__":
    main() 