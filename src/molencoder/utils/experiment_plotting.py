"""
Reusable plotting utilities for experimental analysis with statistical testing.

This module provides functions to analyze experimental results from CSV files,
perform statistical tests, and generate publication-ready plots.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from molencoder.evaluation.statistical_evaluation import repeated_measures_anova, tukey_hsd


def load_and_prepare_data(csv_path: str, model_name_mapping: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load the CSV file and prepare data for each metric.
    
    Args:
        csv_path: Path to the CSV file containing evaluation results
        model_name_mapping: Dictionary mapping full model names to short names
        
    Returns:
        Dictionary with metric names as keys and prepared DataFrames as values
    """
    df = pd.read_csv(csv_path)
    df['model'] = df['model'].map(model_name_mapping)
    
    prepared_data = {}
    metrics = df['metric_name'].unique()
    
    for metric in metrics:
        metric_data = df[df['metric_name'] == metric]
        
        pivot_data = metric_data.pivot(
            index='fold',
            columns='model',
            values='value'
        )
        
        prepared_data[metric] = pivot_data
    
    return prepared_data


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
    model_means = data.mean()
    
    if metric_name in ['mae', 'mse']:
        best_model = model_means.idxmin()
    else:
        best_model = model_means.idxmax()
    
    return best_model


def clean_prepared_data(prepared_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Clean the prepared data by removing 'nan' columns and handling missing values.
    
    Args:
        prepared_data: Dictionary with metric names as keys and DataFrames as values
        
    Returns:
        Cleaned dictionary with metric names as keys and DataFrames as values
    """
    cleaned_data = {}
    
    for metric_name, metric_data in prepared_data.items():
        if 'nan' in metric_data.columns:
            cleaned_data[metric_name] = metric_data.drop(columns=['nan'])
        elif len(metric_data.columns) > 0 and metric_data.columns[0] != metric_data.columns[0]:
            cleaned_data[metric_name] = metric_data.iloc[:, 1:]
        else:
            cleaned_data[metric_name] = metric_data.copy()
    
    return cleaned_data


def analyze_and_plot_experiments(
    csv_paths: List[str], 
    model_name_mapping: Dict[str, str], 
    output_dir: Optional[str] = None,
    dataset_name_transform: callable = None,
    y_axis_label: str = 'Masking Ratio'
) -> None:
    """
    Analyze experimental results and create plots for significant metrics.
    
    Args:
        csv_paths: List of paths to CSV files containing experimental results
        model_name_mapping: Dictionary mapping full model names to short display names
        output_dir: Optional directory to save the generated plots. If None, plots are only displayed.
        dataset_name_transform: Optional function to transform dataset names from file paths
        y_axis_label: Label for the y-axis (default: 'Masking Ratio')
    """
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_data = {}
    for csv_path in csv_paths:
        if dataset_name_transform:
            dataset_name = dataset_name_transform(csv_path)
        else:

            dataset_name = (Path(csv_path).stem
                          .replace('_results', '')
                          .replace('adme-fang-', '')
                          .replace('-1', '')
                          .replace("hclint", "HCLint")
                          .replace("perm", "Permeability")
                          .replace("solu", "Solubility")
                          .replace("-astrazeneca", "")
                          .replace("adme-novartis-", "")
                          .replace("-reg", "")
                          .replace("PERM", "Permeability")
                          .replace("SOLU", "Solubility")
                          .replace("HCLINT", "HCLint")
                          .replace("lipophilicity", "Lipophilicity")
                          .replace("cyp3a4", "CYP"))
        
        prepared_data = load_and_prepare_data(csv_path, model_name_mapping)
        cleaned_data = clean_prepared_data(prepared_data)
        all_data[dataset_name] = cleaned_data


    all_metrics = set()
    for dataset_data in all_data.values():
        all_metrics.update(dataset_data.keys())


    for metric in all_metrics:

        has_significant = False
        significant_datasets = []
        
        for dataset_name, dataset_data in all_data.items():
            if metric in dataset_data:
                try:
                    anova_results = repeated_measures_anova(dataset_data[metric])
                    if anova_results['p_value'] < 0.05:
                        has_significant = True
                        significant_datasets.append((dataset_name, dataset_data[metric]))
                    else:
                        print(f"Non-significant result for {dataset_name} - {metric}: p = {anova_results['p_value']:.4f}")
                except Exception as e:
                    print(f"Error with {dataset_name} - {metric}: {e}")
                    continue
        
        if has_significant:
            print(f"Significant differences found for {metric} with p-value {anova_results['p_value']}")
            

            num_plots = min(len(significant_datasets), 3)
            

            fig, axes = plt.subplots(1, num_plots, figsize=(7.5, 2.5), sharey=True)
            

            if num_plots == 1:
                axes = [axes]
            

            for i, (dataset_name, metric_data) in enumerate(significant_datasets[:3]):

                best_model = find_best_model(metric_data, metric)
                tukey_results = tukey_hsd(metric_data)
                

                tukey_results['results'].plot_simultaneous(
                    comparison_name=best_model, 
                    ax=axes[i]
                )
                axes[i].set_title(dataset_name, fontsize=10, color='#666666')
                axes[i].set_xlabel(metric.upper(), fontsize=10, color='#666666')
                

                axes[i].tick_params(axis='x', labelsize=8, colors='#666666')
                
                if i == 0:
                    axes[i].tick_params(axis='y', labelsize=8, colors='#333333')
                else:
                    axes[i].tick_params(axis='y', labelsize=8, colors='#CCCCCC')
                

                for spine in axes[i].spines.values():
                    spine.set_color('#CCCCCC')
                    spine.set_linewidth(0.8)
                

                yticks = axes[i].get_yticks()
                if len(yticks) > 2:
                    new_yticks = yticks[1:-1]
                    axes[i].set_yticks(new_yticks)
            

            fig.text(0.06, 0.5, y_axis_label, va='center', rotation='vertical', 
                    fontsize=11, color='#666666')
            
            plt.tight_layout()
            plt.subplots_adjust(left=0.12)
            

            if output_dir is not None:
                base_output_path = Path(output_dir) / f'{metric}_comparison'
                

                png_path = base_output_path.with_suffix('.png')
                plt.savefig(png_path, dpi=300, bbox_inches='tight')
                print(f"Saved PNG plot for {metric} to {png_path}")
                

                pdf_path = base_output_path.with_suffix('.pdf')
                plt.savefig(pdf_path, bbox_inches='tight')
                print(f"Saved PDF plot for {metric} to {pdf_path}")
            
            plt.show()
        else:
            print(f"No significant differences found for {metric}")


def analyze_and_plot_five_experiments(
    csv_paths: List[str], 
    model_name_mapping: Dict[str, str], 
    output_dir: Optional[str] = None,
    dataset_name_transform: callable = None,
    y_axis_label: str = 'Masking Ratio'
) -> None:
    """
    Analyze experimental results from exactly 5 CSV files and create plots in a 3+2 layout.
    
    Args:
        csv_paths: List of exactly 5 paths to CSV files containing experimental results
        model_name_mapping: Dictionary mapping full model names to short display names
        output_dir: Optional directory to save the generated plots. If None, plots are only displayed.
        dataset_name_transform: Optional function to transform dataset names from file paths
        y_axis_label: Label for the y-axis (default: 'Masking Ratio')
    """
    if len(csv_paths) != 5:
        raise ValueError(f"This function requires exactly 5 CSV files, but {len(csv_paths)} were provided.")
    

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    

    all_data = {}
    for csv_path in csv_paths:

        if dataset_name_transform:
            dataset_name = dataset_name_transform(csv_path)
        else:

            dataset_name = (Path(csv_path).stem
                          .replace('_results', '')
                          .replace('adme-fang-', '')
                          .replace('-1', '')
                          .replace("hclint", "HCLint")
                          .replace("perm", "Permeability")
                          .replace("solu", "Solubility")
                          .replace("-astrazeneca", "")
                          .replace("adme-novartis-", "")
                          .replace("-reg", "")
                          .replace("PERM", "Permeability")
                          .replace("SOLU", "Solubility")
                          .replace("HCLINT", "HCLint")
                          .replace("lipophilicity", "Lipophilicity")
                          .replace("cyp3a4", "CYP"))
            
        
        prepared_data = load_and_prepare_data(csv_path, model_name_mapping)
        cleaned_data = clean_prepared_data(prepared_data)
        all_data[dataset_name] = cleaned_data


    all_metrics = set()
    for dataset_data in all_data.values():
        all_metrics.update(dataset_data.keys())

    # Create plots for each metric
    for metric in all_metrics:

        has_significant = False
        significant_datasets = []
        
        for dataset_name, dataset_data in all_data.items():
            if metric in dataset_data:
                try:
                    anova_results = repeated_measures_anova(dataset_data[metric])
                    if anova_results['p_value'] < 0.05:
                        has_significant = True
                        significant_datasets.append((dataset_name, dataset_data[metric]))
                    else:
                        print(f"Non-significant result for {dataset_name} - {metric}: p = {anova_results['p_value']:.4f}")
                except Exception as e:
                    print(f"Error with {dataset_name} - {metric}: {e}")
                    continue
        
        if has_significant:
            print(f"Significant differences found for {metric} with p-value {anova_results['p_value']}")
            

            all_datasets = [(name, data[metric]) for name, data in all_data.items() if metric in data]
            
            # Create figure with 2x3 subplot layout (3 top, 2 bottom)
            fig, axes = plt.subplots(2, 3, figsize=(11.25, 5), sharey=True)
            

            axes_flat = axes.flatten()
            

            for i, (dataset_name, metric_data) in enumerate(all_datasets):

                best_model = find_best_model(metric_data, metric)
                tukey_results = tukey_hsd(metric_data)
                

                tukey_results['results'].plot_simultaneous(
                    comparison_name=best_model, 
                    ax=axes_flat[i]
                )
                axes_flat[i].set_title(dataset_name, fontsize=10, color='#666666')
                axes_flat[i].set_xlabel(metric.upper(), fontsize=10, color='#666666')
                

                axes_flat[i].tick_params(axis='x', labelsize=8, colors='#666666')
                
                # Only leftmost subplots get darker y-axis labels
                if i % 3 == 0:
                    axes_flat[i].tick_params(axis='y', labelsize=8, colors='#333333')
                else:
                    axes_flat[i].tick_params(axis='y', labelsize=8, colors='#CCCCCC')
                

                for spine in axes_flat[i].spines.values():
                    spine.set_color('#CCCCCC')
                    spine.set_linewidth(0.8)
                

                yticks = axes_flat[i].get_yticks()
                if len(yticks) > 2:
                    new_yticks = yticks[1:-1]
                    axes_flat[i].set_yticks(new_yticks)
            

            axes_flat[5].set_visible(False)
            


            fig.text(0.08, 0.75, y_axis_label, va='center', rotation='vertical', 
                    fontsize=11, color='#666666')
            

            fig.text(0.08, 0.25, y_axis_label, va='center', rotation='vertical', 
                    fontsize=11, color='#666666')
            
            plt.tight_layout()
            plt.subplots_adjust(left=0.15)
            

            if output_dir is not None:
                base_output_path = Path(output_dir) / f'{metric}_five_datasets_comparison'
                

                png_path = base_output_path.with_suffix('.png')
                plt.savefig(png_path, dpi=300, bbox_inches='tight')
                print(f"Saved PNG plot for {metric} to {png_path}")
                

                pdf_path = base_output_path.with_suffix('.pdf')
                plt.savefig(pdf_path, bbox_inches='tight')
                print(f"Saved PDF plot for {metric} to {pdf_path}")
            
            plt.show()
        else:
            print(f"No significant differences found for {metric}")


def analyze_and_plot_nine_experiments(
    csv_paths: List[str], 
    model_name_mapping: Dict[str, str], 
    output_dir: Optional[str] = None,
    dataset_name_transform: callable = None,
    y_axis_label: str = 'Masking Ratio'
) -> None:
    """
    Analyze experimental results from exactly 9 CSV files and create plots in a 3x3 layout.
    
    Args:
        csv_paths: List of exactly 9 paths to CSV files containing experimental results
        model_name_mapping: Dictionary mapping full model names to short display names
        output_dir: Optional directory to save the generated plots. If None, plots are only displayed.
        dataset_name_transform: Optional function to transform dataset names from file paths
        y_axis_label: Label for the y-axis (default: 'Masking Ratio')
    """
    if len(csv_paths) != 9:
        raise ValueError(f"This function requires exactly 9 CSV files, but {len(csv_paths)} were provided.")
    

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    

    all_data = {}
    for csv_path in csv_paths:

        if dataset_name_transform:
            dataset_name = dataset_name_transform(csv_path)
        else:

            dataset_name = (Path(csv_path).stem
                          .replace('_results', '')
                          .replace('adme-fang-', '')
                          .replace('-1', '')
                          .replace("hclint", "HCLint")
                          .replace("perm", "Permeability")
                          .replace("solu", "Solubility")
                          .replace("-astrazeneca", "")
                          .replace("adme-novartis-", "")
                          .replace("-reg", "")
                          .replace("PERM", "Permeability")
                          .replace("SOLU", "Solubility")
                          .replace("HCLINT", "HCLint")
                          .replace("lipophilicity", "Lipophilicity")
                          .replace("cyp3a4", "CYP"))
            
        
        prepared_data = load_and_prepare_data(csv_path, model_name_mapping)
        cleaned_data = clean_prepared_data(prepared_data)
        all_data[dataset_name] = cleaned_data


    all_metrics = set()
    for dataset_data in all_data.values():
        all_metrics.update(dataset_data.keys())

    # Create plots for each metric
    for metric in all_metrics:

        has_significant = False
        significant_datasets = []
        
        for dataset_name, dataset_data in all_data.items():
            if metric in dataset_data:
                try:
                    anova_results = repeated_measures_anova(dataset_data[metric])
                    if anova_results['p_value'] < 0.05:
                        has_significant = True
                        significant_datasets.append((dataset_name, dataset_data[metric]))
                    else:
                        print(f"Non-significant result for {dataset_name} - {metric}: p = {anova_results['p_value']:.4f}")
                except Exception as e:
                    print(f"Error with {dataset_name} - {metric}: {e}")
                    continue
        
        if has_significant:
            print(f"Significant differences found for {metric} with p-value {anova_results['p_value']}")
            

            all_datasets = [(name, data[metric]) for name, data in all_data.items() if metric in data]
            
            # Create figure with 3x3 subplot layout
            fig, axes = plt.subplots(3, 3, figsize=(16.875, 7.5), sharey=True)
            

            axes_flat = axes.flatten()
            

            for i, (dataset_name, metric_data) in enumerate(all_datasets):

                best_model = find_best_model(metric_data, metric)
                tukey_results = tukey_hsd(metric_data)
                

                tukey_results['results'].plot_simultaneous(
                    comparison_name=best_model, 
                    ax=axes_flat[i]
                )
                axes_flat[i].set_title(dataset_name, fontsize=10, color='#666666')
                axes_flat[i].set_xlabel(metric.upper(), fontsize=10, color='#666666')
                

                axes_flat[i].tick_params(axis='x', labelsize=8, colors='#666666')
                
                # Only leftmost subplots get darker y-axis labels
                if i % 3 == 0:
                    axes_flat[i].tick_params(axis='y', labelsize=8, colors='#333333')
                else:
                    axes_flat[i].tick_params(axis='y', labelsize=8, colors='#CCCCCC')
                

                for spine in axes_flat[i].spines.values():
                    spine.set_color('#CCCCCC')
                    spine.set_linewidth(0.8)
                

                yticks = axes_flat[i].get_yticks()
                if len(yticks) > 2:
                    new_yticks = yticks[1:-1]
                    axes_flat[i].set_yticks(new_yticks)
            


            fig.text(0.08, 0.83, y_axis_label, va='center', rotation='vertical', 
                    fontsize=11, color='#666666')
            

            fig.text(0.08, 0.5, y_axis_label, va='center', rotation='vertical', 
                    fontsize=11, color='#666666')
            

            fig.text(0.08, 0.17, y_axis_label, va='center', rotation='vertical', 
                    fontsize=11, color='#666666')
            
            plt.tight_layout()
            plt.subplots_adjust(left=0.15)
            

            if output_dir is not None:
                base_output_path = Path(output_dir) / f'{metric}_nine_datasets_comparison'
                

                png_path = base_output_path.with_suffix('.png')
                plt.savefig(png_path, dpi=300, bbox_inches='tight')
                print(f"Saved PNG plot for {metric} to {png_path}")
                

                pdf_path = base_output_path.with_suffix('.pdf')
                plt.savefig(pdf_path, bbox_inches='tight')
                print(f"Saved PDF plot for {metric} to {pdf_path}")
            
            plt.show()
        else:
            print(f"No significant differences found for {metric}")


def analyze_experiments_simple(
    csv_paths: List[str], 
    model_name_mapping: Dict[str, str], 
    output_dir: str = None
) -> Dict[str, Dict[str, Any]]:
    """
    Simple analysis function that returns results without plotting.
    
    Args:
        csv_paths: List of paths to CSV files containing experimental results
        model_name_mapping: Dictionary mapping full model names to short display names
        output_dir: Optional directory to save summary results
        
    Returns:
        Dictionary containing analysis results for each dataset and metric
    """
    results = {}
    
    for csv_path in csv_paths:

        dataset_name = (Path(csv_path).stem
                       .replace('_results', '')
                       .replace('adme-fang-', '')
                       .replace('-1', '')
                       .replace("hclint", "HClint")
                       .replace("perm", "Perm")
                       .replace("solu", "Solu"))
        
        prepared_data = load_and_prepare_data(csv_path, model_name_mapping)
        cleaned_data = clean_prepared_data(prepared_data)
        
        results[dataset_name] = {}
        
        for metric, data in cleaned_data.items():
            try:
                anova_results = repeated_measures_anova(data)
                best_model = find_best_model(data, metric)
                
                results[dataset_name][metric] = {
                    'p_value': anova_results['p_value'],
                    'significant': anova_results['p_value'] < 0.05,
                    'best_model': best_model,
                    'data_shape': data.shape
                }
                
                if anova_results['p_value'] < 0.05:
                    tukey_results = tukey_hsd(data)
                    results[dataset_name][metric]['tukey_results'] = tukey_results
                    
            except Exception as e:
                results[dataset_name][metric] = {
                    'error': str(e)
                }
    
    return results


def heatmap_performance(
    csv_paths: List[str], 
    model_name_mapping: Dict[str, str], 
    output_dir: Optional[str] = None,
    dataset_name_transform: callable = None,
    figsize: Tuple[float, float] = (12, 8)
) -> None:
    """
    Create heatmap visualizations showing model performance organized by model size and dataset size.
    
    Args:
        csv_paths: List of paths to CSV files containing experimental results
        model_name_mapping: Dictionary mapping full model names to short display names
        output_dir: Optional directory to save the generated plots. If None, plots are only displayed.
        dataset_name_transform: Optional function to transform dataset names from file paths
        figsize: Figure size for the heatmap (width, height)
    """

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    

    def categorize_model(model_name: str) -> Tuple[str, str]:
        """Extract model size and dataset size from model name."""
        # Check for model size (15M check must be before 5M to avoid conflicts)
        if "111M" in model_name:
            model_size = "111M"
        elif "15M" in model_name:
            model_size = "15M"
        elif "5M" in model_name:
            model_size = "5M"
        else:
            raise ValueError(f"Unknown model size in: {model_name}")
        
        if "baseline" in model_name or "no pretraining" in model_name:
            dataset_size = "No Pretraining"
        elif "half of chembl" in model_name:
            dataset_size = "Half ChEMBL"
        elif "pubchem + chembl" in model_name:
            dataset_size = "PubChem"
        elif "chembl" in model_name and "half of chembl" not in model_name and "pubchem + chembl" not in model_name:
            dataset_size = "ChEMBL"
        else:
            raise ValueError(f"Unknown dataset size in: {model_name}")
        
        return model_size, dataset_size
    

    all_data = {}
    for csv_path in csv_paths:

        if dataset_name_transform:
            dataset_name = dataset_name_transform(csv_path)
        else:

            dataset_name = (Path(csv_path).stem
                          .replace('_results', '')
                          .replace('adme-fang-', '')
                          .replace('-1', '')
                          .replace("hclint", "HCLint")
                          .replace("perm", "Permeability")
                          .replace("solu", "Solubility")
                          .replace("-astrazeneca", "")
                          .replace("adme-novartis-", "")
                          .replace("-reg", "")
                          .replace("PERM", "Permeability")
                          .replace("SOLU", "Solubility")
                          .replace("HCLINT", "HCLint")
                          .replace("lipophilicity", "Lipophilicity")
                          .replace("cyp3a4", "CYP"))
        
        prepared_data = load_and_prepare_data(csv_path, model_name_mapping)
        cleaned_data = clean_prepared_data(prepared_data)
        all_data[dataset_name] = cleaned_data


    all_metrics = set()
    for dataset_data in all_data.values():
        all_metrics.update(dataset_data.keys())


    for metric in all_metrics:

        metric_datasets = []
        for dataset_name, dataset_data in all_data.items():
            if metric in dataset_data:
                metric_datasets.append((dataset_name, dataset_data[metric]))
        
        if not metric_datasets:
            continue
            
        
        n_datasets = len(metric_datasets)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        
        if n_datasets == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (dataset_name, metric_data) in enumerate(metric_datasets):

            anova_results = repeated_measures_anova(metric_data)
            is_significant = anova_results['p_value'] < 0.05
            

            if not is_significant:
                raise ValueError(f"No significant differences found for {dataset_name} - {metric}. "
                               f"ANOVA p-value: {anova_results['p_value']:.4f}")
            
            tukey_results = tukey_hsd(metric_data)
            tukey_obj = tukey_results['results']  # TukeyHSDResults object
            

            best_model = find_best_model(metric_data, metric)
            

            model_stats = {}
            model_names = list(metric_data.columns)
            
            for model_name in model_names:
                values = metric_data[model_name].dropna()
                model_stats[model_name] = {
                    'mean': values.mean(),
                    'values': values
                }
            

            group_means = {}
            confidence_intervals = {}
            

            for i, group_name in enumerate(tukey_obj.groupsunique):
                if group_name not in model_stats:
                    raise ValueError(f"Group {group_name} from Tukey results not found in model_stats")
                    

                ci_width = (tukey_obj.confint[i, 1] - tukey_obj.confint[i, 0]) / 2
                confidence_intervals[group_name] = ci_width
                    

            for model_name in model_stats:
                if model_name not in confidence_intervals:
                    raise ValueError(f"Model {model_name} not found in Tukey confidence intervals")
                model_stats[model_name]['ci_95'] = confidence_intervals[model_name]
            

            model_sizes = []
            dataset_sizes = []
            for model_name in metric_data.columns:
                size, ds_size = categorize_model(model_name)
                model_sizes.append(size)
                dataset_sizes.append(ds_size)
            

            unique_model_sizes = sorted(list(set(model_sizes)), key=lambda x: {"5M": 0, "15M": 1, "111M": 2}.get(x, 3))
            unique_dataset_sizes = ["No Pretraining", "Half ChEMBL", "ChEMBL", "PubChem"]
            unique_dataset_sizes = [ds for ds in unique_dataset_sizes if ds in dataset_sizes]
            

            heatmap_values = np.full((len(unique_model_sizes), len(unique_dataset_sizes)), np.nan)
            heatmap_ci = np.full((len(unique_model_sizes), len(unique_dataset_sizes)), np.nan)
            heatmap_significance = np.full((len(unique_model_sizes), len(unique_dataset_sizes)), 0)
            
            for model_name in metric_data.columns:
                size, ds_size = categorize_model(model_name)
                if size in unique_model_sizes and ds_size in unique_dataset_sizes:
                    i = unique_model_sizes.index(size)
                    j = unique_dataset_sizes.index(ds_size)
                    
                    heatmap_values[i, j] = model_stats[model_name]['mean']
                    heatmap_ci[i, j] = model_stats[model_name]['ci_95']
                    

                    if model_name == best_model:
                        heatmap_significance[i, j] = 2
                    else:

                        summary_df = tukey_results['summary_table']
                        

                        comparison_rows = summary_df[
                            ((summary_df['group1'] == model_name) & (summary_df['group2'] == best_model)) |
                            ((summary_df['group1'] == best_model) & (summary_df['group2'] == model_name))
                        ]
                        
                        if len(comparison_rows) == 0:
                            raise ValueError(f"No pairwise comparison found between {model_name} and {best_model}")
                        
                        p_val = comparison_rows.iloc[0]['p-adj']
                        if p_val < 0.05:
                            if metric in ['mae', 'mse']:
                                if model_stats[model_name]['mean'] > model_stats[best_model]['mean']:
                                    heatmap_significance[i, j] = -1
                                else:
                                    heatmap_significance[i, j] = 1
                            else:
                                if model_stats[model_name]['mean'] < model_stats[best_model]['mean']:
                                    heatmap_significance[i, j] = -1
                                else:
                                    heatmap_significance[i, j] = 1
                        else:
                            heatmap_significance[i, j] = 1
            

            ax = axes[idx] if n_datasets > 1 else axes[0]
            

            print(f"\n{dataset_name} - {metric.upper()} - 95% Confidence Intervals:")
            for model_name in sorted(model_stats.keys()):
                ci_val = model_stats[model_name]['ci_95']
                mean_val = model_stats[model_name]['mean']
                print(f"  {model_name}: {mean_val:.3f} ± {ci_val:.3f}")
            

            mask = np.isnan(heatmap_values)
            sns.heatmap(np.ones_like(heatmap_values), 
                       xticklabels=unique_dataset_sizes,
                       yticklabels=unique_model_sizes,
                       annot=False,
                       cmap='Greys',
                       vmin=0, vmax=1,
                       mask=mask,
                       ax=ax,
                       cbar=False,
                       square=False)
            

            ax.set_aspect(4/3)
            

            for i in range(len(unique_model_sizes)):
                for j in range(len(unique_dataset_sizes)):
                    if not np.isnan(heatmap_values[i, j]):
                        mean_val = heatmap_values[i, j]
                        sig_val = heatmap_significance[i, j]
                        
                        if sig_val == 2:
                            bg_color = 'darkblue'
                        elif sig_val == -1:
                            bg_color = 'brown'
                        elif sig_val == 1:
                            bg_color = '#808080'
                        else:
                            bg_color = '#CCCCCC' 
                        

                        rect = plt.Rectangle((j, i), 1, 1, facecolor=bg_color, edgecolor='white', linewidth=1)
                        ax.add_patch(rect)
                        

                        text = f'{mean_val:.3f}'
                        ax.text(j + 0.5, i + 0.5, text,
                               horizontalalignment='center',
                               verticalalignment='center',
                               color='white',
                               weight='normal',
                               fontsize=8)
            
            ax.set_title(f'{dataset_name} - {metric.upper()}', fontsize=12, color='#666666')
            ax.set_xlabel('Pretraining Dataset Size', fontsize=10, color='#666666')
            ax.set_ylabel('Model Size', fontsize=10, color='#666666')
            

            ax.tick_params(axis='x', rotation=45, labelsize=8, colors='#333333')
            ax.tick_params(axis='y', rotation=0, labelsize=8, colors='#333333')
        

        for idx in range(len(metric_datasets), len(axes)):
            if idx == 5 and len(axes) > 5:
                axes[idx].axis('off')
                legend_text = "Blue = Best model\nGrey = Not significantly worse\nRed = Significantly worse"
                axes[idx].text(0.5, 0.5, legend_text, 
                             horizontalalignment='center', 
                             verticalalignment='center',
                             transform=axes[idx].transAxes,
                             fontsize=11, 
                             color='#666666',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            else:
                axes[idx].set_visible(False)
        

        plt.tight_layout(pad=0.2)
        plt.subplots_adjust(wspace=0.05, hspace=0.7)
        

        if output_dir is not None:
            base_output_path = Path(output_dir) / f'{metric}_heatmap_comparison'
            
            png_path = base_output_path.with_suffix('.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Saved PNG heatmap for {metric} to {png_path}")
            
            pdf_path = base_output_path.with_suffix('.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Saved PDF heatmap for {metric} to {pdf_path}")
        
        plt.show() 