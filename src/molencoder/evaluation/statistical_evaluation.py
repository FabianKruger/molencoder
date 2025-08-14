import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Dict, Any


def repeated_measures_anova(
    data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Perform a repeated measures ANOVA on model comparison data.
    
    This function analyzes whether there are statistically significant differences
    between different models based on their performance metrics. It uses a repeated measures
    ANOVA, which is appropriate for this type of data where the same measurements
    are used for all models.
    
    Args:
        data: DataFrame with model names as columns and measurements as rows
        
    Returns:
        Dictionary containing:
        - 'f_statistic': The F-statistic from the ANOVA
        - 'p_value': The p-value from the ANOVA
        - 'anova_table': The complete ANOVA table as a pandas DataFrame
    """
    # Create a long-format DataFrame for AnovaRM
    # This format has columns: 'subject', 'model', 'value'
    long_data = []
    for idx, row in data.iterrows():
        for model_name in data.columns:
            long_data.append({
                'subject': idx,
                'model': model_name,
                'value': row[model_name]
            })
    
    df = pd.DataFrame(long_data)
    
    # Perform the repeated measures ANOVA using AnovaRM
    # According to the documentation, we need to specify:
    # - data: the DataFrame
    # - depvar: the dependent variable (the metric values)
    # - subject: the subject identifier (in our case, the measurement index)
    # - within: the within-subject factor (in our case, the model)
    aovrm = AnovaRM(data=df, depvar='value', subject='subject', within=['model'])
    anova_results = aovrm.fit()
    
    # Extract F-statistic and p-value
    f_stat = anova_results.anova_table['F Value']['model']
    p_value = anova_results.anova_table['Pr > F']['model']
    
    # Return results as a dictionary
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'anova_table': anova_results.anova_table
    }


def tukey_hsd(
    data: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform Tukey's Honestly Significant Difference (HSD) test on model comparison data.
    
    This function performs pairwise comparisons between all models using Tukey's HSD test,
    which controls for the familywise error rate. It's useful for determining which specific
    models differ significantly from each other after a significant ANOVA result.
    
    Args:
        data: DataFrame with model names as columns and measurements as rows
        alpha: Significance level for the test (default: 0.05)
        
    Returns:
        Dictionary containing:
        - 'results': The TukeyHSDResults object containing all pairwise comparisons
        - 'summary_table': A DataFrame with the summary of all pairwise comparisons
    """
    # Create a long-format DataFrame for Tukey's HSD test
    long_data = []
    for _, row in data.iterrows():
        for model_name in data.columns:
            long_data.append({
                'model': model_name,
                'value': row[model_name]
            })
    
    df = pd.DataFrame(long_data)
    
    # Perform Tukey's HSD test
    # - endog: the dependent variable (the metric values)
    # - groups: the grouping variable (in our case, the model)
    # - alpha: the significance level
    tukey_results = pairwise_tukeyhsd(
        endog=df['value'],
        groups=df['model'],
        alpha=alpha
    )
    
    summary_table = tukey_results.summary()
    
    # Convert the summary table to a pandas DataFrame
    column_names = summary_table.data[0]
    data_rows = summary_table.data[1:]
    df_summary = pd.DataFrame(data_rows, columns=column_names)
    
    return {
        'results': tukey_results,
        'summary_table': df_summary,
    } 