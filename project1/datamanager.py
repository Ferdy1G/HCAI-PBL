import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from django.conf import settings

def upload_parsing(dataset_instance):
    """
    Parses the uploaded dataset model instance, saves a Parquet working copy,
    and returns parsed metadata.
    """
    error = None
    numeric_cols, non_numeric_cols, averages, class_counts = None, None, None, None

    try:
        df = pd.read_csv(dataset_instance.original_file.path)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        averages = df[numeric_cols].mean().to_dict()
        class_counts = df[non_numeric_cols].nunique().to_dict()

        # save working copy
        csv_path = Path(dataset_instance.original_file.path)
        working_parquet_path = str(csv_path.with_suffix('.parquet'))
        df.to_parquet(working_parquet_path)
        print(df.describe())

        # Update model
        dataset_instance.working_file_path = working_parquet_path
        dataset_instance.save()

    except Exception as e:
        error = "Error processing file: " + str(e)

    return numeric_cols, non_numeric_cols, averages, class_counts, error


def get_dataset_summary(df):
    """
    Generates summary metrics for numerical and non-numerical columns.
    Returns a list of dicts for easy rendering in a single Django template table.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    summary_data = []

    # Process Numerical Columns
    for col in numeric_cols:
        series = df[col]
        summary_data.append({
            'name': col,
            'type': 'Numerical',
            'mean_or_top': round(series.mean(), 2) if not series.dropna().empty else 'N/A',
            'min': round(series.min(), 2) if not series.dropna().empty else 'N/A',
            'median': round(series.median(), 2) if not series.dropna().empty else 'N/A',
            'max': round(series.max(), 2) if not series.dropna().empty else 'N/A',
            'invalid_count': series.isna().sum(),
        })

    # Process Non-Numerical Columns
    for col in non_numeric_cols:
        series = df[col]
        # Most frequent value (mode)
        top_val = series.mode().iloc[0] if not series.mode().empty else 'N/A'
        
        summary_data.append({
            'name': col,
            'type': 'Non-Numerical',
            'mean_or_top': top_val,
            'min': '-',
            'median': '-',
            'max': '-',
            'invalid_count': series.isna().sum(),
        })

    return summary_data

def generate_distribution_plot(df, dataset_id, column):
    """ Generates a single distribution plot on demand """
    if not column or column not in df.columns:
        return None

    plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots', str(dataset_id))
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    if pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(df[column].dropna(), kde=True, ax=ax, color="#3b82f6")
    else:
        top_counts = df[column].value_counts().head(10)
        sns.barplot(x=top_counts.values, y=top_counts.index.astype(str), ax=ax, palette="Blues_r")

    ax.set_title(f"Distribution of {column}", fontsize=11)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    plt.tight_layout()

    file_name = f"dist_{column}.png"
    file_path = os.path.join(plots_dir, file_name)
    plt.savefig(file_path, dpi=120)
    plt.close(fig)

    return f"{settings.MEDIA_URL}plots/{dataset_id}/{file_name}"

def generate_scatter_plot(df, dataset_id, x_col, y_col, hue_col=None):
    """ Generates a 2D scatter plot with optional color grouping """
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return None

    plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots', str(dataset_id))
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    hue_param = hue_col if (hue_col and hue_col in df.columns) else None

    sns.scatterplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        hue=hue_param, 
        palette="viridis" if hue_param else None,
        alpha=0.8,
        ax=ax
    )

    ax.set_title(f"{x_col} vs {y_col}", fontsize=11)
    plt.tight_layout()

    hue_str = f"_{hue_col}" if hue_param else ""
    file_name = f"scatter_{x_col}_{y_col}{hue_str}.png"
    file_path = os.path.join(plots_dir, file_name)
    plt.savefig(file_path, dpi=120)
    plt.close(fig)

    return f"{settings.MEDIA_URL}plots/{dataset_id}/{file_name}"

def reset_dataset(dataset_instance):
    """Reloads the original CSV file and overwrites the Parquet file."""
    try:
        df = pd.read_csv(dataset_instance.original_file.path)
        csv_path = Path(dataset_instance.original_file.path)
        working_parquet_path = str(csv_path.with_suffix('.parquet'))
        df.to_parquet(working_parquet_path)
        
        dataset_instance.working_file_path = working_parquet_path
        dataset_instance.save()
        return df, None
    except Exception as e:
        return None, f"Failed to reset dataset: {str(e)}"


def drop_missing_rows(dataset_instance):
    """Drops all rows containing missing/null values from the working Parquet dataset."""
    try:
        df = pd.read_parquet(dataset_instance.working_file_path)
        df = df.dropna()
        df.to_parquet(dataset_instance.working_file_path)
        return df, None
    except Exception as e:
        return None, f"Failed to drop missing rows: {str(e)}"


def impute_missing_values(dataset_instance):
    """
    Fills missing values:
    - Numerical columns with mean value.
    - Non-numerical columns with mode (most frequent) value.
    """
    try:
        df = pd.read_parquet(dataset_instance.working_file_path)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

        # Fill numeric NaNs with Mean
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)

        # Fill non-numeric NaNs with Mode
        for col in non_numeric_cols:
            if df[col].isna().sum() > 0:
                mode_series = df[col].mode()
                if not mode_series.empty:
                    df[col] = df[col].fillna(mode_series.iloc[0])

        df.to_parquet(dataset_instance.working_file_path)
        return df, None
    except Exception as e:
        return None, f"Failed to impute missing values: {str(e)}"
    

def filter_out_of_distribution(dataset_instance, column, min_value, max_value):
    """Filters out rows in the specified column that are outside the given min/max range."""
    try:
        if not column:
            return None, "No column selected for filtering."

        df = pd.read_parquet(dataset_instance.working_file_path)

        # Convert column to numeric if needed
        if column in df.columns:
            # Parse string inputs to numbers
            has_min = min_value is not None and min_value != ""
            has_max = max_value is not None and max_value != ""

            if has_min:
                min_val = float(min_value)
                df = df[df[column] >= min_val]

            if has_max:
                max_val = float(max_value)
                df = df[df[column] <= max_val]

            # Save filtered dataset back to Parquet
            df.to_parquet(dataset_instance.working_file_path)
            return df, None
        else:
            return None, f"Column '{column}' not found in dataset."

    except ValueError:
        return None, "Invalid numerical value provided for min/max filter."
    except Exception as e:
        return None, f"Failed to filter out of distribution: {str(e)}"