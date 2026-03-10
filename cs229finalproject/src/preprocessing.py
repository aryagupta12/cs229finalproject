"""
Data preprocessing pipeline for recidivism prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional

TARGET_COLS = [
    'Recidivism_Within_3years',
    'Recidivism_Arrest_Year1',
    'Recidivism_Arrest_Year2',
    'Recidivism_Arrest_Year3',
]

CATEGORICAL_COLS = [
    'Gender', 'Race', 'Age_at_Release', 'Gang_Affiliated',
    'Supervision_Level_First', 'Education_Level', 'Dependents',
    'Prison_Offense', 'Prison_Years', 'Employment_Exempt',
    'Prior_Revocations_Parole', 'Prior_Revocations_Probation',
    'Condition_MH_SA', 'Condition_Cog_Ed', 'Condition_Other',
    'Prior_Arrest_Episodes_Felony', 'Prior_Arrest_Episodes_Misd',
    'Prior_Arrest_Episodes_Violent', 'Prior_Arrest_Episodes_Property',
    'Prior_Arrest_Episodes_Drug', 'Prior_Arrest_Episodes_DVCharges',
    'Prior_Arrest_Episodes_GunCharges',
    'Prior_Conviction_Episodes_Felony', 'Prior_Conviction_Episodes_Misd',
    'Prior_Conviction_Episodes_Viol', 'Prior_Conviction_Episodes_Prop',
    'Prior_Conviction_Episodes_Drug',
    'Violations_ElectronicMonitoring', 'Violations_Instruction',
    'Violations_FailToReport', 'Violations_MoveWithoutPermission',
    'Delinquency_Reports', 'Program_Attendances',
    'Program_UnexcusedAbsences', 'Residence_Changes',
    '_v1', '_v2', '_v3', '_v4',
]

CONTINUOUS_COLS = [
    'Supervision_Risk_Score_First',
    'Avg_Days_per_DrugTest',
    'DrugTests_THC_Positive', 'DrugTests_Cocaine_Positive',
    'DrugTests_Meth_Positive', 'DrugTests_Other_Positive',
    'Percent_Days_Employed', 'Jobs_Per_Year',
    'Residence_PUMA',
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV file."""
    df = pd.read_csv(filepath)
    return df


def explore_data(df: pd.DataFrame) -> Dict:
    """
    Return summary statistics:
    - Shape
    - Missing values per column
    - Unique values for categorical columns
    - Distribution of target variable
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()

    unique_vals = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            unique_vals[col] = df[col].unique().tolist()

    target_dist = {}
    for t in TARGET_COLS:
        if t in df.columns:
            target_dist[t] = df[t].value_counts(normalize=True).to_dict()

    return {
        'shape': df.shape,
        'missing_values': missing,
        'unique_categorical': unique_vals,
        'target_distribution': target_dist,
    }


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - For numeric columns: impute with median, add binary indicator column
    - For categorical columns: impute with mode or 'Unknown'
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_val)
    return df


def encode_categorical(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    Return the encoded dataframe.
    """
    cols_present = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=False)
    return df


def standardize_continuous(
    df: pd.DataFrame,
    continuous_cols: List[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize continuous columns (zero mean, unit variance).
    If scaler is None, fit a new one. Otherwise, use the provided scaler.
    Return (dataframe, scaler).
    """
    df = df.copy()
    cols_present = [c for c in continuous_cols if c in df.columns]
    if scaler is None:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present].astype(float))
    else:
        df[cols_present] = scaler.transform(df[cols_present].astype(float))
    return df, scaler


def _bool_to_int(series: pd.Series) -> pd.Series:
    """Convert Yes/No, true/false, or bool columns to int (1/0)."""
    if series.dtype == bool:
        return series.astype(int)
    mapping = {
        'Yes': 1, 'No': 0,
        'yes': 1, 'no': 0,
        'true': 1, 'false': 0,
        'True': 1, 'False': 0,
        True: 1, False: 0,
        1: 1, 0: 0,
    }
    return series.map(mapping).astype(int)


def preprocess_pipeline(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: str = 'Recidivism_Within_3years',
) -> Dict:
    """
    Full preprocessing pipeline.

    Returns:
    {
        'X_train': np.ndarray,
        'X_test': np.ndarray,
        'y_train': np.ndarray,
        'y_test': np.ndarray,
        'y_train_year1': np.ndarray,
        'y_train_year2': np.ndarray,
        'y_train_year3': np.ndarray,
        'y_test_year1': np.ndarray,
        'y_test_year2': np.ndarray,
        'y_test_year3': np.ndarray,
        'feature_names': List[str],
        'scaler': StandardScaler,
        'target_col': str
    }
    """
    df = load_data(filepath)

    # Convert bool targets
    for col in TARGET_COLS:
        if col in df.columns:
            df[col] = _bool_to_int(df[col])

    # Separate targets and drop ID
    drop_cols = ['ID'] + TARGET_COLS
    drop_cols = [c for c in drop_cols if c in df.columns]

    y_all = {col: df[col].values for col in TARGET_COLS if col in df.columns}
    df = df.drop(columns=drop_cols)

    # Handle missing values
    df = handle_missing_values(df)

    # Filter categorical cols to those present
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # One-hot encode categoricals
    df = encode_categorical(df, cat_cols)

    # Standardize continuous
    df, scaler = standardize_continuous(df, CONTINUOUS_COLS)

    # Ensure all remaining columns are numeric
    df = df.astype(float)

    X = df.values
    feature_names = list(df.columns)

    y = y_all[target_col]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build year-level label splits aligned to train/test
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    result: Dict = {
        'X_train': X_train.astype(np.float32),
        'X_test': X_test.astype(np.float32),
        'y_train': y_train.astype(np.float32),
        'y_test': y_test.astype(np.float32),
        'feature_names': feature_names,
        'scaler': scaler,
        'target_col': target_col,
    }

    for year_col in ['Recidivism_Arrest_Year1', 'Recidivism_Arrest_Year2', 'Recidivism_Arrest_Year3']:
        suffix = year_col.split('_')[-1].lower()  # year1/year2/year3
        if year_col in y_all:
            yr_arr = y_all[year_col]
            result[f'y_train_{suffix}'] = yr_arr[idx_train].astype(np.float32)
            result[f'y_test_{suffix}'] = yr_arr[idx_test].astype(np.float32)

    return result
