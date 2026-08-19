import pandas as pd

def binaryEncoder(s: pd.Series) -> pd.Series:

    '''
    Finds any columns with len(2) or binary columns, and turns their values into integers.

    :param s: Telco Dataset dataframe
    :return s: Updated dataframe
    '''

    # get unique values and remove NAN
    vals = list(pd.Series(s.dropna().unique()).astype(str))

    if len(vals) == 2:

        print(f"Binary Encoding {len(vals)} values")
        print(f"Old Shape {s.shape}")

        # Sort values to ensure consistent mapping across runs
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}

        print(f"Binary Encoding Complete")
        print(f"New Shape {s.shape}")

        return s.astype(str).map(mapping).astype("Int64")



    # Return unchanged - will be handled by one-hot encoding
    return s


def oneHotEncoder(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    '''
    Finds columns with len>2 and turns their values into integers.

    :param df, columns: Telco Dataset dataFrame and affected columns
    :return df: Updated dataFrame
    '''

    # get unique values and remove NAN
    if not columns:
        return df

    print(f"One-hot encoding columns: {columns}")
    print(f"Old shape: {df.shape}")

    dummies = pd.get_dummies(df[columns], prefix=columns, drop_first=True)
    df = df.drop(columns=columns).join(dummies)

    print(f"New shape: {df.shape}")
    return df


def build_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.

    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    """

    df = df.copy()

    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]  # remove NAN, get unique cols that have len>2
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2] # same thing but len==2

    # Binary encoding
    for c in binary_cols:
        print(f"Binary Encoding Started")
        df[c] = binaryEncoder(df[c])

    # One-hot encoding
    print(f"One Hot Encoding Started")
    df = oneHotEncoder(df, multi_cols)

    # Convert booleans to int
    print(f"Converting booleans to integers")
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Fill NaNs
    print(f"Filling missing values")
    df = df.fillna(0)

    print(f" Feature engineering complete: {df.shape[1]} final features")
    return df