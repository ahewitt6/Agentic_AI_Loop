import pandas as pd

#Raw Data -> Features -> Candidates -> Evaluation -> Acceptance -> Validation
#useful transformations to raw data
def add_rolling_mean(df, column, window):
    res = df.copy()
    res = res.sort_values(["ticker", "date"])
    res[f"{column}_{window}d_mean"] = res.groupby("ticker")[column].transform(lambda x: x.rolling(window).mean())
    return res

def add_rolling_std(df, column, window):
    res = df.copy()
    res = res.sort_values(["ticker","date"])
    res[f"{column}_{window}d_std"] = res.groupby("ticker")[column].transform(lambda x: x.rolling(window).std())
    return res

def add_relative_volume(df, window=20):
    res = df.copy()
    res = res.sort_values(["ticker","date"])
    res["vol_rolling_avg"] = res.groupby("ticker")["volume"].transform(lambda x: x.shift(1).rolling(window).mean())
    res["relative_volume"] = res["volume"]/res["vol_rolling_avg"]
    return res

def add_momentum(df, window=20):
    res = df.copy()
    res = res.sort_values(["ticker", "date"])
    res[f"momentum_{window}d"] = res.groupby("ticker")["close"].transform(lambda x: x/x.shift(window)-1)
    return res

def add_rolling_volatility(df, window=20):
    res = df.copy()
    res = res.sort_values(["ticker", "date"])
    res[f"volatility_{window}d"] = res.groupby("ticker")["return_1d"].transform(lambda x: x.rolling(window).std())
    return res

def add_cross_sectional_rank(df, column):
    res = df.copy()
    res = res.sort_values(["ticker", "date"])
    res[f"{column}_rank"] = res.groupby("date")[column].rank(pct = True, ascending = True)
    return res

def add_cross_sectional_zscore(df, column):
    res = df.copy()
    res = res.sort_values(["ticker", "date"])
    res[f"{column}_z"] = res.groupby("date")[column].transform(lambda x: (x-x.mean())/x.std() if x.std()!=0 else 0)
    return res

#helper verification function
def validate_columns(df, required_columns):
    missing_cols = []
    for col in required_columns:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        raise ValueError(f"Required columns: {missing_cols} are missing from the dataframe")

#build features using the transformations on raw data
def build_features(df,window):
    #check required columns exist
    required_columns = ["date", "ticker", "close", "volume", "return_1d"]
    validate_columns(df, required_columns)

    res = add_rolling_mean(df, "volume", window)
    res = add_rolling_std(res, "volume", window)
    res = add_relative_volume(res, window)
    res = add_momentum(res, window)
    res = add_rolling_volatility(res, window)

    momentum_col = f"momentum_{window}d"
    
    res = add_cross_sectional_rank(res, momentum_col)
    res = add_cross_sectional_zscore(res, momentum_col)

    return res

    
