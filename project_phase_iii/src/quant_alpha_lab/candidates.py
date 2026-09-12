import pandas as pd
import pytest
import numpy as np
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#functions for generating different combinations of features as candidate alpha signals
def generate_raw_candidates(df, feature_columns):
    
    candidates = {}

    for feature in feature_columns:
        if feature not in df.columns:
            raise ValueError(f"{feature} not in the dataframe")
        candidates[feature] = df[feature]

    for candidate in candidates.values():
        if not candidate.index.equals(df.index):
            raise ValueError("Candidate index does not match dataframe index")

    return candidates

def generate_sign_reversed_candidates(candidates):

    negative_candidates = {}

    for candidate in candidates:
        negative_candidates[f"negative_{candidate}"] = - candidates[candidate]

    return negative_candidates

def generate_rank_candidates(df, candidates):
    candidates_rank = {}

    for name, signal in candidates.items():
        ranked_signal = signal.groupby(df["date"]).rank(pct = True, ascending = True)
        candidates_rank[f"{name}_rank"] = ranked_signal

    return candidates_rank

def generate_signed_sqrt_candidates(candidates):

    candidates_signed_sqrt = {}

    for name, signal in candidates.items():
        signal_signed_sqrt = np.sign(signal)*np.sqrt(np.abs(signal))
        candidates_signed_sqrt[f"{name}_signed_sqrt"] = signal_signed_sqrt
    
    return candidates_signed_sqrt

def generate_winsorized_candidates(df, candidates, lower=0.05, upper=0.95):

    candidates_winsorized = {}

    for name, signal in candidates.items():
        lower_bound = signal.groupby(df["date"]).transform(lambda x: x.quantile(lower))
        upper_bound = signal.groupby(df["date"]).transform(lambda x: x.quantile(upper))
        winsorized = signal.clip(lower = lower_bound, upper = upper_bound)
        candidates_winsorized[f"{name}_winsorized"] = winsorized

    return candidates_winsorized

def generate_pairwise_interactions(candidates):

    candidates_pairwise = {}

    names = list(candidates.keys())

    for name1, name2 in combinations(names,2):
        candidates_pairwise[f"{name1}_x_{name2}"] = candidates[name1]*candidates[name2]
    
    return candidates_pairwise

def generate_pairwise_differences(candidates):

    candidates_pairwise = {}

    names = list(candidates.keys())

    for name1, name2 in combinations(names,2):
        candidates_pairwise[f"{name1}_minus_{name2}"] = candidates[name1] - candidates[name2]
    
    return candidates_pairwise

def generate_pairwise_ratios(candidates):

    candidates_pairwise = {}

    names = list(candidates.keys())

    for name1, name2 in combinations(names,2):
        safe_denominator = candidates[name2].replace(0,np.nan)
        candidates_pairwise[f"{name1}_div_{name2}"] = candidates[name1]/safe_denominator
    
    return candidates_pairwise

def generate_ridge_candidate(train_df, predict_df, feature_columns, target_column, alpha=1.0):

    train_df = train_df.dropna(subset = feature_columns + [target_column])
    predict_df = predict_df.dropna(subset = feature_columns)

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_predict = predict_df[feature_columns]
    
    model = Ridge(alpha = alpha)
    model.fit(X_train, y_train)

    predictions = model.predict(X_predict)

    return model, predictions

def generate_lasso_candidate(train_df, predict_df, feature_columns, target_column, alpha=1.0):

    train_df = train_df.dropna(subset = feature_columns + [target_column])
    predict_df = predict_df.dropna(subset = feature_columns)

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_predict = predict_df[feature_columns]

    model = Lasso(alpha = alpha)
    model.fit(X_train,y_train)

    predictions = model.predict(X_predict)

    return model, predictions

#Master function to combine candidate signals from raw candidates
def generate_candidate_library(train_df, predict_df, feature_columns, target_column):

    raw_candidates = generate_raw_candidates(predict_df, feature_columns)
    candidates = raw_candidates.copy()

    negative_candidates = generate_sign_reversed_candidates(raw_candidates)
    candidates.update(negative_candidates)

    rank_candidates = generate_rank_candidates(predict_df, raw_candidates)
    candidates.update(rank_candidates)

    signed_sqrt_candidates = generate_signed_sqrt_candidates(raw_candidates)
    candidates.update(signed_sqrt_candidates)

    winsorized_candidates = generate_winsorized_candidates(predict_df, raw_candidates, lower=0.05, upper=0.95)
    candidates.update(winsorized_candidates)

    pairwise_mult_candidates = generate_pairwise_interactions(raw_candidates)
    candidates.update(pairwise_mult_candidates)

    pairwise_diff_candidates = generate_pairwise_differences(raw_candidates)
    candidates.update(pairwise_diff_candidates)

    pairwise_ratio_candidates = generate_pairwise_ratios(raw_candidates)
    candidates.update(pairwise_ratio_candidates)

    predict_clean = predict_df.dropna(subset=feature_columns)
    model_ridge, predictions_ridge = generate_ridge_candidate(train_df, predict_df, feature_columns, target_column, alpha=1.0)
    candidates["ridge"] = pd.Series(predictions_ridge, index = predict_clean.index)

    model_lasso, predictions_lasso = generate_lasso_candidate(train_df, predict_df, feature_columns, target_column, alpha=1.0)
    candidates["lasso"] = pd.Series(predictions_lasso, index = predict_clean.index)

    return candidates


'''                 TRAIN DATA
                     │
                     ▼
              Ridge/Lasso learns
                     │
                     ▼
PREDICT DATA ────────┼──────→ Ridge/Lasso signal
     │
     ├───────────────────────→ momentum signal
     ├───────────────────────→ rank signal
     ├───────────────────────→ ratio signal
     └───────────────────────→ interaction signal

                     ↓
            candidate library
       (all on the same predict rows)'''


#for serializing; helper functions
def make_signal_definition(name, signal_type, source_features, operation, parameters=None):

    signal_dict = {
        "name": name,
        "signal_type": signal_type,
        "source_features": source_features,
        "operation": operation,
        "parameters": parameters
    }
    
    return signal_dict

def generate_raw_signal_definitions(feature_columns):

    feature_dict = {}
    
    for feature in feature_columns:
        definition = {
            "name": feature,
            "signal_type": "raw",
            "source_features": [feature],
            "operation": "identity",
            "parameters": None
        }
        feature_dict[feature] = definition
    
    return feature_dict

def generate_negative_signal_definitions(feature_columns):

    signal_dict = {}

    for feature in feature_columns:
        signal_name = f"negative_{feature}"
        definition = {
            "name": signal_name,
            "signal_type": "transformed",
            "source_features": [feature],
            "operation": "negate",
            "parameters": None
        }
        signal_dict[signal_name] = definition
    
    return signal_dict

def generate_rank_signal_definitions(feature_columns):

    signal_dict = {}

    for feature in feature_columns:
        signal_name = f"{feature}_rank"
        definition = {
            "name": signal_name,
            "signal_type": "transformed",
            "source_features": [feature],
            "operation": "rank",
            "parameters": {
                "pct": True,
                "ascending": True
            }
        }
        signal_dict[signal_name] = definition
    
    return signal_dict

def generate_signed_sqrt_signal_definitions(feature_columns):
    
    signal_dict = {}

    for feature in feature_columns:
        signal_name = f"{feature}_signed_sqrt"
        definition = {
            "name": signal_name,
            "signal_type": "transformed",
            "source_features": [feature],
            "operation": "signed_sqrt",
            "parameters": None
        }
        signal_dict[signal_name] = definition
    
    return signal_dict

def generate_winsorized_signal_definitions(feature_columns,lower=0.05,upper=0.95):

    signal_dict = {}

    for feature in feature_columns:
        signal_name = f"{feature}_winsorized"
        definition = {
            "name": signal_name,
            "signal_type": "transformed",
            "source_features": [feature],
            "operation": "winsorize",
            "parameters": {
                "lower": lower,
                "upper": upper
            }
        }
        signal_dict[signal_name] = definition
    
    return signal_dict

from itertools import combinations
def generate_pairwise_interaction_definitions(feature_columns):

    signal_dict = {}

    for feature1, feature2 in combinations(feature_columns, 2):
        signal_name = f"{feature1}_x_{feature2}"
        definition = {
            "name": signal_name,
            "signal_type": "pairwise",
            "source_features": [feature1,feature2],
            "operation": "multiply",
            "parameters": None
        }
        signal_dict[signal_name] = definition

    return signal_dict

def generate_pairwise_difference_definitions(feature_columns):

    signal_dict = {}

    for feature1, feature2 in combinations(feature_columns, 2):
        signal_name = f"{feature1}_minus_{feature2}"
        definition = {
            "name": signal_name,
            "signal_type": "pairwise",
            "source_features": [feature1, feature2],
            "operation": "subtract",
            "parameters": None
        }
        signal_dict[signal_name] = definition
    
    return signal_dict

def generate_pairwise_ratio_definitions(feature_columns):
    
    signal_dict = {}

    for feature1, feature2 in combinations(feature_columns, 2):
        signal_name = f"{feature1}_div_{feature2}"
        definition = {
            "name": signal_name,
            "signal_type": "pairwise",
            "source_features": [feature1, feature2],
            "operation": "divide",
            "parameters": {
                "zero_denominator": "nan"
            }
        }
        signal_dict[signal_name] = definition
    
    return signal_dict

def generate_ridge_signal_definition(feature_columns, alpha, model):

    definition = {
        "name": "ridge",
        "signal_type": "model",
        "source_features": feature_columns,
        "operation": "ridge",
        "parameters": {
            "alpha": alpha,
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_)
        }
    }

    return definition

def generate_lasso_signal_definition(feature_columns, alpha, model):

    definition = {
        "name": "lasso",
        "signal_type": "model",
        "source_features": feature_columns,
        "operation": "lasso",
        "parameters": {
            "alpha": alpha,
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_)
        }
    }

    return definition

#final function combining helper functions for serialization
def generate_signal_definition_library(feature_columns, ridge_model, lasso_model, ridge_alpha=1.0, lasso_alpha=1.0):

    raw_signal = generate_raw_signal_definitions(feature_columns)
    definitions = raw_signal.copy()

    definitions.update(generate_negative_signal_definitions(feature_columns))
    definitions.update(generate_rank_signal_definitions(feature_columns))
    definitions.update(generate_signed_sqrt_signal_definitions(feature_columns))
    definitions.update(generate_winsorized_signal_definitions(feature_columns,lower=0.05,upper=0.95))
    definitions.update(generate_pairwise_interaction_definitions(feature_columns))
    definitions.update(generate_pairwise_difference_definitions(feature_columns))
    definitions.update(generate_pairwise_ratio_definitions(feature_columns))

    definitions["ridge"] = generate_ridge_signal_definition(feature_columns, ridge_alpha, ridge_model)
    definitions["lasso"] = generate_lasso_signal_definition(feature_columns, lasso_alpha, lasso_model)

    return definitions

#serialization functions for definitions as json file and to read them back out
import json
def save_signal_definitions(definitions, path):
    
    with open(path, "w") as f:
        json.dump(definitions, f, indent=4)

def load_signal_definitions(path):

    with open(path, "r") as f:
        definitions = json.load(f)
    
    return definitions



        


