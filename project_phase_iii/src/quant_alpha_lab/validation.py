from quant_alpha_lab.evaluation import evaluate_all_candidates, evaluate_candidate_library
import pandas as pd
import json


#needs practice
def filter_accepted_signals(results_df, min_rank_ic=0.02, min_sharpe=0.5, max_turnover=1.0):

    res = results_df.copy()

    res["passes_rank_ic"] = res["mean_rank_ic"] >= min_rank_ic
    res["passes_sharpe"] = res["sharpe"] >= min_sharpe
    res["passes_turnover"] = res["mean_turnover"] <= max_turnover

    res["accepted"] = (res["passes_rank_ic"]) & (res["passes_sharpe"]) & (res["passes_turnover"]) & (res["passes_bonferroni"])


    def get_rejection_reason(row):

        reasons = []

        if not row["passes_rank_ic"]:
            reasons.append("rank_ic")
        if not row["passes_sharpe"]:
            reasons.append("sharpe")
        if not row["passes_turnover"]:
            reasons.append("turnover")
        if not row["passes_bonferroni"]:
            reasons.append("bonferroni")
        if not reasons:
            return "accepted"

        return ", ".join(reasons)

    res["rejection_reason"] = res.apply(get_rejection_reason,axis = 1)

    accepted_signals = res[res["accepted"]].copy()

    return accepted_signals, res

def evaluate_and_filter_candidates(df, candidates, target_column, alpha=0.05, min_rank_ic=0.02, min_sharpe=0.5, max_turnover=1.0):

    results_df = evaluate_all_candidates(df, candidates, target_column, alpha=alpha)
    accepted_signals, audit = filter_accepted_signals(results_df, min_rank_ic=min_rank_ic, min_sharpe=min_sharpe, max_turnover=max_turnover)

    return accepted_signals, audit

#old need 4 periods because evaluation uses validation, then for validation we need an extra set before testing
def temporal_split_old(df, train_end, validation_end):

    train_df = df.loc[df["date"]<= train_end].copy()
    valid_df = df.loc[(df["date"] > train_end) & (df["date"] <= validation_end)].copy()
    test_df = df.loc[df["date"] > validation_end].copy()

    return train_df, valid_df, test_df

def temporal_split(df, train_end, discovery_end, validation_end):

    train_df = df.loc[df["date"] <= train_end].copy()

    discovery_df = df.loc[(df["date"] > train_end) & (df["date"] <= discovery_end)].copy()

    validation_df = df.loc[(df["date"] > discovery_end) & (df["date"] <= validation_end)].copy()

    test_df = df.loc[df["date"] > validation_end].copy()

    return (train_df, discovery_df, validation_df, test_df)

def validate_accepted_signals(validation_df, accepted_candidates, target_column):

    results = evaluate_candidate_library(validation_df, accepted_candidates, target_column)

    return results

def passes_validation_robustness(sharpe_values, rank_ic_values, min_sharpe=0.5, min_rank_ic=0.02):
    return (min(sharpe_values) >= min_sharpe) and (min(rank_ic_values) >= min_rank_ic)

def build_validation_report(validation_results):

    valid_res_df = pd.DataFrame.from_dict(validation_results, orient="index")
    valid_res_df = valid_res_df.reset_index()
    valid_res_df = valid_res_df.rename(columns = {"index":"signal"})
    valid_res_df = valid_res_df.sort_values("mean_rank_ic", ascending = False)

    return valid_res_df

def save_validated_signals(validated_df, path):
    validated_df.to_csv(path, index = False)

def save_validated_signal_definitions(validated_df, signal_definitions, path):

    signal_names = set(validated_df["signal"])

    validated_signal_dict = {}
    for name, dic in signal_definitions.items():
        if name in signal_names:
            validated_signal_dict[name] = dic

    with open(path, "w") as f:
        json.dump(validated_signal_dict, f, indent = 4)




