import numpy as np
import pandas as pd

def compute_daily_ic(df, signal_column, target_column):

    daily_ic = df.groupby("date")[[signal_column, target_column]].apply(lambda x: x[signal_column].corr(x[target_column]))

    return daily_ic

def compute_daily_rank_ic(df, signal_column, target_column):

    res = df.copy()

    res[f"{signal_column}_rank"] = res.groupby("date")[signal_column].rank(pct = True, ascending = True)
    res[f"{target_column}_rank"] = res.groupby("date")[target_column].rank(pct = True, ascending = True)

    daily_rank_ic = res.groupby("date")[[f"{signal_column}_rank", f"{target_column}_rank"]].apply(lambda x: x[f"{signal_column}_rank"].corr(x[f"{target_column}_rank"]))

    return daily_rank_ic

def summarize_ic_metrics(df, signal_column, target_column):

    daily_ic = compute_daily_ic(df, signal_column, target_column)
    daily_rank_ic = compute_daily_rank_ic(df, signal_column, target_column)

    return {"mean_ic": daily_ic.mean(), "mean_rank_ic": daily_rank_ic.mean()}

def assign_signal_positions(df, signal_column):

    res = df.copy()

    rank_col = f"{signal_column}_rank"
    res[rank_col] = res.groupby("date")[signal_column].rank(pct = True, ascending = True)
    res["position"] = 0

    cond_max = res.groupby("date")[rank_col].transform(lambda x: x == x.max())
    cond_min = res.groupby("date")[rank_col].transform(lambda x: x == x.min())

    res.loc[cond_max, "position"] = 1
    res.loc[cond_min, "position"] = -1

    return res

def compute_daily_long_short_returns(df, signal_column, target_column):

    res = assign_signal_positions(df, signal_column)

    long_short_ret = res.groupby("date")[["position", target_column]].apply(lambda x: (x["position"]*x[target_column]).sum())

    return long_short_ret

def compute_long_short_sharpe(df, signal_column, target_column):

    long_short_ret = compute_daily_long_short_returns(df, signal_column, target_column)

    mean_ret = long_short_ret.mean()
    std_ret = long_short_ret.std()
    if std_ret == 0:
        return np.nan
    sharpe = mean_ret*np.sqrt(252)/std_ret

    return sharpe

def compute_signal_turnover(df, signal_column):

    res = assign_signal_positions(df, signal_column)
    res_pivot = res.pivot(index = "date", columns = "ticker", values = "position")
    turnovers = 0.5*(res_pivot.diff().abs().sum(axis = 1))
    turnovers = turnovers.iloc[1:]

    return turnovers

def summarize_signal_turnover(df, signal_column):

    turnovers = compute_signal_turnover(df, signal_column)

    return {"mean_turnover": turnovers.mean(), "max_turnover": turnovers.max()}

from scipy.stats import ttest_1samp
#understand this step a bit more
def compute_ic_p_value(df, signal_column, target_column):

    daily_ic = compute_daily_ic(df, signal_column, target_column)
    daily_ic = daily_ic.dropna()
    result = ttest_1samp(daily_ic, popmean=0)

    return result.pvalue

def evaluate_signal(df, signal_column, target_column):


    ic_metrics = summarize_ic_metrics(df, signal_column, target_column)
    sharpe = compute_long_short_sharpe(df, signal_column, target_column)
    turnover_metrics = summarize_signal_turnover(df, signal_column)
    p_value = compute_ic_p_value(df, signal_column, target_column)
    return {"mean_ic": ic_metrics["mean_ic"], "mean_rank_ic": ic_metrics["mean_rank_ic"],
            "sharpe": sharpe, "mean_turnover": turnover_metrics["mean_turnover"], "max_turnover": turnover_metrics["max_turnover"], 
            "p_value": p_value}

def evaluate_candidate_library(df, candidates, target_column):

    results = {}

    for name, signal in candidates.items():
        res = df.copy()
        res["signal"] = signal
        metrics = evaluate_signal(res, "signal", target_column)
        results[name] = metrics
    
    return results

#needs more practice
def results_to_dataframe(results):

    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.reset_index()
    df = df.rename(columns = {"index" : "signal"})

    return df

def apply_bonferroni_filter(results_df, alpha=0.05):

    res = results_df.copy()
    threshold = alpha/len(res)
    res["passes_bonferroni"] = res["p_value"] < threshold

    return res

def evaluate_all_candidates(df, candidates, target_column, alpha=0.05):

    results = evaluate_candidate_library(df, candidates, target_column)
    results_df = results_to_dataframe(results)
    results_df = apply_bonferroni_filter(results_df, alpha=alpha)

    return results_df

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

def build_evaluation_report(audit):

    audit_copy = audit.copy()
    audit_copy = audit_copy.sort_values("mean_rank_ic", ascending = False)
    kept_columns = [
        "signal",
        "mean_ic",
        "mean_rank_ic",
        "sharpe",
        "mean_turnover",
        "max_turnover",
        "p_value",
        "passes_bonferroni",
        "accepted",
        "rejection_reason"
    ]
    audit_copy = audit_copy[kept_columns]

    return audit_copy

def save_evaluation_reports(report_df, accepted_df, report_path, accepted_path):
    report_df.to_csv(report_path, index = False)
    accepted_df.to_csv(accepted_path, index = False)


    
    


    
    




