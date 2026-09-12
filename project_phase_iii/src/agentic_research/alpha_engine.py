import numpy as np
import pandas as pd

from quant_alpha_lab.evaluation import evaluate_signal

'''def evaluate_proposal(proposal, train_df, discovery_df):

    #Check feature_1 exists in discovery_df
    if proposal.feature_1 not in discovery_df.columns:
        raise ValueError(f"{proposal.feature_1} is not a feature in discovery_df.")

    #Check feature_2 exists in discovery_df
    if proposal.feature_2 not in discovery_df.columns:
        raise ValueError(f"{proposal.feature_2} is not a feature in discovery_df.")

    #Check operation is allowed
    allowed_operations = {"add", "subtract", "multiply", "divide"}
    if proposal.operation not in allowed_operations:
        raise ValueError(f"{proposal.operation} is not an allowed operation.")

    #reconstruct proposal
    x1 = proposal.w1 * discovery_df[proposal.feature_1]
    x2 = proposal.w2 * discovery_df[proposal.feature_2]

    if proposal.operation == "add":
        signal = x1 + x2

    elif proposal.operation == "subtract":
        signal = x1 - x2

    elif proposal.operation == "multiply":
        signal = x1 * x2

    elif proposal.operation == "divide":
        denominator = x2.copy()
        epsilon = 1e-8
        denominator = denominator.where(denominator.abs() > epsilon, np.nan)
        signal = x1 / denominator

    #note train_df is not used, it will be used once we allow proposals that require fitted parameters

    signal = signal.replace([np.inf, -np.inf], np.nan)

    #evaluate
    evaluation_df = discovery_df.copy()
    evaluation_df["signal"] = signal

    metrics = evaluate_signal(evaluation_df, "signal", "future_5d_return")
    metrics = { key: float(value) for key, value in metrics.items()}

    return metrics'''

def construct_signal(proposal, df):

    # Check feature_1 exists
    if proposal.feature_1 not in df.columns:
        raise ValueError(f"{proposal.feature_1} is not a feature in the dataframe.")

    # Check feature_2 exists
    if proposal.feature_2 not in df.columns:
        raise ValueError(f"{proposal.feature_2} is not a feature in the dataframe.")

    # Check operation is allowed
    allowed_operations = {"add", "subtract", "multiply", "divide"}

    if proposal.operation not in allowed_operations:
        raise ValueError(f"{proposal.operation} is not an allowed operation.")

    # Construct proposal
    x1 = proposal.w1 * df[proposal.feature_1]
    x2 = proposal.w2 * df[proposal.feature_2]

    if proposal.operation == "add":
        signal = x1 + x2

    elif proposal.operation == "subtract":
        signal = x1 - x2

    elif proposal.operation == "multiply":
        signal = x1 * x2

    elif proposal.operation == "divide":
        denominator = x2.copy()
        epsilon = 1e-8
        denominator = denominator.where(
            denominator.abs() > epsilon,
            np.nan
        )
        signal = x1 / denominator

    signal = signal.replace([np.inf, -np.inf], np.nan)

    return signal


def evaluate_proposal(proposal, train_df, discovery_df):

    signal = construct_signal(proposal, discovery_df)

    # Note: train_df is not currently used.
    # It will be used once proposals can require fitted parameters.

    evaluation_df = discovery_df.copy()
    evaluation_df["signal"] = signal

    metrics = evaluate_signal(
        evaluation_df,
        "signal",
        "future_5d_return"
    )

    metrics = {
        key: float(value)
        for key, value in metrics.items()
    }

    return metrics

def normalize_proposal(proposal):

    scale = np.sqrt(proposal.w1**2 + proposal.w2**2)
    if scale == 0:
        return 0.0, 0.0
    normalized_w1 = proposal.w1/scale
    normalized_w2 = proposal.w2/scale

    return normalized_w1, normalized_w2

def proposal_key(proposal):
    normalized_w1, normalized_w2 = normalize_proposal(proposal)
    normalized_w1 = round(normalized_w1,6)
    normalized_w2 = round(normalized_w2,6)
    return (proposal.feature_1, proposal.feature_2, proposal.operation, normalized_w1, normalized_w2)




