from agentic_research.alpha_engine import evaluate_proposal
from agentic_research.schemas import AlphaProposal
from quant_alpha_lab.pipeline import prepare_phase_iii_data

'''train_df, discovery_df, validation_df, test_df = prepare_phase_iii_data()

proposal = AlphaProposal(
    feature_1="volatility_20d",
    feature_2="relative_volume",
    operation="multiply",
    w1=1.0,
    w2=1.0,
    thesis="Higher volatility combined with unusual trading volume may identify stocks with stronger near-term cross-sectional return effects."
)

metrics = evaluate_proposal(
    proposal,
    train_df,
    discovery_df
)

print(metrics)
print(proposal)'''

from quant_alpha_lab.pipeline import prepare_phase_iii_data

from agentic_research.loop import run_agentic_loop

train_df, discovery_df, validation_df, test_df = prepare_phase_iii_data()

print("columns: ", train_df.columns)
allowed_features = [
    col for col in discovery_df.columns
    if col not in {
        "date",
        "ticker",
        "close",
        "volume",
        "future_5d_return"
    }
]

initial_prompt = (
    "Propose an alpha using exactly two features from this allowed list:\n"
    f"{allowed_features}\n\n"
    "Allowed operations are: add, subtract, multiply, divide.\n"
    "Return two feature names, one operation, weights, and an economic thesis. "
    "Do not invent feature names outside this list."
)

final_proposal, final_metrics, history = run_agentic_loop(

    initial_prompt,

    train_df,

    discovery_df,

    max_iterations=5

)



#print(history)

validation_metrics = evaluate_proposal(

    final_proposal,

    train_df,

    validation_df

)

'''print(final_proposal)

print(final_metrics)

print(validation_metrics)'''

validation_passed = (

    validation_metrics["mean_rank_ic"] >= 0.01

    and validation_metrics["sharpe"] >= 0

)

print(validation_passed)