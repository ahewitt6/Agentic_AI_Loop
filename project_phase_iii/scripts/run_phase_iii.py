from quant_alpha_lab.pipeline import prepare_phase_iii_data
from agentic_research.loop import run_agentic_loop
from agentic_research.alpha_engine import evaluate_proposal
import json
import pandas as pd
import matplotlib.pyplot as plt


train_df, discovery_df, validation_df, test_df = prepare_phase_iii_data()

print("columns: ", train_df.columns)

initial_prompt = (
    "Propose an alpha signal using the available engineered market features. "
    "Use two features, one allowed operation, weights, and an economic thesis."
)

best_proposal, discovery_metrics, history = run_agentic_loop(
    initial_prompt,
    train_df,
    discovery_df,
    max_iterations=5
)

'''print("\n=== SLIDE 4: FULL ITERATION HISTORY ===")
for record in history:
    print(record)'''
print("\n=== SLIDE 4: SELECTED DISCOVERY WINNER ===")
print(best_proposal)
print(discovery_metrics)

validation_metrics = evaluate_proposal(best_proposal, train_df, validation_df)

validation_passed = (validation_metrics["mean_rank_ic"] >= 0.01 and validation_metrics["sharpe"] >= 0)

print("\n=== SLIDE 4: VALIDATION ===")
print(validation_metrics)

print("\n=== SLIDE 4: VALIDATION DECISION ===")
print(validation_passed)

#if validation_passed:
test_metrics = evaluate_proposal(best_proposal, train_df, test_df)
#else:
#    test_metrics = None

print("Best proposal:")
print(best_proposal)

print("\nDiscovery metrics:")
print(discovery_metrics)

print("\nValidation metrics:")
print(validation_metrics)

print("\nValidation passed:")
print(validation_passed)

print("\nTest metrics:")
print(test_metrics)

# Save iteration history
with open("phase_iii_history.json", "w") as f:
    json.dump(history, f, indent=4)

# Convert history to DataFrame
history_df = pd.DataFrame(history)

# Save history as CSV too
history_df.to_csv("phase_iii_history.csv", index=False)

# Plot discovery Rank IC over iterations
plt.figure()

plt.plot(history_df["iteration"], history_df["mean_rank_ic"], marker="o")

plt.xlabel("Iteration")
plt.ylabel("Mean Rank IC")
plt.title("Agentic Discovery Rank IC Over Iterations")

plt.tight_layout()
plt.savefig("phase_iii_rank_ic.png", dpi=300)

plt.show()