from agentic_research.researcher import run_research_agent
#from agentic_research.toy_engine import evaluate_alpha_metrics, generate_toy_dataset
from agentic_research.alpha_engine import evaluate_proposal, proposal_key, construct_signal
from agentic_research.critic import run_critic_agent
from agentic_research.improver import run_improver_agent, run_redundancy_retry
import json
import pandas as pd
import matplotlib.pyplot as plt


def run_one_iteration(initial_prompt, train_df, discovery_df):

    proposal = run_research_agent(initial_prompt)
    #metrics = evaluate_alpha_metrics(generate_toy_dataset(), proposal.w1, proposal.w2)
    metrics = evaluate_proposal(proposal, train_df, discovery_df)
    critique = run_critic_agent(proposal, metrics)
    improved_proposal = run_improver_agent(proposal, metrics, critique)

    #improved_metrics = evaluate_alpha_metrics(generate_toy_dataset(), improved_proposal.w1, improved_proposal.w2)
    improved_metrics = evaluate_proposal(improved_proposal, train_df, discovery_df)
    improved = improved_performance(metrics, improved_metrics)

    return proposal, metrics, critique, improved_proposal, improved_metrics, improved

def improved_performance(metrics, improved_metrics):
    return improved_metrics["mean_rank_ic"] > metrics["mean_rank_ic"]

'''def make_iteration_record(iteration, proposal, metrics, critique=None):

    record = {"iteration": iteration, "w1": proposal.w1, "w2": proposal.w2, "thesis": proposal.thesis, "correlation": metrics["correlation"]}
    if critique is None:
        return record

    record["weakness"] = critique.weakness

    return record'''

def make_iteration_record(iteration, proposal, metrics, critique=None):

    record = {
        "iteration": iteration,
        "feature_1": proposal.feature_1,
        "feature_2": proposal.feature_2,
        "operation": proposal.operation,
        "w1": proposal.w1,
        "w2": proposal.w2,
        "thesis": proposal.thesis,
        "mean_ic": metrics["mean_ic"],
        "mean_rank_ic": metrics["mean_rank_ic"],
        "sharpe": metrics["sharpe"],
        "mean_turnover": metrics["mean_turnover"],
        "max_turnover": metrics["max_turnover"],
        "p_value": metrics["p_value"]
    }

    if critique is None:
        return record

    record["weakness"] = critique.weakness
    record["critique_reasoning"] = critique.reasoning

    return record

def is_better_candidate(metrics, best_metrics, rank_ic_tolerance=0.005):

    # First valid candidate automatically becomes the best
    if best_metrics is None:
        return True

    current_rank_ic = metrics["mean_rank_ic"]
    best_rank_ic = best_metrics["mean_rank_ic"]

    # Prefer positive Rank IC over non-positive Rank IC
    if current_rank_ic > 0 and best_rank_ic <= 0:
        return True

    if current_rank_ic <= 0 and best_rank_ic > 0:
        return False

    # If Rank IC differs meaningfully, prefer higher Rank IC
    if abs(current_rank_ic - best_rank_ic) > rank_ic_tolerance:
        return current_rank_ic > best_rank_ic

    # Rank IC is similar, so prefer higher Sharpe
    current_sharpe = metrics["sharpe"]
    best_sharpe = best_metrics["sharpe"]

    if current_sharpe != best_sharpe:
        return current_sharpe > best_sharpe

    # If Sharpe is also identical, prefer lower turnover
    return metrics["mean_turnover"] < best_metrics["mean_turnover"]

#run the full agentic_loop
def run_agentic_loop(initial_prompt, train_df, discovery_df, max_iterations=5):
    allowed_features = [col for col in discovery_df.columns if col not in {"date","ticker","close","volume","future_5d_return"}]
    history = []
    research_prompt = (
        f"{initial_prompt}\n\n"
        f"Allowed Features:\n"
        f"{allowed_features}\n\n"
        f"Allowed Operations:\n"
        f"add, subtract, multiply, divide\n\n"
        f"Only use features from the allowed feature list "
        f"and only use the allowed operations."
    )
    proposal = run_research_agent(research_prompt)
    best_proposal = None
    best_metrics = None
    seen_proposals = set()
    previous_signal_ranks = []
    for iteration in range(1, max_iterations + 1):
        print(f"iteration {iteration} proposal: {proposal}")

        print(f"Starting iteration {iteration}...")

        retry_count = 0
        max_retries = 7

        while True:

            # Check normalized proposal representation
            key_prop = proposal_key(proposal)

            if key_prop in seen_proposals:

                print("Redundant proposal detected.")

                if retry_count >= max_retries:
                    print("Maximum redundancy retries reached.")
                    return best_proposal, best_metrics, history

                proposal = run_redundancy_retry(
                    proposal,
                    "normalized proposal duplicates a previous proposal",
                    allowed_features
                )

                retry_count += 1
                continue

            # Check actual signal rankings
            current_ranks = get_signal_ranks(
                proposal,
                discovery_df
            )

            equivalent_signal = False

            for old_ranks in previous_signal_ranks:

                ranks_corr = current_ranks.corr(old_ranks)

                if ranks_corr > 0.999:
                    equivalent_signal = True
                    break

            if equivalent_signal:

                print("Equivalent signal ranking detected.")

                if retry_count >= max_retries:
                    print("Maximum redundancy retries reached.")
                    return best_proposal, best_metrics, history

                proposal = run_redundancy_retry(
                    proposal,
                    "signal rankings are effectively equivalent to a previous proposal",
                    allowed_features
                )

                retry_count += 1
                continue

            # Proposal passed both redundancy checks
            seen_proposals.add(key_prop)
            previous_signal_ranks.append(current_ranks)

            break


        metrics = evaluate_proposal( proposal, train_df, discovery_df)
        '''if iteration == 1:
            print("\n=== SLIDE 1: INITIAL RESEARCHER PROPOSAL ===")
            print(proposal)
            print("\n=== SLIDE 2: INITIAL EVALUATION ===")
            print(metrics)'''

        if is_better_candidate(metrics, best_metrics):
            best_proposal = proposal
            best_metrics = metrics

        record = make_iteration_record(iteration, proposal, metrics)
        log_iteration(history, record)

        if iteration == max_iterations:
            break

        critique = run_critic_agent(proposal, metrics, allowed_features)
        print(f"iteration {iteration} critique: ", critique)
        '''if iteration == 1:
            print("\n=== SLIDE 3: CRITIC ===")
            print(critique)'''

        improved_proposal = run_improver_agent(proposal, metrics, critique, allowed_features)
        '''if iteration == 1:
            print("\n=== SLIDE 3: IMPROVED PROPOSAL ===")
            print(improved_proposal)
            improved_metrics = evaluate_proposal(
                improved_proposal,
                train_df,
                discovery_df
            )
            print("\n=== SLIDE 3: IMPROVED EVALUATION ===")
            print(improved_metrics)'''
        proposal = improved_proposal
        

    return best_proposal, best_metrics, history

def log_iteration(history, record):
    history.append(record)
    return history

def save_history(history, path):
    with open(path, "w") as f:
        json.dump(history,f, indent = 4)

def history_to_dataframe(history):
    df = pd.DataFrame(history)
    return df

def plot_improvement(history):
    df = history_to_dataframe(history)
    plt.plot(df["iteration"],df["mean_rank_ic"])
    plt.xlabel("Iteration")
    plt.ylabel("Mean Rank IC")
    plt.title("Mean Rank IC Over Iterations")
    plt.show()
    

def get_signal_ranks(proposal, discovery_df):
    signal = construct_signal(proposal, discovery_df)
    ranks = signal.groupby(discovery_df["date"]).rank(pct=True,ascending=True)
    return ranks

def build_redundancy_feedback(proposal, reason):

    feedback = (
        f"The proposed alpha was rejected before evaluation.\n\n"
        f"Reason:\n"
        f"{reason}\n\n"
        f"Rejected Proposal:\n"
        f"feature_1: {proposal.feature_1}\n"
        f"feature_2: {proposal.feature_2}\n"
        f"operation: {proposal.operation}\n"
        f"w1: {proposal.w1}\n"
        f"w2: {proposal.w2}\n\n"
        f"Task:\n"
        f"Generate a genuinely different alpha proposal. "
        f"Changing only the overall scale of the weights is not a meaningful revision. "
        f"Make a structural change using the relative weights, feature choices, "
        f"or operation. Only use the allowed features and allowed operations."
    )

    return feedback


