from agents import Agent, Runner, set_default_openai_key
from agentic_research.schemas import AlphaCritique

'''def create_critic_agent():
    agent = Agent(name = "Critic Agent", instructions = "Critique toy alpha proposals using the evaluation results and suggest improved weights.", model = "gpt-5.6-luna", output_type = AlphaCritique)
    return agent

def build_critic_prompt(proposal, metrics):

    proposal_str = (
        f"Proposal:\n"
        + f"w1: {proposal.w1}\n"
        + f"w2: {proposal.w2}\n"
        + f"thesis: {proposal.thesis}\n\n"
        + f"Evaluation:\n"
        + f"correlation: {metrics['correlation']}\n\n"
        + f"Task:\n"
        + "Identify the main weakness in this proposal and suggest improved weights."
    )

    return proposal_str

def run_critic_agent(proposal, metrics):

    agent = create_critic_agent()
    prompt = build_critic_prompt(proposal, metrics)
    result = Runner.run_sync(agent, prompt)
    
    return result.final_output'''

def create_critic_agent():

    agent = Agent(
        name="Critic Agent",
        instructions=(
            "Critique alpha proposals using real evaluation metrics. "
            "Identify the main weakness and suggest a concrete revision "
            "to the features, operation, weights, or thesis."
        ),
        model="gpt-5.6-luna",
        output_type=AlphaCritique
    )


    return agent


def build_critic_prompt(proposal, metrics, allowed_features):

    proposal_str = (
        f"Allowed Features:\n"
        f"{allowed_features}\n\n"
        f"Allowed Operations:\n"
        f"add, subtract, multiply, divide\n\n"
        f"Proposal:\n"
        f"feature_1: {proposal.feature_1}\n"
        f"feature_2: {proposal.feature_2}\n"
        f"operation: {proposal.operation}\n"
        f"w1: {proposal.w1}\n"
        f"w2: {proposal.w2}\n"
        f"thesis: {proposal.thesis}\n\n"

        f"Evaluation:\n"
        f"mean_ic: {metrics['mean_ic']}\n"
        f"mean_rank_ic: {metrics['mean_rank_ic']}\n"
        f"sharpe: {metrics['sharpe']}\n"
        f"mean_turnover: {metrics['mean_turnover']}\n"
        f"max_turnover: {metrics['max_turnover']}\n"
        f"p_value: {metrics['p_value']}\n\n"

        f"Task:\n"
        "Identify the main weakness in this proposal. "
        "Suggest improved feature choices, operation, weights, "
        "and explain the reasoning. "
        "Only suggest features from the allowed feature list "
        "and only use the allowed operations."
    )

    return proposal_str


def run_critic_agent(proposal, metrics, allowed_features):

    agent = create_critic_agent()
    prompt = build_critic_prompt(proposal, metrics, allowed_features)
    result = Runner.run_sync(agent, prompt)

    return result.final_output