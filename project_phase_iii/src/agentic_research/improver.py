from agents import Agent, Runner, set_default_openai_key
from agentic_research.schemas import AlphaProposal

'''def create_improver_agent():
    agent = Agent(name = "Improver Agent", instructions = "Revise toy alpha proposals using critique and evaluation results.", model = "gpt-5.6-luna", output_type = AlphaProposal)
    return agent

def build_improver_prompt(proposal, metrics, critique):

    proposal_full = (
        f"Current Proposal:\n"
        + f"w1: {proposal.w1}\n" 
        + f"w2: {proposal.w2}\n"
        + f"thesis: {proposal.thesis}\n\n"
        + f"Evaluation:\n"
        + f"correlation: {metrics['correlation']}\n\n"
        + f"Critique:\n"
        + f"weakness: {critique.weakness}\n"
        + f"suggested_w1: {critique.suggested_w1}\n"
        + f"suggested_w2: {critique.suggested_w2}\n"
        + f"reasoning: {critique.reasoning} \n\n"
        + f"Task:\n"
        + "Propose improved weights and an updated thesis."
    )

    return proposal_full

def run_improver_agent(proposal, metrics, critique):

    agent = create_improver_agent()
    prompt = build_improver_prompt(proposal, metrics, critique)
    result = Runner.run_sync(agent,prompt)

    return result.final_output'''

def create_improver_agent():

    agent = Agent(
        name="Improver Agent",
        instructions=(
            "Revise alpha proposals using critique and real evaluation results. "
            "You may change the features, operation, weights, and thesis. "
            "Only use features and operations explicitly provided in the prompt."
        ),
        model="gpt-5.6-luna",
        output_type=AlphaProposal
    )

    return agent

def build_improver_prompt(proposal, metrics, critique, allowed_features):

    proposal_full = (
        f"Allowed Features:\n"
        f"{allowed_features}\n\n"
        f"Allowed Operations:\n"
        f"add, subtract, multiply, divide\n\n"

        f"Current Proposal:\n"
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

        f"Critique:\n"
        f"weakness: {critique.weakness}\n"
        f"suggested_feature_1: {critique.suggested_feature_1}\n"
        f"suggested_feature_2: {critique.suggested_feature_2}\n"
        f"suggested_operation: {critique.suggested_operation}\n"
        f"suggested_w1: {critique.suggested_w1}\n"
        f"suggested_w2: {critique.suggested_w2}\n"
        f"reasoning: {critique.reasoning}\n\n"

        f"Task:\n"
        "Return an improved alpha proposal using valid features, "
        "a valid operation, updated weights, and an updated thesis. "
        "Only suggest features from the allowed feature list "
        "and only use the allowed operations."
    )

    return proposal_full

def run_improver_agent(proposal, metrics, critique, allowed_features):

    agent = create_improver_agent()
    prompt = build_improver_prompt(proposal, metrics, critique, allowed_features)
    result = Runner.run_sync(agent,prompt)

    return result.final_output


def build_redundancy_feedback(proposal, reason, allowed_features):

    feedback = (
        f"Allowed Features:\n"
        f"{allowed_features}\n\n"
        f"Allowed Operations:\n"
        f"add, subtract, multiply, divide\n\n"

        f"The proposed alpha was rejected before evaluation.\n\n"

        f"Reason:\n"
        f"{reason}\n\n"

        f"Rejected Proposal:\n"
        f"feature_1: {proposal.feature_1}\n"
        f"feature_2: {proposal.feature_2}\n"
        f"operation: {proposal.operation}\n"
        f"w1: {proposal.w1}\n"
        f"w2: {proposal.w2}\n"
        f"thesis: {proposal.thesis}\n\n"

        f"Task:\n"
        f"Generate a genuinely different alpha proposal. "
        f"Changing only the overall scale of the weights is not a meaningful revision. "
        f"Make a structural change to the relative weights, feature choices, "
        f"or operation. Only use the allowed features and allowed operations."
    )

    return feedback


def run_redundancy_retry(proposal, reason, allowed_features):

    agent = create_improver_agent()

    prompt = build_redundancy_feedback(
        proposal,
        reason,
        allowed_features
    )

    result = Runner.run_sync(agent, prompt)

    return result.final_output