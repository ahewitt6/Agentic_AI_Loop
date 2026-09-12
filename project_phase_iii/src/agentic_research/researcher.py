

from agents import Agent, Runner, set_default_openai_key
from agentic_research.schemas import AlphaProposal
#from agentic_research.toy_engine import evaluate_alpha_tool

'''def create_research_agent():

    agent = Agent(name = "Research Agent", instructions = "Propose a toy alpha with weights and a thesis.", model = "gpt-5.6-luna", output_type = AlphaProposal, tools = [evaluate_alpha_tool])

    return agent'''

def create_research_agent():

    agent = Agent(
        name="Research Agent",
        instructions=(
            "Propose a real alpha signal using two allowed features, "
            "one valid operation, weights, and an economic thesis. "
            "The proposal must be implementable by the evaluation engine."
        ),
        model="gpt-5.6-luna",
        output_type=AlphaProposal
    )

    return agent

def run_research_agent(prompt):

    agent = create_research_agent()
    result = Runner.run_sync(agent,prompt)

    return result.final_output