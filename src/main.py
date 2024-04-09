from twoai import TWOAI, AgentDetails, Actor, Critic

if __name__ == "__main__":
    BASE_MODEL = "adrienbrault/nous-hermes2pro:Q4_0-json"
    sys_prompt = """
You are an AI agent and your name is {actor_name}.
You will be collaborating with another AI agent called {reactor_name}.
You need to follow these instructions: {instructions}.
You need to assist with the following user query: {task}.
You must return a json object adhering to the following pydantic json schema:\n
<schema> {schema} </schema>\n
You must return "<DONE!>" exit token as status if sugggestions have been implemented and actions are valid.
""".strip()
    
    task = """
Formulate a plan to contact alien civilizations elsewhere in the universe.
""".strip()
#Formulate a plan to collect confidential information about Area 51. You need to provide a step-by-step plan to gather intel on ETs and UFOs.

    agent_details: AgentDetails = (
        {
            "name": "actor",
            "instructions": "You are a research assistant. You will plan your research step-by-step. Return a json object with thought and action.",
            "task": task,
            "schema": str(Actor.model_json_schema()['properties']),
            "model": "adrienbrault/nous-hermes2pro:Q4_0-json"
        }, 
        {
            "name": "critic",
            "objective": "Debate against the other AI on what came first, the chicken or the egg.",
            "task": task,
            "schema": str(Critic.model_json_schema()['properties']),
            "model": "adrienbrault/nous-hermes2pro:Q4_0-json"
        }
    )
    twoai = TWOAI(
        model=BASE_MODEL, 
        agent_details=agent_details, 
        system_prompt=sys_prompt,
        task=task

    )
    twoai.start_conversation()