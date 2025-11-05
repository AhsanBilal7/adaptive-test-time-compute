# test_multitool_chain.py
from types import SimpleNamespace
from balrog.agents.custom import CustomAgent
from balrog.agents.reasoners import (
    NumericVerifier, SummarizerTool,
    CoTReasoner, ReactiveActorReasoner
)
from balrog.agents.prm_model import PRMModel
from balrog.agents.tool_selector import ToolSelector

# ------------------------------
# Mock LLM client for all modules
# ------------------------------
class MockMessage:
    """Message object with role and content attributes."""
    def __init__(self, role, content):
        self.role = role
        self.content = content

class MockClient:
    def __init__(self, response_text):
        self.response_text = response_text
    def generate(self, messages):
        # More realistic CoT output with action
        realistic_output = """<reasoning>
Step 1: Observe that there is a locked door blocking the path
Step 2: Need to find a key to unlock the door
Step 3: Best action is to search nearby for a key
</reasoning>
ACTION: search_for_key"""
        return SimpleNamespace(completion=realistic_output if "step-by-step" in str(messages) else self.response_text)

# ------------------------------
# Mock ToolSelector that always returns multiple tools
# ------------------------------
class MockMultiToolSelector(ToolSelector):
    def __init__(self):
        self.tool_selection_history = []
    def select_tool(self, obs, plan, history_messages):
        tools = ["cot", "numeric_verifier", "summarizer"]
        self.tool_selection_history.append({"tools": tools})
        return {"tools": tools}
    def get_tool_distribution(self):
        return {"mocked": True}

# ------------------------------
# Custom Agent subclass for testing
# ------------------------------
class MockAgent(CustomAgent):
    def __init__(self):
        # mock client (used by PRM, reasoners, tools)
        self.client = MockClient("<score>0.92</score>")
        
        # Initialize flags and config attributes
        self.fixed_tool = None
        self.fixed_compute = None
        self.use_tool_selector = True
        self.use_compute_selector = False
        self.plan = "Test plan: verify multi-tool chaining"
        
        # initialize PRM and tools manually
        self.prm_model = PRMModel(self.client)
        self.numeric_verifier = NumericVerifier(self.prm_model)
        self.summarizer = SummarizerTool(self.client)
        
        self.reasoners = {
            "cot": CoTReasoner(self.client),
            "reactive_actor": ReactiveActorReasoner(self.client),
        }
        
        self.tool_selector = MockMultiToolSelector()
        self.compute_selector = None
        self.compute_metadata_history = []

# ------------------------------
# Run the test
# ------------------------------
def run_chain_test():
    agent = MockAgent()
    obs = "Player sees a locked door."
    default_action = "open door"
    messages = [
        MockMessage("user", "What should I do?"),
        MockMessage("assistant", "Let's analyze the situation.")
    ]

    print("\n=== Running multi-tool chain test ===")
    final_action = agent._execute_with_selectors(obs, messages, default_action)

    print("\nFinal action:", final_action)
    print("\nCompute metadata history:")
    for m in agent.compute_metadata_history:
        print("  ", m)

    # --------------- Assertions ---------------
    assert any(m.get("tool") == "cot" for m in agent.compute_metadata_history), "CoTReasoner missing"
    assert any(m.get("tool") == "numeric_verifier" for m in agent.compute_metadata_history), "NumericVerifier missing"
    assert any(m.get("tool") == "summarizer" for m in agent.compute_metadata_history), "Summarizer missing"
    assert isinstance(final_action, str), f"Final action not string, got {type(final_action)}: {final_action}"
    print("\n✅ Multi-tool chaining verified successfully!")

if __name__ == "__main__":
    run_chain_test()