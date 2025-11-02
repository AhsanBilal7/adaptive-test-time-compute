"""
Simple test example for CustomAgent with tool and compute selection.
This demonstrates how to use the updated agent with a single question.
"""

from custom import CustomAgent


def test_custom_agent_with_selectors():
    """
    Test the CustomAgent with tool and compute selectors enabled.
    This example shows a complete workflow with a single observation.
    """
    
    # Step 1: Create the agent with selectors enabled
    # use_tool_selector=True: Master LLM will choose which reasoner to use
    # use_compute_selector=True: Master LLM will choose which compute strategy to use
    agent = CustomAgent(
        client_factory,           # Your LLM client factory
        prompt_builder,           # Your prompt builder
        mode='dynamic',           # Agent decides when to plan
        use_planner=True,         # Enable planning
        use_tool_selector=True,   # Enable tool selection (reactive_actor, cot, heuristic_script)
        use_compute_selector=True,# Enable compute selection (best_of_n, beam_search, lookahead)
        dataset='crafter'         # Use Crafter policy for heuristic reasoner
    )
    
    # Step 2: Define a test observation (example from Crafter environment)
    observation = """
    You are in a forest. 
    Health: 8/10
    Hunger: 6/10
    Thirst: 7/10
    Inventory: wood: 0, stone: 0
    Nearby objects: tree, stone, plant
    """
    
    # Step 3: Call act() - NO CHANGES to how you call this!
    # The agent will internally:
    # 1. Decide whether to plan (dt ∈ {0,1})
    # 2. If planning (dt=1):
    #    a. Tool Selector chooses reasoner (reactive_actor, cot, or heuristic_script)
    #    b. Compute Selector chooses strategy (best_of_n, beam_search, or lookahead)
    #    c. Execute with selected tool and strategy
    #    d. PRM scores responses if needed
    # 3. Return best action
    response = agent.act(observation, prev_action=None)
    
    # Step 4: Extract results from response
    action = response.completion   # The action to execute
    plan = response.reasoning      # The plan (if agent decided to plan)
    
    # Step 5: Print results
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"\nObservation: {observation.strip()}")
    print(f"\nPlanning Decision (dt): {agent.planning_decisions[-1]}")
    
    if plan:
        print(f"\nPlan Generated:\n{plan}")
    else:
        print("\nNo plan generated (agent decided not to plan)")
    
    print(f"\nAction Selected: {action}")
    
    # Step 6: Get detailed statistics
    # This shows what happened internally
    stats = agent.get_planning_stats()
    
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"\nTotal timesteps: {stats['total_timesteps']}")
    print(f"Planning frequency: {stats['planning_frequency']:.2f}")
    print(f"Planner enabled: {stats['use_planner']}")
    print(f"Tool selector enabled: {stats['use_tool_selector']}")
    print(f"Compute selector enabled: {stats['use_compute_selector']}")
    
    # If selectors were used, show their decisions
    if 'tool_selection' in stats:
        print(f"\nTool Selection Stats:")
        print(f"  {stats['tool_selection']}")
    
    if 'compute_selection' in stats:
        print(f"\nCompute Selection Stats:")
        print(f"  {stats['compute_selection']}")
    
    if 'prm_scoring' in stats:
        print(f"\nPRM Scoring Stats:")
        print(f"  {stats['prm_scoring']}")
    
    # Step 7: Get compute metadata (shows which tools and strategies were used)
    if agent.compute_metadata_history:
        metadata = agent.compute_metadata_history[-1]
        print("\n" + "=" * 80)
        print("EXECUTION DETAILS")
        print("=" * 80)
        print(f"\nTool Used: {metadata.get('tool', 'N/A')}")
        print(f"Compute Strategy: {metadata.get('compute_strategy', 'N/A')}")
        if 'compute_config' in metadata:
            print(f"Compute Config: {metadata['compute_config']}")
        if 'best_score' in metadata:
            print(f"Best Score (PRM): {metadata['best_score']:.3f}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return action, plan, stats


def test_custom_agent_original_behavior():
    """
    Test the CustomAgent with original behavior (no selectors).
    This shows backward compatibility - same as original CustomAgent.
    """
    
    # Create agent with selectors DISABLED (original behavior)
    agent = CustomAgent(
        client_factory,
        prompt_builder,
        mode='dynamic',
        use_planner=True,
        use_tool_selector=False,   # Disabled
        use_compute_selector=False # Disabled
    )
    
    # Same observation
    observation = "You are in a forest. Health: 8/10. You see a tree."
    
    # Same act() call - works exactly as before!
    response = agent.act(observation)
    
    action = response.completion
    plan = response.reasoning
    
    print("=" * 80)
    print("ORIGINAL BEHAVIOR TEST")
    print("=" * 80)
    print(f"\nObservation: {observation}")
    print(f"\nAction: {action}")
    if plan:
        print(f"\nPlan: {plan}")
    
    # Stats will NOT include tool_selection or compute_selection
    stats = agent.get_planning_stats()
    print(f"\nStatistics: {stats}")
    print("\n" + "=" * 80)
    
    return action, plan


def test_custom_agent_tool_only():
    """
    Test with only tool selector enabled (no compute selector).
    Shows you can enable features independently.
    """
    
    # Enable only tool selector
    agent = CustomAgent(
        client_factory,
        prompt_builder,
        use_tool_selector=True,    # Enabled
        use_compute_selector=False # Disabled
    )
    
    observation = "You are hungry. Health: 5/10, Hunger: 2/10. You see a plant."
    
    response = agent.act(observation)
    
    print("=" * 80)
    print("TOOL SELECTOR ONLY TEST")
    print("=" * 80)
    print(f"\nObservation: {observation}")
    print(f"\nAction: {response.completion}")
    
    stats = agent.get_planning_stats()
    if 'tool_selection' in stats:
        print(f"\nTool Selection: {stats['tool_selection']}")
    
    print("\n" + "=" * 80)
    
    return response.completion


def test_fixed_frequency_with_selectors():
    """
    Test fixed-frequency planning with selectors.
    Plans every 4 steps, uses selectors when planning occurs.
    """
    
    # Create agent with fixed-frequency planning
    agent = CustomAgent(
        client_factory,
        prompt_builder,
        mode='fixed',              # Fixed-frequency mode
        planning_frequency=4,      # Plan every 4 steps
        use_planner=True,
        use_tool_selector=True,
        use_compute_selector=True
    )
    
    # Simulate multiple steps
    observations = [
        "Step 1: You spawn in forest. Health: 9/10",
        "Step 2: You see trees. Health: 9/10",
        "Step 3: You collected wood. Wood: 1",
        "Step 4: You see stone. Wood: 2",
        "Step 5: You collected stone. Wood: 2, Stone: 1"
    ]
    
    print("=" * 80)
    print("FIXED-FREQUENCY TEST (K=4)")
    print("=" * 80)
    
    prev_action = None
    for i, obs in enumerate(observations):
        response = agent.act(obs, prev_action)
        action = response.completion
        
        print(f"\n{obs}")
        print(f"  Planning (dt): {agent.planning_decisions[-1]}")
        print(f"  Action: {action}")
        if response.reasoning:
            print(f"  Plan: {response.reasoning[:50]}...")
        
        prev_action = action
    
    # Final statistics
    stats = agent.get_planning_stats()
    print("\n" + "=" * 80)
    print(f"Planning Frequency: {stats['planning_frequency']:.2f}")
    print(f"Expected: {stats.get('expected_planning_frequency', 'N/A')}")
    print("=" * 80)


# Example usage with mock objects (replace with your actual implementations)
if __name__ == "__main__":
    # You need to provide these:
    # - client_factory: Function that returns an LLM client
    # - prompt_builder: Your prompt builder object
    
    # Mock example (replace with your actual objects):
    # from your_module import create_client, PromptBuilder
    # client_factory = create_client
    # prompt_builder = PromptBuilder()
    
    # Uncomment the test you want to run:
    
    # Test 1: Full system with selectors
    # test_custom_agent_with_selectors()
    
    # Test 2: Original behavior (backward compatibility)
    # test_custom_agent_original_behavior()
    
    # Test 3: Tool selector only
    # test_custom_agent_tool_only()
    
    # Test 4: Fixed-frequency planning
    # test_fixed_frequency_with_selectors()
    
    print("\nTo run tests, provide client_factory and prompt_builder,")
    print("then uncomment the desired test function.")