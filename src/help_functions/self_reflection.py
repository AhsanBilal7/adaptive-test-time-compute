"""Self-Reflection (SR) reasoner: initial attempt, critique, and refinement."""

import re


class SelfReflectionReasoner:
    """Self-Reflection reasoner: attempt, critique, refine."""
    def __init__(self, client):
        self.client = client
    
    def generate_action(self, messages, instruction_prompt):
        """
        Self-reflection reasoning: first attempt → critique → refined answer
        Returns: (final_action, full_reasoning_trace)
        """
        # Step 1: Generate initial attempt
        initial_messages = messages + [
            {"role": "user", "content": instruction_prompt}
        ]
        
        initial_response = self.client.generate(initial_messages)
        initial_completion = initial_response.completion if hasattr(initial_response, "completion") else str(initial_response)
        
        initial_reasoning, initial_action = self._extract_reasoning_and_action(initial_completion)
        
        # Step 2: Critique the initial attempt
        critique_prompt = f"""Review your previous reasoning and solution approach:

<initial_reasoning>
{initial_reasoning}
</initial_reasoning>

<initial_action>
{initial_action}
</initial_action>

Critically analyze:
1. Are there any logical gaps or errors in the reasoning?
2. Is the chosen action truly the most effective next step?
3. Are there overlooked edge cases or alternative approaches?
4. Is the reasoning mathematically sound?

Provide your critique in <critique>...</critique> tags."""
        
        critique_messages = initial_messages + [
            {"role": "assistant", "content": initial_completion},
            {"role": "user", "content": critique_prompt}
        ]
        
        critique_response = self.client.generate(critique_messages)
        critique_completion = critique_response.completion if hasattr(critique_response, "completion") else str(critique_response)
        critique = self._extract_critique(critique_completion)
        
        # Step 3: Generate refined solution based on critique
        refinement_prompt = f"""Based on your critique, provide an improved reasoning and action.

Previous attempt:
{initial_reasoning}
{initial_action}

Your critique:
{critique}

Now provide your refined solution using the same format as before:
<reasoning>Improved explanation incorporating the critique</reasoning>
<action>ActionName: refined description</action>"""
        
        refinement_messages = critique_messages + [
            {"role": "assistant", "content": critique_completion},
            {"role": "user", "content": refinement_prompt}
        ]
        
        refined_response = self.client.generate(refinement_messages)
        refined_completion = refined_response.completion if hasattr(refined_response, "completion") else str(refined_response)
        
        refined_reasoning, refined_action = self._extract_reasoning_and_action(refined_completion)
        
        # Compile full trace
        full_reasoning = f"""INITIAL ATTEMPT:
Reasoning: {initial_reasoning}
Action: {initial_action}

CRITIQUE:
{critique}

REFINED SOLUTION:
Reasoning: {refined_reasoning}
Action: {refined_action}"""
        
        return refined_action, refined_reasoning
    
    def _extract_reasoning_and_action(self, text):
        """Extract reasoning and action from XML tags"""
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE | re.DOTALL)
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        action = action_match.group(1).strip() if action_match else "No action provided"
        
        return reasoning, action
    
    def _extract_critique(self, text):
        """Extract critique from XML tags"""
        critique_match = re.search(r"<critique>(.*?)</critique>", text, re.IGNORECASE | re.DOTALL)
        return critique_match.group(1).strip() if critique_match else text.strip()