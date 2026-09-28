"""Reactive actor reasoner that outputs an action with brief reasoning."""

import re

class ReactiveActorReasoner:
    
    def __init__(self, client, domain='crafter'):
        """
        Initialize Reactive Actor Reasoner.
        
        Args:
            client: LLM client
            domain: 'crafter' or 'math' - determines instruction format (default: 'crafter')
        """
        self.client = client
        self.domain = domain.lower()
        
        if self.domain not in ['crafter', 'math']:
            raise ValueError(f"domain must be 'crafter' or 'math', got '{domain}'")
    
    def _get_message_role(self, msg):
        """
        Get role from message (handles both dict and object formats).
        
        Args:
            msg: Message (dict or object)
            
        Returns:
            str: Role of the message
        """
        if isinstance(msg, dict):
            return msg.get("role", "")
        else:
            return getattr(msg, "role", "")
    
    def _get_message_content(self, msg):
        """
        Get content from message (handles both dict and object formats).
        
        Args:
            msg: Message (dict or object)
            
        Returns:
            str: Content of the message
        """
        if isinstance(msg, dict):
            return msg.get("content", "")
        else:
            return getattr(msg, "content", "")
    
    def _set_message_content(self, msg, content):
        """
        Set content for message (handles both dict and object formats).
        
        Args:
            msg: Message (dict or object)
            content: New content
            
        Returns:
            Modified message (dict format)
        """
        if isinstance(msg, dict):
            msg["content"] = content
            return msg
        else:
            # Convert object to dict format
            return {
                "role": getattr(msg, "role", "user"),
                "content": content
            }
    
    def generate_action(self, messages, plan=None):
        """
        Generate action using reactive (fast) reasoning.
        
        Args:
            messages: Conversation history (list of dicts or objects)
            plan: Current plan (optional)
            
        Returns:
            tuple: (action, reasoning) for MATH domain, or
            str: action for Crafter domain (for backward compatibility)
        """
        instruction = self._get_reactive_instruction(plan)
        
        messages_copy = messages.copy()
        
        # Add instruction to last user message - handles both dict and object formats
        if messages_copy:
            last_msg = messages_copy[-1]
            last_role = self._get_message_role(last_msg)
            
            if last_role == "user":
                current_content = self._get_message_content(last_msg)
                new_content = current_content + "\n\n" + instruction
                messages_copy[-1] = self._set_message_content(last_msg, new_content)
        
        response = self.client.generate(messages_copy)
        
        if self.domain == 'math':
            # For MATH domain, extract both action and reasoning
            action, reasoning = self._extract_action_and_reasoning(response.completion)
            return action, reasoning
        else:
            # For Crafter domain, only return action (backward compatible)
            action = self._extract_action(response.completion)
            return action, None  # Return tuple for consistency
    
    def _get_reactive_instruction(self, plan):
        """
        Get instruction for reactive action generation based on domain.
        
        Args:
            plan: Current plan (optional)
            
        Returns:
            str: Reactive instruction
        """
        if self.domain == 'math':
            return self._get_reactive_instruction_math(plan)
        else:  # crafter (default)
            return self._get_reactive_instruction_crafter(plan)
    
    def _get_reactive_instruction_crafter(self, plan):
        """
        Get instruction for reactive action generation (Crafter domain).
        
        Args:
            plan: Current plan (optional)
            
        Returns:
            str: Reactive instruction for Crafter
        """
        base = "Based on your observation"
        if plan:
            base += " and plan"
        
        return f"""{base}, choose exactly ONE action from the allowed actions.

Respond quickly and instinctively to the current environment based on what you see.

Output format:
ACTION: [Noop|Move West|Move East|Move North|Move South|Do|Sleep|Place Stone|Place Table|Place Furnace|Place Plant|Make Wood Pickaxe|Make Stone Pickaxe|Make Iron Pickaxe|Make Wood Sword|Make Stone Sword|Make Iron Sword]

Output only the action line, nothing else."""
    
    def _get_reactive_instruction_math(self, plan):
        """
        Get instruction for reactive action generation (MATH domain).
        
        Args:
            plan: Current plan (optional)
            
        Returns:
            str: Reactive instruction for MATH
        """
        base = "Based on the problem"
        if plan:
            base += " and your plan"
        
        return f"""{base}, solve it directly and efficiently.

Use your mathematical intuition to quickly identify the solution approach and execute it.
Focus on the most direct path to the answer.

You are about to solve a math problem. Your task RIGHT NOW is to choose exactly next action from the list below.

Action set:
- ParseProblem: Extract key variables, conditions and constraints from the problem statement.
- ClassifySubjectDifficulty: Determine the subject area (Algebra, Geometry, etc.) and difficulty level.
- IdentifyKeywordsHeuristics: Scan for cues or heuristics that suggest specific techniques.
- ReformulateProblem: Restate the problem in a clearer or more formal mathematical notation.
- CheckAssumptions: Identify implicit domain constraints or assumptions.
- SelectStrategy: Choose a problem-solving strategy or heuristic to apply.
- DecomposeSubproblems: Break the main problem into smaller sub-tasks or cases.
- IdentifyToolsFormulas: List relevant formulas, theorems or tools required.
- EstimateFeasibilityCheck: Do a quick check of plausibility, bounds or magnitudes.
- PerformComputation: Execute an algebraic or numeric computation step.
- CaseAnalysis: Carry out one case in a case-by-case analysis.
- ConstructDiagramOrAuxiliary: For geometry or spatial problems, create an auxiliary construction or diagram.
- CombineResults: Combine results from sub-tasks or cases into an aggregate expression.
- SimplifyFinalizeExpression: Simplify the final expression into a standard form.
- SanityCheckFinalAnswer: Plug in special cases or check boundary values to verify reasonableness.
- BoxFinalAnswer: Format the final answer in the expected output style.
- ReviewSolution: Review the full solution chain for logic or hidden assumptions.
- GeneraliseOrEdgeCaseCheck: Consider extreme or boundary cases to ensure full correctness.
- AnnotateHeuristicUsed: Record which heuristic(s) were applied.
- FormatSolutionText: Prepare the full derivation or solution text for output or training.

Rules:
- Output ONLY the chosen action inside <action>...</action> tags.
- Inside the tags, write exactly: ActionName: one-line description.
- Do NOT solve the problem, do NOT include any other text before or after the tags.

Output format (exact):
<action>ActionName: one-line description</action>"""

    def _extract_action_and_reasoning(self, response_text):
        """
        Extract action (answer) and reasoning from response.
        Used for MATH domain.
        
        Args:
            response_text: LLM response
            
        Returns:
            tuple: (action/answer, reasoning)
        """
        reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
        reasoning_match = re.search(reasoning_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        reasoning = None
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract action
        action_pattern = r'<action>(.*?)</action>'
        action_match = re.search(action_pattern, response_text, re.IGNORECASE | re.DOTALL)

        if action_match:
            action = action_match.group(1).strip()
        else:
            # Fallback: use the full text after reasoning
            if reasoning_match:
                text_after_reasoning = response_text[reasoning_match.end():]
                action = text_after_reasoning.strip()
            else:
                action = response_text.strip()
            
            # Try to clean up
            action = self._extract_action(action)
        
        return action, reasoning
    
    def _extract_action(self, response_text):
        """
        Extract action from response.
        
        Args:
            response_text: LLM response
            
        Returns:
            str: Extracted action
        """
        # Remove XML tags except answer tags for MATH domain
        if self.domain == 'math':
            # Try to extract from answer tags first
            answer_pattern = r'<answer>(.*?)</answer>'
            answer_match = re.search(answer_pattern, response_text, re.IGNORECASE | re.DOTALL)
            if answer_match:
                return answer_match.group(1).strip()
        
        # Remove all XML tags
        response_text = re.sub(r'<[^>]+>', '', response_text)
        
        # Remove ACTION prefix for Crafter
        response_text = re.sub(
            r'^[\s]*(?:ACTION|Action)[\s]*[:=]?\s*',
            '',
            response_text,
            flags=re.IGNORECASE
        )
        
        action = response_text.split('\n')[0].strip()
        
        return action