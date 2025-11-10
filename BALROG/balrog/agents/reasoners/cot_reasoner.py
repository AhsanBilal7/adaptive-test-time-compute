from email.mime import base
import re


class CoTReasoner:
    
    def __init__(self, client, domain='crafter'):
        """
        Initialize CoT Reasoner.
        
        Args:
            client: LLM client
            domain: 'crafter' or 'math' - determines instruction format (default: 'crafter')
        """
        self.client = client
        self.reasoning_history = []
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
        Generate action using chain-of-thought reasoning.
        
        Args:
            messages: Conversation history (list of dicts or objects)
            plan: Current plan (optional)
            
        Returns:
            tuple: (action, reasoning)
        """
        instruction = self._get_cot_instruction(plan)
        
        messages_copy = messages.copy()
        # print("messages_copy before adding instruction:", messages_copy)  # Debugging line
        
        # Add instruction to last user message
        if messages_copy:
            last_msg = messages_copy[-1]
            last_role = self._get_message_role(last_msg)
            
            if last_role == "user":
                current_content = self._get_message_content(last_msg)
                new_content = current_content + "\n\n" + instruction
                messages_copy[-1] = self._set_message_content(last_msg, new_content)
        
        response = self.client.generate(messages_copy)
        action, reasoning = self._extract_action_and_reasoning(response.completion)
        
        # print("messages_copy after adding instruction:", messages_copy)  # Debugging line
        # print("CoT Reasoner response:", response.completion)  # Debugging line
        # print("CoT Reasoner action:", action)  # Debugging line
        # print("CoT Reasoner reasoning:", reasoning)  # Debugging line

        self.reasoning_history.append(reasoning)
        
        return action, reasoning
    
    def _get_cot_instruction(self, plan):
        """
        Get instruction for chain-of-thought reasoning based on domain.
        
        Args:
            plan: Current plan (optional)
            
        Returns:
            str: CoT instruction
        """
        if self.domain == 'math':
            return self._get_cot_instruction_math(plan)
        else:  # crafter (default)
            return self._get_cot_instruction_crafter(plan)
    
    def _get_cot_instruction_crafter(self, plan):
        """
        Get instruction for chain-of-thought reasoning (Crafter domain).
        
        Args:
            plan: Current plan (optional)
            
        Returns:
            str: CoT instruction for Crafter
        """
        base = "Look at your observation"
        if plan:
            base += " and plan"
        
        return f"""{base}, then decide the best next move based on the current environment, inventory, and visible entities.

After your reasoning, output exactly ONE action from the allowed actions below.
Choose exactly one action from:
Noop, Move West, Move East, Move North, Move South,
Do, Sleep,
Place Stone, Place Table, Place Furnace, Place Plant,
Make Wood Pickaxe, Make Stone Pickaxe, Make Iron Pickaxe,
Make Wood Sword, Make Stone Sword, Make Iron Sword.

Output format:
<reasoning>
[Your brief step-by-step thought process]
</reasoning>
ACTION: <action_name>

Return only the reasoning and the ACTION line, nothing else."""
    
    def _get_cot_instruction_math(self, plan):
        """
        Get instruction for chain-of-thought reasoning (MATH domain).
        
        Args:
            plan: Current plan (optional)
            
        Returns:
            str: CoT instruction for MATH
        """
        base = "Analyze the problem carefully"
        if plan:
            base += " following your plan"
        
        return f"""{base} and solve it step by step.

You are about to solve a math problem. Your task RIGHT NOW is to choose next action from the list below.

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
- Do NOT solve the problem, do NOT include any other text before or after the tags

Output format (exact):
<action>ActionName: one-line description</action>"""
    
    def _extract_action_and_reasoning(self, response_text):
        """
        Extract action and reasoning from response.
        
        Args:
            response_text: LLM response
            
        Returns:
            tuple: (action, reasoning)
        """
        reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
        reasoning_match = re.search(reasoning_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        reasoning = None
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            text_after_reasoning = response_text[reasoning_match.end():]
            action_text = text_after_reasoning.strip()
        else:
            action_text = response_text.strip()
        
        action = self._extract_action(action_text)
        
        return action, reasoning
    
    def _extract_action(self, action_text):
        """
        Extract action from text.
        
        Args:
            action_text: Text containing action
            
        Returns:
            str: Extracted action
        """
        # Remove XML tags
        action_text = re.sub(r'<[^>]+>', '', action_text)
        
        # For MATH domain, try to extract answer first
        if self.domain == 'math':
            # Try to extract content from <action> tags first
            action_pattern = r'<action>(.*?)</action>'
            action_match = re.search(action_pattern, action_text, re.IGNORECASE | re.DOTALL)
            if action_match:
                return action_match.group(1).strip()

        # For Crafter or fallback, extract ACTION line
        action_text = re.sub(
            r'^[\s]*(?:ACTION|Action)[\s]*[:=]?\s*',
            '',
            action_text,
            flags=re.IGNORECASE
        )
        
        action = action_text.split('\n')[0].strip()
        
        return action
    
    def get_reasoning_history(self):
        """
        Get all reasoning steps from history.
        
        Returns:
            list: List of reasoning steps
        """
        return [r for r in self.reasoning_history if r is not None]
    
    def reset(self):
        """
        Reset reasoning history.
        """
        self.reasoning_history = []