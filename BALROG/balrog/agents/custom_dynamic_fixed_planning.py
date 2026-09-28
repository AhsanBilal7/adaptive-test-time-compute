"""
Simplified CustomAgent with dual-mode support:
1. Dynamic Planning Mode: Agent decides when to plan
2. Fixed-Frequency Planning Mode: Agent plans every K steps

Minimal function structure matching original BALROG patterns.
"""

import re
import copy
from balrog.agents.base import BaseAgent


class CustomAgent(BaseAgent):
    """
    Dual-mode agent supporting dynamic and fixed-frequency planning.
    
    Modes:
    - 'dynamic': Agent decides when to plan (dt ∈ {0,1})
    - 'fixed': Agent plans every K steps (dt = 1/K)
    
    Parameters for initialization:
        mode (str): 'dynamic' or 'fixed' (default: 'dynamic')
        planning_frequency (int): K value for 'fixed' mode (required if mode='fixed')
    
    Example usage:
        # Dynamic mode (agent decides)
        agent = CustomAgent(client_factory, prompt_builder, mode='dynamic')
        
        # Fixed mode (every 4 steps)
        agent = CustomAgent(client_factory, prompt_builder, mode='fixed', planning_frequency=4)
        
        # Via hydra config:
        # agent.mode=dynamic
        # agent.planning_frequency=4
    """
    
    def __init__(self, client_factory, prompt_builder, mode='dynamic', planning_frequency=None):
        """Initialize the CustomAgent with mode configuration."""
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Mode configuration
        self.mode = mode
        self.planning_frequency_k = planning_frequency
        
        # Validate configuration
        if self.mode == 'fixed':
            if planning_frequency is None or planning_frequency < 1:
                raise ValueError(
                    f"planning_frequency must be ≥ 1 for 'fixed' mode, got {planning_frequency}"
                )
        elif self.mode != 'dynamic':
            raise ValueError(
                f"mode must be 'dynamic' or 'fixed', got {mode}"
            )
        
        # Internal state
        self.plan = None
        self.timestep = 0
        
        # Metrics (optional, for analysis)
        self.planning_decisions = []
        self.plan_history = []
        self.plan_length = []
    
    def act(self, obs, prev_action=None):
        """
        Generate action with optional planning (dynamic or fixed-frequency).
        
        Args:
            obs: Current observation
            prev_action: Previous action (optional)
        
        Returns:
            Response with reasoning (plan or None) and completion (action)
        """
        self.timestep += 1
        
        # Update context
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        # Get prompt and add planning instruction
        messages = self.prompt_builder.get_prompt()
        
        if self.mode == 'dynamic':
            # Dynamic mode: Agent decides whether to plan
            instruction = self._get_dynamic_instruction()
        else:  # 'fixed' mode
            # Fixed mode: Enforce planning schedule
            should_plan = (self.timestep - 1) % self.planning_frequency_k == 0
            instruction = (
                self._get_fixed_planning_instruction() 
                if should_plan 
                else self._get_action_only_instruction()
            )
        
        # Append instruction to prompt
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + instruction

        # Generate response
        response = self.client.generate(messages)
        
        # Parse response and extract plan and action
        plan, action, planning_decision = self._extract_plan_and_action(
            response.completion,
            self.mode
        )
        
        # Update plan state
        if plan is not None:
            self.plan = plan
        
        # Track metrics
        self.planning_decisions.append(planning_decision)
        if plan:
            self.plan_history.append(plan)
            self.plan_length.append(len(plan.split()))
        else:
            self.plan_length.append(0)
        
        # Format response: reasoning=plan, completion=action
        response = response._replace(reasoning=plan, completion=action)
        
        return response
    
    def _get_dynamic_instruction(self):
        """
        Get planning instruction for dynamic mode (Prompt 20 from paper).
        
        Allows agent to decide when to plan based on context.
        """
        return """Review your current plan and observations.
• If you do not have a plan yet, create one.
• If your plan is outdated or needs changes, create a new plan.

If you create a new plan, output it in the following format:
<plan>YOUR NEW PLAN</plan>
Replace YOUR NEW PLAN with your revised plan.

If your current plan is still valid, proceed without outputting it again.

After this evaluation (and any necessary replanning), output exactly ONE allowed action.
Output nothing else except an optional <plan>...</plan> block and that single action."""
    
    def _get_fixed_planning_instruction(self):
        """
        Get instruction for planning step in fixed-frequency mode (Prompt 19).
        
        Instructs agent to generate a plan at this step.
        """
        return """Review your previous observations and plan, then make a high-level plan 
for completing the task. Your plan can include reasoning about how to solve the task.

After this planning phase, you will be asked to take actions one at a time.

Output your plan strictly in the following format:
<plan>YOUR PLAN</plan>
Replace YOUR PLAN with your own thinking and plan.

After your plan, choose exactly ONE action from the allowed actions listed previously.
Output no other text except your plan and action."""
    
    def _get_action_only_instruction(self):
        """
        Get instruction for action-only step in fixed-frequency mode (Prompt 18).
        
        Instructs agent to output ONLY action (no plan).
        """
        return """Look at your previous plan and observations, then choose exactly ONE 
action from the allowed actions listed previously.

Output no other text except the action."""
    
    def _extract_plan_and_action(self, response_text, mode):
        """
        Extract plan and action from LLM response.
        
        Handles both dynamic and fixed-frequency modes.
        
        Args:
            response_text (str): Raw LLM completion
            mode (str): 'dynamic' or 'fixed'
        
        Returns:
            tuple: (plan, action, planning_decision)
                - plan: str or None (None if no planning)
                - action: str (extracted action)
                - planning_decision: int (0 or 1, dt value)
        """
        # Search for <plan>...</plan> tags
        plan_pattern = r'<plan>(.*?)</plan>'
        plan_match = re.search(plan_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        plan = None
        planning_decision = 0
        
        if plan_match:
            # Found plan tags
            plan = plan_match.group(1).strip()
            planning_decision = 1
            
            # Extract action after </plan>
            text_after_plan = response_text[plan_match.end():]
            action = text_after_plan.strip()
        else:
            # No plan tags: entire response is action
            action = response_text.strip()
        
        # Clean action text
        action = self._clean_action(action)
        
        return plan, action, planning_decision
    
    def _clean_action(self, action_text):
        """
        Clean extracted action text.
        
        Args:
            action_text (str): Raw action text
        
        Returns:
            str: Cleaned action
        """
        # Remove XML tags
        action_text = re.sub(r'<[^>]+>', '', action_text)
        
        # Remove "Action:" prefixes
        action_text = re.sub(
            r'^[\s]*(?:ACTION|Action)[\s]*[:=]?\s*',
            '',
            action_text,
            flags=re.IGNORECASE
        )
        
        # Get first line only
        action = action_text.split('\n')[0].strip()
        
        return action
    
    # ========================================================================
    # METRICS METHODS (Optional - for analysis only)
    # ========================================================================
    
    def get_planning_stats(self):
        """
        Get statistics about planning behavior during episode.
        
        Returns:
            dict: Planning metrics including frequency, token count, etc.
        """
        if not self.planning_decisions:
            return {}
        
        total_plans = sum(self.planning_decisions)
        total_tokens = sum(self.plan_length)
        plan_frequency = total_plans / len(self.planning_decisions)
        avg_plan_length = total_tokens / total_plans if total_plans > 0 else 0
        
        stats = {
            'total_timesteps': self.timestep,
            'total_plans': total_plans,
            'planning_frequency': plan_frequency,
            'avg_plan_length': avg_plan_length,
            'total_tokens_in_plans': total_tokens,
            'plan_history_count': len(self.plan_history),
            'mode': self.mode,
        }
        
        if self.mode == 'fixed':
            stats['planning_frequency_k'] = self.planning_frequency_k
            stats['expected_planning_frequency'] = 1.0 / self.planning_frequency_k
        
        return stats
    
    def get_planning_frequency(self):
        """Get the actual planning frequency (0.0 to 1.0)."""
        if not self.planning_decisions:
            return 0.0
        return sum(self.planning_decisions) / len(self.planning_decisions)
    
    def get_plan_history(self):
        """Get all plans generated during episode."""
        return self.plan_history.copy()
    
    def reset_metrics(self):
        """Reset all metrics for a new episode."""
        self.planning_decisions = []
        self.plan_history = []
        self.plan_length = []
        self.timestep = 0
        self.plan = None


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
USAGE EXAMPLES FOR DIFFERENT STRATEGIES:

1. DYNAMIC PLANNING (Paper's main approach):
   ──────────────────────────────────────────
   agent = create_dynamic_agent(client_factory, prompt_builder)
   
   Expected behavior:
   - Agent decides when to plan based on context
   - Variable dt ∈ {0,1}
   - Matches paper's Section 5.3 (RL-trained agent)


2. ALWAYS PLANNING BASELINE (ReAct):
   ──────────────────────────────────
   agent = create_always_planning_agent(client_factory, prompt_builder)
   
   Expected behavior:
   - Plans every step (dt=1 always)
   - High token cost, moderate performance
   - Matches paper's Section 5.1 (always-plan baseline)


3. NEVER PLANNING BASELINE:
   ────────────────────────
   agent = create_never_planning_agent(client_factory, prompt_builder)
   
   Expected behavior:
   - Never plans (dt=0 always)
   - Low token cost, lower performance
   - Matches paper's Section 5.1 (never-plan baseline)


4. FIXED-FREQUENCY BASELINE (e.g., every 4 steps):
   ────────────────────────────────────────────────
   agent = create_fixed_frequency_agent(client_factory, prompt_builder, k=4)
   
   Expected behavior:
   - Plans at steps 1, 5, 9, 13, ...
   - Moderate token cost
   - Matches paper's Section 5.1 (fixed-frequency baseline)


5. DIRECT INSTANTIATION WITH CUSTOM MODE:
   ──────────────────────────────────────
   # Dynamic mode
   agent = EnhancedDynamicPlanningAgent(
       client_factory,
       prompt_builder,
       mode='dynamic'
   )
   
   # Fixed mode with k=2
   agent = EnhancedDynamicPlanningAgent(
       client_factory,
       prompt_builder,
       mode='fixed',
       planning_frequency=2
   )


6. EVALUATION COMMAND EXAMPLES:
   ────────────────────────────
   
   # Dynamic planning (main approach):
   python eval.py \\
     agent.type=custom \\
     agent.class_name=EnhancedDynamicPlanningAgent \\
     agent.mode=dynamic \\
     ...
   
   # Always planning baseline:
   python eval.py \\
     agent.type=custom \\
     agent.class_name=EnhancedDynamicPlanningAgent \\
     agent.mode=fixed \\
     agent.planning_frequency=1 \\
     ...
   
   # Plan every 4 steps:
   python eval.py \\
     agent.type=custom \\
     agent.class_name=EnhancedDynamicPlanningAgent \\
     agent.mode=fixed \\
     agent.planning_frequency=4 \\
     ...


7. METRICS AND ANALYSIS:
   ─────────────────────
   
   stats = agent.get_planning_stats()
   print(f"Planning frequency: {stats['planning_frequency']}")
   print(f"Total plans: {stats['total_plans']}")
   print(f"Average plan length: {stats['avg_plan_length']}")
   
   # Can compare across different modes:
   # - Dynamic: fp ∈ (0, 1) - variable
   # - Always: fp = 1.0
   # - Never: fp = 0.0
   # - Every K: fp = 1/K


8. RESET FOR MULTIPLE EPISODES:
   ────────────────────────────
   
   for episode in range(10):
       agent.reset_metrics()  # Start fresh for new episode
       # ... run episode ...
       stats = agent.get_planning_stats()
       print(f"Episode {episode}: {stats}")
"""


