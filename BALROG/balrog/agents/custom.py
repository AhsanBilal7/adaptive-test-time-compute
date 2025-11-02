import re
import copy
from balrog.agents.base import BaseAgent

from tool_selector import ToolSelector
from compute_selector import ComputeSelector
from prm_model import PRMModel
from reasoners.reactive_actor import ReactiveActorReasoner
from reasoners.cot_reasoner import CoTReasoner
from reasoners.heuristic_script_reasoner import HeuristicScriptReasoner
from policies.crafter_policy import CrafterPolicy
from compute_strategies import get_compute_strategy


class CustomAgent(BaseAgent):
    """
    Dual-mode agent supporting dynamic and fixed-frequency planning
    with optional tool selection and compute selection.
    
    Modes:
    - 'dynamic': Agent decides when to plan (dt ∈ {0,1})
    - 'fixed': Agent plans every K steps (dt = 1/K)
    
    Parameters for initialization:
        mode (str): 'dynamic' or 'fixed' (default: 'dynamic')
        planning_frequency (int): K value for 'fixed' mode (required if mode='fixed')
        use_planner (bool): Enable planning (default: True)
        use_tool_selector (bool): Enable tool selector (default: False)
        use_compute_selector (bool): Enable compute selector (default: False)
        dataset (str): Dataset name for heuristic policy (default: 'crafter')
    """
    
    def __init__(
        self, 
        client_factory, 
        prompt_builder, 
        mode='dynamic', 
        planning_frequency=None,
        use_planner=True,
        use_tool_selector=False,
        use_compute_selector=False,
        dataset='crafter'
    ):
        """
        Initialize the CustomAgent with mode configuration and optional selectors.
        """
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        self.mode = mode
        self.planning_frequency_k = planning_frequency
        
        if self.mode == 'fixed':
            if planning_frequency is None or planning_frequency < 1:
                raise ValueError(
                    f"planning_frequency must be ≥ 1 for 'fixed' mode, got {planning_frequency}"
                )
        elif self.mode != 'dynamic':
            raise ValueError(
                f"mode must be 'dynamic' or 'fixed', got {mode}"
            )
        
        self.use_planner = use_planner
        self.use_tool_selector = use_tool_selector
        self.use_compute_selector = use_compute_selector
        self.dataset = dataset
        
        self.plan = None
        self.timestep = 0
        
        self.planning_decisions = []
        self.plan_history = []
        self.plan_length = []
        
        self.tool_selector = None
        self.compute_selector = None
        self.prm_model = None
        self.reasoners = {}
        self.compute_metadata_history = []
        
        if self.use_tool_selector or self.use_compute_selector:
            self._initialize_selectors_and_reasoners()
    
    def _initialize_selectors_and_reasoners(self):
        """
        Initialize tool selector, compute selector, PRM, and reasoners.
        """
        if self.use_tool_selector:
            self.tool_selector = ToolSelector(self.client)
        
        if self.use_compute_selector:
            self.compute_selector = ComputeSelector(self.client)
        
        self.prm_model = PRMModel(self.client)
        
        reactive_actor = ReactiveActorReasoner(self.client)
        cot_reasoner = CoTReasoner(self.client)
        
        policy = self._get_policy_for_dataset(self.dataset)
        heuristic_reasoner = HeuristicScriptReasoner(policy)
        
        self.reasoners = {
            'reactive_actor': reactive_actor,
            'cot': cot_reasoner,
            'heuristic_script': heuristic_reasoner
        }
    
    def _get_policy_for_dataset(self, dataset):
        """
        Get rule-based policy for specified dataset.
        """
        if dataset.lower() == 'crafter':
            return CrafterPolicy()
        else:
            return CrafterPolicy()
    
    def act(self, obs, prev_action=None):
        """
        Generate action with optional planning, tool selection, and compute selection.
        
        All functionality controlled by config parameters (use_planner, use_tool_selector, use_compute_selector).
        
        Args:
            obs: Current observation
            prev_action: Previous action (optional)
        
        Returns:
            Response with reasoning (plan or None) and completion (action)
        """
        self.timestep += 1
        
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        
        messages = self.prompt_builder.get_prompt()
        
        if not self.use_planner:
            instruction = self._get_action_only_instruction()
            if messages and messages[-1].role == "user":
                messages[-1].content += "\n\n" + instruction
            
            response = self.client.generate(messages)
            action = self._clean_action(response.completion)
            
            self.planning_decisions.append(0)
            self.plan_length.append(0)
            
            response = response._replace(reasoning=None, completion=action)
            return response
        
        if self.mode == 'dynamic':
            instruction = self._get_dynamic_instruction()
        else:
            should_plan = (self.timestep - 1) % self.planning_frequency_k == 0
            instruction = (
                self._get_fixed_planning_instruction() 
                if should_plan 
                else self._get_action_only_instruction()
            )
        
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + instruction
        
        response = self.client.generate(messages)
        
        plan, action, planning_decision = self._extract_plan_and_action(
            response.completion,
            self.mode
        )
        
        if plan is not None:
            self.plan = plan
        
        if planning_decision == 1 and (self.use_tool_selector or self.use_compute_selector):
            action = self._execute_with_selectors(obs, messages, action)
        
        self.planning_decisions.append(planning_decision)
        if plan:
            self.plan_history.append(plan)
            self.plan_length.append(len(plan.split()))
        else:
            self.plan_length.append(0)
        
        response = response._replace(reasoning=plan, completion=action)
        
        return response
    
    def _execute_with_selectors(self, obs, messages, default_action):
        """
        Execute action generation with tool and compute selection.
        """
        tool_name = 'reactive_actor'
        if self.use_tool_selector and self.tool_selector:
            tool_name = self.tool_selector.select_tool(obs, self.plan, messages)
        
        compute_config = {'strategy': 'best_of_n', 'param': 1}
        if self.use_compute_selector and self.compute_selector:
            compute_config = self.compute_selector.select_compute_strategy(
                obs, self.plan, tool_name, messages
            )
        
        if not self.use_tool_selector and not self.use_compute_selector:
            return default_action
        
        if not self.reasoners:
            return default_action
        
        reasoner = self.reasoners.get(tool_name, self.reasoners.get('reactive_actor'))
        
        if compute_config['param'] == 1:
            if tool_name == 'heuristic_script':
                action, rule = reasoner.generate_action(obs, self.plan)
            elif tool_name == 'cot':
                action, reasoning = reasoner.generate_action(messages, self.plan)
            else:
                action = reasoner.generate_action(messages, self.plan)
            
            metadata = {
                'tool': tool_name,
                'compute_strategy': 'direct',
                'compute_config': compute_config
            }
        else:
            compute_strategy = get_compute_strategy(
                compute_config['strategy'], 
                self.prm_model
            )
            
            action, metadata = compute_strategy.execute(
                reasoner, messages, obs, self.plan, compute_config
            )
            
            metadata['tool'] = tool_name
        
        self.compute_metadata_history.append(metadata)
        
        return action
    
    def _get_dynamic_instruction(self):
        """
        Get planning instruction for dynamic mode.
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
        Get instruction for planning step in fixed-frequency mode.
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
        Get instruction for action-only step in fixed-frequency mode.
        """
        return """Look at your previous plan and observations, then choose exactly ONE 
action from the allowed actions listed previously.

Output no other text except the action."""
    
    def _extract_plan_and_action(self, response_text, mode):
        """
        Extract plan and action from LLM response.
        """
        plan_pattern = r'<plan>(.*?)</plan>'
        plan_match = re.search(plan_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        plan = None
        planning_decision = 0
        
        if plan_match:
            plan = plan_match.group(1).strip()
            planning_decision = 1
            
            text_after_plan = response_text[plan_match.end():]
            action = text_after_plan.strip()
        else:
            action = response_text.strip()
        
        action = self._clean_action(action)
        
        return plan, action, planning_decision
    
    def _clean_action(self, action_text):
        """
        Clean extracted action text.
        """
        action_text = re.sub(r'<[^>]+>', '', action_text)
        
        action_text = re.sub(
            r'^[\s]*(?:ACTION|Action)[\s]*[:=]?\s*',
            '',
            action_text,
            flags=re.IGNORECASE
        )
        
        action = action_text.split('\n')[0].strip()
        
        return action
    
    def get_planning_stats(self):
        """
        Get statistics about planning behavior during episode.
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
            'use_planner': self.use_planner,
            'use_tool_selector': self.use_tool_selector,
            'use_compute_selector': self.use_compute_selector,
        }
        
        if self.mode == 'fixed':
            stats['planning_frequency_k'] = self.planning_frequency_k
            stats['expected_planning_frequency'] = 1.0 / self.planning_frequency_k
        
        if self.use_tool_selector and self.tool_selector:
            stats['tool_selection'] = self.tool_selector.get_tool_distribution()
        
        if self.use_compute_selector and self.compute_selector:
            stats['compute_selection'] = self.compute_selector.get_compute_distribution()
        
        if self.prm_model:
            stats['prm_scoring'] = self.prm_model.get_scoring_stats()
        
        return stats
    
    def get_planning_frequency(self):
        """
        Get the actual planning frequency.
        """
        if not self.planning_decisions:
            return 0.0
        return sum(self.planning_decisions) / len(self.planning_decisions)
    
    def get_plan_history(self):
        """
        Get all plans generated during episode.
        """
        return self.plan_history.copy()
    
    def get_compute_metadata(self):
        """
        Get metadata about compute strategy usage.
        """
        return self.compute_metadata_history.copy()
    
    def reset_metrics(self):
        """
        Reset all metrics for a new episode.
        """
        self.planning_decisions = []
        self.plan_history = []
        self.plan_length = []
        self.timestep = 0
        self.plan = None
        self.compute_metadata_history = []
        
        if self.tool_selector:
            self.tool_selector.reset()
        if self.compute_selector:
            self.compute_selector.reset()
        if self.prm_model:
            self.prm_model.reset()
        
        if self.reasoners:
            cot = self.reasoners.get('cot')
            if cot and hasattr(cot, 'reset'):
                cot.reset()
            
            heuristic = self.reasoners.get('heuristic_script')
            if heuristic and hasattr(heuristic, 'reset'):
                heuristic.reset()