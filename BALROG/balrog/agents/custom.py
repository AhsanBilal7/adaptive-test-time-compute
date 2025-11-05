import re
import copy
import time
from balrog.agents.base import BaseAgent

from balrog.agents.tool_selector import ToolSelector
from balrog.agents.compute_selector import ComputeSelector
from balrog.agents.prm_model import PRMModel
from balrog.agents.reasoners.reactive_actor import ReactiveActorReasoner
from balrog.agents.reasoners.cot_reasoner import CoTReasoner
from balrog.agents.reasoners.heuristic_script_reasoner import HeuristicScriptReasoner
from balrog.agents.policies.crafter_policy import CrafterPolicy
from balrog.agents.compute_strategies import get_compute_strategy


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
        fixed_tool (str): Fixed tool to always use (default: None)
            Options: 'reactive_actor', 'cot', 'heuristic_script'
            Overrides use_tool_selector when set
        fixed_compute (dict): Fixed compute config to always use (default: None)
            Format: {'strategy': str, 'param': int}
            Example: {'strategy': 'best_of_n', 'param': 3}
            Overrides use_compute_selector when set
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
        fixed_tool=None,
        fixed_compute=None,
        dataset='crafter'
    ):
        """
        Initialize the CustomAgent with mode configuration and optional selectors.
        
        Args:
            fixed_tool (str): Fixed tool name to always use. Options: 'reactive_actor', 'cot', 'heuristic_script'
                            If set, overrides use_tool_selector (sets it to False)
            fixed_compute (dict): Fixed compute config to always use. Format: {'strategy': str, 'param': int}
                                Example: {'strategy': 'best_of_n', 'param': 3}
                                If set, overrides use_compute_selector (sets it to False)
        """
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        
        # Debug confirmation of agent initialization
        print("[DEBUG] ✅ CustomAgent initialized with multi-tool chaining")
        
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
        self.dataset = dataset
        
        self.fixed_tool = fixed_tool
        
        if fixed_tool is not None:
            valid_tools = ['reactive_actor', 'cot', 'heuristic_script']
            if fixed_tool not in valid_tools:
                raise ValueError(
                    f"fixed_tool must be one of {valid_tools}, got {fixed_tool}"
                )
            self.use_tool_selector = False
        else:
            self.use_tool_selector = use_tool_selector
        
        # Handle fixed_compute: treat null/empty dict as "not set"
        if fixed_compute is not None and isinstance(fixed_compute, dict):
            # Check if it's a valid config (not all null values)
            has_valid_strategy = 'strategy' in fixed_compute and fixed_compute['strategy'] is not None
            has_valid_param = 'param' in fixed_compute and fixed_compute['param'] is not None
            
            if has_valid_strategy and has_valid_param:
                # Valid fixed_compute config
                valid_strategies = ['best_of_n', 'beam_search', 'lookahead']
                if fixed_compute['strategy'] not in valid_strategies:
                    raise ValueError(
                        f"fixed_compute strategy must be one of {valid_strategies}"
                    )
                self.use_compute_selector = False
                self.fixed_compute = fixed_compute
            elif has_valid_strategy or has_valid_param:
                # Partial config - error
                raise ValueError(
                    "fixed_compute must have both 'strategy' and 'param' set, or both should be null"
                )
            else:
                # All null values - treat as "not set"
                self.fixed_compute = None
                self.use_compute_selector = use_compute_selector
        else:
            self.fixed_compute = None
            self.use_compute_selector = use_compute_selector
        
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
        
        if self.use_tool_selector or self.use_compute_selector or fixed_tool or fixed_compute:
            self._initialize_selectors_and_reasoners()
    
    def _initialize_selectors_and_reasoners(self):
        """
        Initialize tool selector, compute selector, PRM, and reasoners (including new tools).
        """
        if self.use_tool_selector:
            self.tool_selector = ToolSelector(self.client)
        
        if self.use_compute_selector:
            self.compute_selector = ComputeSelector(self.client)
        
        self.prm_model = PRMModel(self.client)
        
        # Import new tools
        from balrog.agents.reasoners import (
            NumericVerifier, SummarizerTool, ReframeTool,
            VerifierTool, WebSearchTool
        )
        
        reactive_actor = ReactiveActorReasoner(self.client)
        cot_reasoner = CoTReasoner(self.client)
        
        policy = self._get_policy_for_dataset(self.dataset)
        heuristic_reasoner = HeuristicScriptReasoner(policy)
        
        # Initialize new tools
        self.numeric_verifier = NumericVerifier(self.prm_model)
        self.verifier = VerifierTool(self.prm_model)
        self.summarizer = SummarizerTool(self.client)
        self.reframe_tool = ReframeTool(self.client)
        self.web_tool = WebSearchTool(self.client)
        
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
        
        if planning_decision == 1 and (self.use_tool_selector or self.use_compute_selector or self.fixed_tool or self.fixed_compute):
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
        Now supports multi-tool chaining: sequential execution of multiple tools.
        """
        # Get tool selection (now returns {"tools": [...]})
        if self.fixed_tool:
            selected_tools = [self.fixed_tool]
        elif self.use_tool_selector and self.tool_selector:
            tool_selection = self.tool_selector.select_tool(obs, self.plan, messages)
            selected_tools = tool_selection.get('tools', ['reactive_actor'])
        else:
            selected_tools = ['reactive_actor']
        
        # Get compute config
        compute_config = {'strategy': 'best_of_n', 'param': 1}
        
        if self.fixed_compute:
            compute_config = self.fixed_compute
        elif self.use_compute_selector and self.compute_selector:
            # Use first tool for compute selection
            compute_config = self.compute_selector.select_compute_strategy(
                obs, self.plan, selected_tools[0], messages
            )
        
        if not self.use_tool_selector and not self.use_compute_selector and not self.fixed_tool and not self.fixed_compute:
            return default_action
        
        if not self.reasoners:
            return default_action
        
        # Context for chaining tools
        context = {
            'obs': obs,
            'plan': self.plan,
            'messages': messages.copy() if messages else [],
            'last_action': None
        }
        
        action = default_action
        
        # Execute tools sequentially with timing and token tracking
        for tool_name in selected_tools:
            start_time = time.time()
            tokens_used = 0
            
            if tool_name in self.reasoners:
                # Standard reasoners
                reasoner = self.reasoners[tool_name]
                
                if compute_config['param'] == 1:
                    # Direct execution
                    if tool_name == 'heuristic_script':
                        action, rule = reasoner.generate_action(context['obs'], context['plan'])
                    elif tool_name == 'cot':
                        action, reasoning = reasoner.generate_action(context['messages'], context['plan'])
                        tokens_used = len(str(context['messages'])) // 4 + len(str(action)) // 4
                    else:
                        action = reasoner.generate_action(context['messages'], context['plan'])
                        tokens_used = len(str(context['messages'])) // 4 + len(str(action)) // 4
                    
                    metadata = {
                        'tool': tool_name,
                        'compute_strategy': 'direct',
                        'compute_config': compute_config,
                        'latency_ms': round((time.time() - start_time) * 1000, 2),
                        'tokens_used': tokens_used,
                    }
                else:
                    # Compute strategy execution
                    compute_strategy = get_compute_strategy(
                        compute_config['strategy'],
                        self.prm_model
                    )
                    
                    action, metadata = compute_strategy.execute(
                        reasoner, context['messages'], context['obs'], context['plan'], compute_config
                    )
                    
                    metadata['tool'] = tool_name
                    metadata['latency_ms'] = round((time.time() - start_time) * 1000, 2)
                    # Estimate tokens for N generations
                    tokens_per_gen = len(str(context['messages'])) // 4 + 50
                    metadata['tokens_used'] = tokens_per_gen * compute_config['param']
                
                # Update context
                context['last_action'] = action
                context['messages'].append({'role': 'assistant', 'content': str(action)})
                self.compute_metadata_history.append(metadata)
                
            elif tool_name == 'numeric_verifier':
                # Numeric verification
                result = self.numeric_verifier.verify(
                    context['messages'], context['obs'], context['plan'],
                    context.get('last_action', action)
                )
                tokens_used = len(str(context['messages'])) // 4 + 50
                context['messages'].append({'role': 'system', 'content': str(result)})
                self.compute_metadata_history.append({
                    'tool': 'numeric_verifier',
                    'result': result,
                    'latency_ms': round((time.time() - start_time) * 1000, 2),
                    'tokens_used': tokens_used
                })
                
            elif tool_name == 'verifier':
                # General verification
                result = self.verifier.verify(
                    context['messages'], context['obs'], context['plan'],
                    context.get('last_action', action)
                )
                tokens_used = len(str(context['messages'])) // 4 + 50
                context['messages'].append({'role': 'system', 'content': str(result)})
                self.compute_metadata_history.append({
                    'tool': 'verifier',
                    'result': result,
                    'latency_ms': round((time.time() - start_time) * 1000, 2),
                    'tokens_used': tokens_used
                })
                
            elif tool_name == 'summarizer':
                # Summarize reasoning chain
                summary = self.summarizer.summarize(str(context['messages']))
                tokens_used = len(str(context['messages'])) // 4 + len(summary) // 4
                context['messages'].append({'role': 'system', 'content': f'Summary: {summary}'})
                self.compute_metadata_history.append({
                    'tool': 'summarizer',
                    'summary': summary,
                    'latency_ms': round((time.time() - start_time) * 1000, 2),
                    'tokens_used': tokens_used
                })
                
            elif tool_name == 'reframe':
                # Reframe question/plan
                reframed = self.reframe_tool.reframe(str(context['obs']), context['plan'])
                tokens_used = len(str(context['obs'])) // 4 + len(reframed) // 4
                context['messages'].append({'role': 'system', 'content': f'Reframed: {reframed}'})
                self.compute_metadata_history.append({
                    'tool': 'reframe',
                    'reframed': reframed,
                    'latency_ms': round((time.time() - start_time) * 1000, 2),
                    'tokens_used': tokens_used
                })
                
            elif tool_name == 'web_search':
                # Web search
                results = self.web_tool.search(context['obs'])
                tokens_used = len(str(context['obs'])) // 4 + len(results) // 4
                context['messages'].append({'role': 'system', 'content': f'Web results: {results}'})
                self.compute_metadata_history.append({
                    'tool': 'web_search',
                    'results': results,
                    'latency_ms': round((time.time() - start_time) * 1000, 2),
                    'tokens_used': tokens_used
                })
        
        # Normalize action before returning
        normalized_action = self._normalize_action(action)
        
        # Debug logging for action verification
        print(f"[DEBUG] Tools used: {selected_tools}, Final action (normalized): {normalized_action} | raw: {action}")
        
        return normalized_action
    
    def _normalize_action(self, text):
        """
        Normalize action text to valid Crafter action tokens.
        
        Args:
            text: Raw action text from reasoners
            
        Returns:
            str: Normalized action token (noop, left, right, jump, use)
        """
        if not text:
            return "noop"
        
        t = str(text).strip().lower()
        
        # Handle common phrasing
        if "left" in t:
            return "left"
        if "right" in t:
            return "right"
        if "jump" in t:
            return "jump"
        if any(k in t for k in ("use", "mine", "attack", "interact", "open", "chop", "craft", "collect", "gather")):
            return "use"
        
        # Accept exact tokens
        if t in {"noop", "left", "right", "jump", "use"}:
            return t
        
        # Default fallback
        return "noop"
    
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
            'fixed_tool': self.fixed_tool,
            'fixed_compute': self.fixed_compute,
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