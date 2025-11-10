import re
import copy
import time
import json
from pathlib import Path
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm
from pydantic import BaseModel, Field, ConfigDict
from BALROG.balrog.schemas import FinalAnswer
from BALROG.balrog.agents.tool_selector import ToolSelector
from BALROG.balrog.agents.compute_selector import ComputeSelector
from BALROG.balrog.agents.prm_model import PRMModel
from BALROG.balrog.agents.reasoners.reactive_actor import ReactiveActorReasoner
from BALROG.balrog.agents.reasoners.cot_reasoner import CoTReasoner
from BALROG.balrog.agents.reasoners.heuristic_script_reasoner import HeuristicScriptReasoner
from BALROG.balrog.agents.policies.crafter_policy import CrafterPolicy
from BALROG.balrog.agents.compute_strategies import get_compute_strategy
import json

def get_field_from_completion(completion_text: str, field: str, default=None):
    """
    completion_text: the JSON string (e.g. structured_response.completion)
    field: key name to extract. Supports dotted paths like "tool.name".
    default: value to return if the field is missing.
    """
    # 1) parse JSON string into Python dict
    try:
        data = json.loads(completion_text)   # use loads() for string input
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in completion_text: {e}")

    # 2) walk through nested keys if "a.b.c" style is given
    current = data
    for part in field.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default

    return current


def _get(obj, key, default=None):
    """Helper to get attribute from object, dict, or OmegaConf."""
    if obj is None:
        return default
    val = getattr(obj, key, None)
    if val is not None:
        return val
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return default


def _provided(x):
    """Check if value is meaningfully provided."""
    return x is not None and not (isinstance(x, str) and x.strip() == "")





@dataclass
class MathResponse:
    """Response from MATH agent containing reasoning and answer."""
    reasoning: Optional[str]
    answer: str
    plan: Optional[str]
    metadata: Dict[str, Any]


class MathPromptBuilder:
    """Builds prompts for MATH problem solving with memory."""
    
    def __init__(self, max_text_history: int = 16, remember_cot: bool = True):
        self.max_text_history = max_text_history
        self.remember_cot = remember_cot
        self.problem = None
        self.history = []
        
    def set_problem(self, problem: str):
        """Set the current math problem."""
        self.problem = problem
        self.history = []
        
    def update_step(self, reasoning: str, answer: str):
        """Update history with reasoning step."""
        if self.remember_cot and reasoning:
            self.history.append({
                'role': 'assistant',
                'content': f"Reasoning: {reasoning}\nAnswer: {answer}"
            })
            # Trim history if needed
            if len(self.history) > self.max_text_history:
                self.history = self.history[-self.max_text_history:]
    
    def get_prompt(self, plan: Optional[str] = None) -> List[Dict[str, str]]:
        """Build the full prompt with problem, history, and optional plan."""
        messages = []
        
        # System message
        system_msg = """You are a mathematical problem solver. Solve problems step by step.
When solving, you can use the following format:
- For planning: <plan>YOUR SOLUTION APPROACH</plan>
- For final answer: <answer>YOUR FINAL ANSWER</answer>

Show your work clearly and provide the final answer in the specified format."""
        
        messages.append({'role': 'system', 'content': system_msg})
        
        # Problem statement
        problem_msg = f"Problem: {self.problem}"
        if plan:
            problem_msg += f"\n\nCurrent plan: {plan}"
        
        messages.append({'role': 'user', 'content': problem_msg})
        
        # Add history
        messages.extend(self.history)
        
        return messages


class MathAgent:
    """
    Agent for solving MATH dataset problems with planning, tool selection, and compute strategies.
    
    Supports:
    - Dynamic or fixed-frequency planning
    - Multi-tool chaining (CoT, reactive, verifiers, etc.)
    - Compute strategies (best-of-n, beam search, lookahead)
    - Memory of reasoning steps
    """
    
    def __init__(
        self,
        client_factory,
        mode: str = 'dynamic',
        planning_frequency: Optional[int] = None,
        use_planner: bool = True,
        use_tool_selector: bool = False,
        use_compute_selector: bool = False,
        fixed_tool: Optional[str] = None,
        fixed_compute: Optional[Dict] = None,
        remember_cot: bool = True,
        max_text_history: int = 16,
        max_image_history: int = 0,
        domain: str = 'math'
    ):
        """
        Initialize MATH agent.
        
        Args:
            client_factory: Factory to create LLM client
            mode: 'dynamic' or 'fixed' planning mode
            planning_frequency: K value for fixed mode
            use_planner: Enable planning
            use_tool_selector: Enable tool selection
            use_compute_selector: Enable compute selection
            fixed_tool: Fixed tool name to always use
            fixed_compute: Fixed compute config {'strategy': str, 'param': int}
            remember_cot: Remember chain-of-thought in history
            max_text_history: Max reasoning steps to keep in context
            max_image_history: Not used for MATH (kept for compatibility)
            domain: 'crafter' or 'math' - determines reasoner instructions (default: 'math')
        """
        self.client = client_factory()
        self.prompt_builder = MathPromptBuilder(max_text_history, remember_cot)
        self.domain = domain
        
        print(f"[DEBUG] ✅ MathAgent initialized with mode={mode}, domain={domain}")
        
        self.mode = mode
        self.planning_frequency_k = planning_frequency
        
        if self.mode == 'fixed':
            if planning_frequency is None or planning_frequency < 1:
                raise ValueError(
                    f"planning_frequency must be ≥ 1 for 'fixed' mode, got {planning_frequency}"
                )
        elif self.mode != 'dynamic':
            raise ValueError(f"mode must be 'dynamic' or 'fixed', got {mode}")
        
        self.use_planner = use_planner
        self.fixed_tool = fixed_tool
        
        # Handle fixed tool
        if fixed_tool is not None:
            valid_tools = ['reactive_actor', 'cot', 'heuristic_script']
            if fixed_tool not in valid_tools:
                raise ValueError(f"fixed_tool must be one of {valid_tools}, got {fixed_tool}")
            self.use_tool_selector = False
        else:
            self.use_tool_selector = use_tool_selector
        
        # Handle fixed compute
        self.use_compute_selector = use_compute_selector
        self.fixed_compute = None
        
        fc = fixed_compute
        strategy = _get(fc, "strategy")
        param = _get(fc, "param")
        
        if _provided(strategy) and _provided(param):
            valid_strategies = ["best_of_n", "beam_search", "lookahead"]
            if strategy not in valid_strategies:
                raise ValueError(f"fixed_compute strategy must be one of {valid_strategies}, got {strategy}")
            if isinstance(param, str):
                param = int(param) if param.isdigit() else float(param)
            self.fixed_compute = {"strategy": strategy, "param": param}
            self.use_compute_selector = False
        elif _provided(strategy) or _provided(param):
            raise ValueError("fixed_compute must provide BOTH 'strategy' and 'param', or neither.")
        
        self.plan = None
        self.step_count = 0
        self.planning_decisions = []
        self.plan_history = []
        self.compute_metadata_history = []
        self.tools_used = []  # Track tool names in order
        self.compute_configs_used = []  # Track compute configs in order
        
        # Initialize selectors and reasoners
        self.tool_selector = None
        self.compute_selector = None
        self.prm_model = None
        self.reasoners = {}
        
        if self.use_tool_selector or self.use_compute_selector or fixed_tool or fixed_compute:
            self._initialize_selectors_and_reasoners(domain=self.domain)
    
    def _initialize_selectors_and_reasoners(self, domain='math'):
        """
        Initialize tool selector, compute selector, PRM, and reasoners.
        
        Args:
            domain: 'crafter' or 'math' - determines instruction format (default: 'math')
        """
        if self.use_tool_selector:
            self.tool_selector = ToolSelector(self.client)
        
        if self.use_compute_selector:
            self.compute_selector = ComputeSelector(self.client)
        
        self.prm_model = PRMModel(self.client)
        
        # Import new tools
        from BALROG.balrog.agents.reasoners import (
            NumericVerifier, SummarizerTool, ReframeTool,
            VerifierTool, WebSearchTool
        )
        
        # Initialize reasoners with domain parameter
        reactive_actor = ReactiveActorReasoner(self.client, domain=domain)
        cot_reasoner = CoTReasoner(self.client, domain=domain)
        
        # For MATH, heuristic script is less relevant, but include for compatibility
        policy = CrafterPolicy()  # Placeholder, won't be used much for MATH
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
    
    def solve(self, problem: str) -> MathResponse:
        """
        Solve a MATH problem with proper execution flow.
        
        Flow:
        1. Planning decision (if use_planner=True, generate plan)
        2. Tool selection (if use_tool_selector=True, select tool)
        3. Compute selection (if use_compute_selector=True, select strategy)
        4. Execute with selected tool and compute to get final answer
        
        Args:
            problem: The math problem text
            
        Returns:
            MathResponse with reasoning, answer, plan, and metadata
        """
        self.prompt_builder.set_problem(problem)
        self.step_count = 0
        self.plan = None
        self.tools_used = []  # Reset for this problem
        self.compute_configs_used = []  # Reset for this problem
        
        # ========================================
        # PHASE 1: PLANNING (if enabled)
        # ========================================
        if not self.use_planner:
            # No planning - go directly to answer generation
            print("[DEBUG] No planning mode - generating answer directly")
            messages = self.prompt_builder.get_prompt(None)
            instruction = self._get_solve_instruction()
            messages.append({'role': 'user', 'content': instruction})
            
            response = self.client.generate_with_structured(messages, FinalAnswer.model_json_schema())
            answer = get_field_from_completion(response.completion, 'answer')

            # response = self.client.generate(messages)
            # answer = self._extract_answer(response.completion)
            
            self.planning_decisions.append(0)
            
            return MathResponse(
                reasoning=None,
                answer=answer,
                plan=None,
                metadata={
                    'mode': 'no_planning',
                    'has_plan': False,
                    'planning_decision': 0,
                    'tools_used': self.tools_used,
                    'compute_configs_used': self.compute_configs_used
                }
            )
        
        # Planning is enabled - generate plan
        self.step_count += 1
        
        # Determine if we should plan based on mode
        should_plan = True
        if self.mode == 'fixed':
            should_plan = (self.step_count - 1) % self.planning_frequency_k == 0
        
        if should_plan:
            # Generate plan
            print("[DEBUG] Planning mode - generating plan")
            messages = self.prompt_builder.get_prompt(None)
            
            if self.mode == 'dynamic':
                instruction = self._get_planning_only_instruction()
            else:
                instruction = self._get_planning_only_instruction()
            
            messages.append({'role': 'user', 'content': instruction})
            response = self.client.generate(messages)
            
            # Extract plan
            plan, remaining_text, has_plan = self._extract_plan(response.completion)
            
            if has_plan:
                self.plan = plan
                self.plan_history.append(plan)
                self.planning_decisions.append(1)
                print(f"[DEBUG] Plan generated: {plan[:100]}...")
            else:
                self.planning_decisions.append(0)
                print("[DEBUG] No plan extracted from response")
        else:
            # Fixed mode, not a planning step
            self.planning_decisions.append(0)
            print(f"[DEBUG] Fixed mode - not planning at step {self.step_count}")
        
        # ========================================
        # PHASE 2: ANSWER GENERATION
        # ========================================
        # Now we have a plan (or not), proceed to generate answer
        
        # Check if we should use selectors
        use_selectors = (
            (self.use_tool_selector or self.fixed_tool or 
             self.use_compute_selector or self.fixed_compute) and 
            self.plan is not None
        )
        
        if use_selectors:
            # Use tool and/or compute selection with structured output
            print("[DEBUG] Using selectors for answer generation")
            solution, answer = self._execute_with_selectors_and_get_answer(problem)
        else:
            # Direct answer generation without selectors
            print("[DEBUG] Direct answer generation (no selectors)")
            messages = self.prompt_builder.get_prompt(self.plan)
            instruction = self._get_solve_with_plan_instruction()
            messages.append({'role': 'user', 'content': instruction})
            
            # response = self.client.generate(messages)
            solution = self.client.generate_with_structured(messages, FinalAnswer.model_json_schema())
            answer = get_field_from_completion(solution.completion, 'answer')
            # solution = response.completion
            # answer = self._extract_answer(solution)
        
        # Update history
        self.prompt_builder.update_step(solution, answer)
        
        metadata = {
            'mode': self.mode,
            'has_plan': self.plan is not None,
            'plan': self.plan,
            'step_count': self.step_count,
            'planning_decision': self.planning_decisions[-1] if self.planning_decisions else 0,
            'compute_metadata': self.compute_metadata_history,
            'tools_used': self.tools_used,
            'compute_configs_used': self.compute_configs_used
        }
        
        print(f"[DEBUG] Final answer: {answer}")
        
        return MathResponse(
            reasoning=solution,
            answer=answer,
            plan=self.plan,
            metadata=metadata
        )

    def _execute_with_selectors_and_get_answer(self, problem: str) -> tuple:
        """
        Execute answer generation with tool and compute selection.
        Uses structured output for final answer.
        Returns (solution_text, final_answer)
        """
        messages = self.prompt_builder.get_prompt(self.plan)
        
        # ========================================
        # STEP 1: TOOL SELECTION
        # ========================================
        if self.fixed_tool:
            selected_tools = [self.fixed_tool]
            print(f"[DEBUG] Using fixed tool: {self.fixed_tool}")
        elif self.use_tool_selector and self.tool_selector:
            tool_selection = self.tool_selector.select_tool(problem, self.plan, messages)
            selected_tools = tool_selection.get('tools', ['cot'])
            print(f"[DEBUG] Tool selector chose: {selected_tools}")
        else:
            selected_tools = ['cot']
            print(f"[DEBUG] Using default tool: cot")
        
        # ========================================
        # STEP 2: COMPUTE STRATEGY SELECTION
        # ========================================
        compute_config = {'strategy': 'best_of_n', 'param': 1}
        
        if self.fixed_compute:
            compute_config = self.fixed_compute
            print(f"[DEBUG] Using fixed compute: {compute_config}")
        elif self.use_compute_selector and self.compute_selector:
            compute_config = self.compute_selector.select_compute_strategy(
                problem, self.plan, selected_tools[0], messages
            )
            print(f"[DEBUG] Compute selector chose: {compute_config}")
        else:
            print(f"[DEBUG] Using default compute: {compute_config}")
        
        # ========================================
        # STEP 3: EXECUTE TOOLS AND ACCUMULATE REASONING
        # ========================================
        if not self.reasoners:
            print("[DEBUG] No reasoners initialized, returning empty solution")
            return "", ""
        
        # Context for execution - accumulate all reasoning
        accumulated_reasoning = []
        accumulated_reasoning.append(f"Problem: {problem}")
        if self.plan:
            accumulated_reasoning.append(f"\nPlan: {self.plan}\n")
        
        # Execute tools sequentially
        for tool_name in selected_tools:
            start_time = time.time()
            tokens_used = 0
            print(f"[DEBUG] Executing tool: {tool_name}")
            
            # Track tool name
            self.tools_used.append(tool_name)
            
            if tool_name in self.reasoners:
                reasoner = self.reasoners[tool_name]
                # print(f"[DEBUG] Executing tool: {tool_name} with compute: {compute_config}")   #TODO right compute comfig is only working with reasoner tools and not with the verifiers
                
                # Track compute config for this tool
                self.compute_configs_used.append(copy.deepcopy(compute_config))
                
                if compute_config['param'] == 1:
                    # Direct execution (no compute strategy)
                    # reasoner.generate_action returns (action, reasoning)
                    action, reasoning = reasoner.generate_action(messages, self.plan)
                    
                    accumulated_reasoning.append(f"\n--- {tool_name.upper()} OUTPUT ---")
                    accumulated_reasoning.append(f"Reasoning: {reasoning}")
                    accumulated_reasoning.append(f"Action/Step: {action}")
                    
                    tokens_used = len(str(messages)) // 4 + len(reasoning) // 4 + len(action) // 4
                    
                    metadata = {
                        'tool': tool_name,
                        'compute_strategy': 'direct',
                        'compute_config': compute_config,
                        'latency_ms': round((time.time() - start_time) * 1000, 2),
                        'tokens_used': tokens_used,
                    }
                else:
                    # Compute strategy execution (best-of-n, beam search, etc.)
                    # compute_strategy.execute returns (action, metadata)
                    compute_strategy = get_compute_strategy(
                        compute_config['strategy'],
                        self.prm_model
                    )
                    
                    action, metadata = compute_strategy.execute(
                        reasoner, messages, problem, self.plan, compute_config
                    )
                    
                    accumulated_reasoning.append(f"\n--- {tool_name.upper()} OUTPUT (with {compute_config['strategy']}) ---")
                    accumulated_reasoning.append(f"Best action: {action}")
                    
                    metadata['tool'] = tool_name
                    metadata['latency_ms'] = round((time.time() - start_time) * 1000, 2)
                    tokens_per_gen = len(str(messages)) // 4 + 50
                    metadata['tokens_used'] = tokens_per_gen * compute_config['param']
                
                self.compute_metadata_history.append(metadata)
                print(f"[DEBUG] Tool {tool_name} execution complete")
                
            elif tool_name == 'numeric_verifier':
                # Verifier tools don't have compute config in same way
                self.compute_configs_used.append({'strategy': 'direct', 'param': 1})
                
                result = self.numeric_verifier.verify(
                    messages, problem, self.plan, accumulated_reasoning[-1] if accumulated_reasoning else ""
                )
                accumulated_reasoning.append(f"\n--- NUMERIC VERIFICATION ---")
                accumulated_reasoning.append(f"Verification result: {result}")
                tokens_used = len(str(messages)) // 4 + 50
                self.compute_metadata_history.append({
                    'tool': 'numeric_verifier',
                    'result': result,
                    'latency_ms': round((time.time() - start_time) * 1000, 2),
                    'tokens_used': tokens_used
                })
                
            elif tool_name == 'verifier':
                self.compute_configs_used.append({'strategy': 'direct', 'param': 1})
                
                result = self.verifier.verify(
                    messages, problem, self.plan, accumulated_reasoning[-1] if accumulated_reasoning else ""
                )
                accumulated_reasoning.append(f"\n--- GENERAL VERIFICATION ---")
                accumulated_reasoning.append(f"Verification result: {result}")
                tokens_used = len(str(messages)) // 4 + 50
                self.compute_metadata_history.append({
                    'tool': 'verifier',
                    'result': result,
                    'latency_ms': round((time.time() - start_time) * 1000, 2),
                    'tokens_used': tokens_used
                })
                
            elif tool_name == 'summarizer':
                self.compute_configs_used.append({'strategy': 'direct', 'param': 1})
                
                summary = self.summarizer.summarize('\n'.join(accumulated_reasoning))
                accumulated_reasoning.append(f"\n--- SUMMARY ---")
                accumulated_reasoning.append(f"{summary}")
                tokens_used = len('\n'.join(accumulated_reasoning)) // 4 + len(summary) // 4
                self.compute_metadata_history.append({
                    'tool': 'summarizer',
                    'summary': summary,
                    'latency_ms': round((time.time() - start_time) * 1000, 2),
                    'tokens_used': tokens_used
                })
                
            elif tool_name == 'reframe':
                self.compute_configs_used.append({'strategy': 'direct', 'param': 1})
                
                reframed = self.reframe_tool.reframe(problem, self.plan)
                accumulated_reasoning.append(f"\n--- REFRAMED PROBLEM ---")
                accumulated_reasoning.append(f"{reframed}")
                tokens_used = len(problem) // 4 + len(reframed) // 4
                self.compute_metadata_history.append({
                    'tool': 'reframe',
                    'reframed': reframed,
                    'latency_ms': round((time.time() - start_time) * 1000, 2),
                    'tokens_used': tokens_used
                })
                
            elif tool_name == 'web_search':
                self.compute_configs_used.append({'strategy': 'direct', 'param': 1})
                
                results = self.web_tool.search(problem)
                accumulated_reasoning.append(f"\n--- WEB SEARCH RESULTS ---")
                accumulated_reasoning.append(f"{results}")
                tokens_used = len(problem) // 4 + len(results) // 4
                self.compute_metadata_history.append({
                    'tool': 'web_search',
                    'results': results,
                    'latency_ms': round((time.time() - start_time) * 1000, 2),
                    'tokens_used': tokens_used
                })
        
        # ========================================
        # STEP 4: GENERATE FINAL ANSWER WITH STRUCTURED OUTPUT
        # ========================================
        full_reasoning = '\n'.join(accumulated_reasoning)
        
        # Build final prompt for answer extraction
        final_messages = [
            {'role': 'system', 'content': 'You are a mathematical problem solver. Extract the final numerical answer from the reasoning below.'},
            {'role': 'user', 'content': f"""{full_reasoning}

Based on all the reasoning above, provide the final numerical answer or simplified expression.
Only provide the answer value, nothing else."""}
        ]
        
        print(f"[DEBUG] Generating final answer with structured output")
        
        # Use generate_with_structured to get JSON output
        structured_response = self.client.generate_with_structured(
            messages=final_messages,
            schema=FinalAnswer.model_json_schema()
        )
        
        # final_answer = structured_response.completion.answer
        final_answer = get_field_from_completion(structured_response.completion, 'answer')
        
        print(f"[DEBUG] Final answer extracted via structured output: {final_answer}")
        
        return full_reasoning, final_answer

    def _get_planning_only_instruction(self) -> str:
        """Get instruction for generating plan only (separate from solving)."""
        return """Review the problem carefully and create a high-level plan for solving it.

Output your plan in this format:
<plan>YOUR SOLUTION APPROACH</plan>

Be specific about the steps you will take, but do NOT solve the problem yet."""

    def _get_solve_with_plan_instruction(self) -> str:
        """Get instruction for solving with an existing plan."""
        return """Now solve the problem step by step following your plan. Show your work clearly.

When you have the final answer, output it in this format:
<answer>YOUR FINAL ANSWER</answer>

Provide only the numeric value or simplified expression in the answer tags."""

    def _get_solve_instruction(self) -> str:
        """Get instruction for direct solving (no planning)."""
        return """Solve this problem step by step. Show your work clearly.

When you have the final answer, output it in this format:
<answer>YOUR FINAL ANSWER</answer>

Provide only the numeric value or simplified expression in the answer tags."""
    
    def _extract_plan(self, text: str) -> tuple:
        """Extract plan from response."""
        plan_pattern = r'<plan>(.*?)</plan>'
        plan_match = re.search(plan_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if plan_match:
            plan = plan_match.group(1).strip()
            remaining = text[plan_match.end():].strip()
            return plan, remaining, True
        
        return None, text, False
    
    def _extract_answer(self, text: str) -> str:
        """Extract final answer from response (fallback for non-selector path)."""
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback: try to find answer after common patterns
        patterns = [
            r'final answer is[:\s]+(.+?)(?:\n|$)',
            r'answer is[:\s]+(.+?)(?:\n|$)',
            r'therefore[,:\s]+(.+?)(?:\n|$)',
            r'thus[,:\s]+(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Last resort: return last line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else text.strip()
    
    def get_stats(self) -> Dict:
        """Get statistics about agent performance."""
        stats = {
            'step_count': self.step_count,
            'total_plans': sum(self.planning_decisions),
            'planning_frequency': sum(self.planning_decisions) / len(self.planning_decisions) if self.planning_decisions else 0,
            'mode': self.mode,
            'use_planner': self.use_planner,
            'use_tool_selector': self.use_tool_selector,
            'use_compute_selector': self.use_compute_selector,
            'fixed_tool': self.fixed_tool,
            'fixed_compute': self.fixed_compute,
        }
        
        if self.use_tool_selector and self.tool_selector:
            stats['tool_selection'] = self.tool_selector.get_tool_distribution()
        
        if self.use_compute_selector and self.compute_selector:
            stats['compute_selection'] = self.compute_selector.get_compute_distribution()
        
        if self.prm_model:
            stats['prm_scoring'] = self.prm_model.get_scoring_stats()
        
        return stats
    
    def reset(self):
        """Reset agent state for new problem."""
        self.plan = None
        self.step_count = 0
        self.planning_decisions = []
        self.plan_history = []
        self.compute_metadata_history = []
        self.tools_used = []
        self.compute_configs_used = []
        
        if self.tool_selector:
            self.tool_selector.reset()
        if self.compute_selector:
            self.compute_selector.reset()
        if self.prm_model:
            self.prm_model.reset()


def normalize_answer(answer: str) -> str:
    """Normalize answer text for comparison."""
    if answer is None:
        return ""
    
    # Ensure we're working with a string
    answer = str(answer).strip()
    
    # Remove LaTeX boxed{} wrapper
    answer = re.sub(r'\\boxed\{(.*?)\}', r'\1', answer)
    
    # Remove inline math $...$
    answer = re.sub(r'\$(.+?)\$', r'\1', answer)
    
    # Collapse multiple spaces
    answer = re.sub(r'\s+', ' ', answer)
    
    # Simple normalization for \frac (optional; keeps structure recognizable)
    answer = answer.replace('\\frac', 'frac')
    
    return answer.lower().strip()


def _extract_number(text: str):
    """
    Extract the first numeric value from text.
    Supports integers and decimals like -12, 3.14.
    (If no number found, returns None.)
    """
    match = re.search(r'-?\d+(\.\d+)?', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def check_answer_equivalence(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer (text or numeric)."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    # Direct normalized match
    if pred_norm == gold_norm:
        return True
    
    # Numeric comparison (if both contain a number)
    pred_num = _extract_number(pred_norm)
    gold_num = _extract_number(gold_norm)
    
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-6
    
    # If no exact text match and numeric check fails
    return False


import csv

def evaluate_math_dataset(
    agent_factory,
    output_dir: str,
    num_workers: int = 1,
    max_problems: Optional[int] = None,
    split: str = 'train',
    problem_types: Optional[List[str]] = None,
    difficulty_levels: Optional[List[str]] = None,
    agent_config: Optional[Dict] = None
):
    """
    Evaluate agent on MATH dataset from Hugging Face.
    
    Args:
        agent_factory: Function that creates agent instances
        output_dir: Directory to save results
        num_workers: Number of parallel workers (for future parallel support)
        max_problems: Maximum number of problems to evaluate
        split: Dataset split ('train', 'test')
        problem_types: List of problem types to filter (e.g., ['Algebra', 'Geometry'])
        difficulty_levels: List of difficulty levels to filter (e.g., ['Level 1', 'Level 2'])
        agent_config: Configuration dict for agent
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading MATH dataset from Hugging Face (qwedsacf/competition_math)")
    
    # Load dataset
    ds = load_dataset("qwedsacf/competition_math")
    
    print(f"[INFO] Dataset splits available: {list(ds.keys())}")
    # Get the appropriate split
    dataset = ds[split]
    
    print(f"[INFO] Loaded {len(dataset)} problems from {split} split")
    
    # Filter by problem type if specified
    if problem_types:
        dataset = dataset.filter(lambda x: x['type'] in problem_types)
        print(f"[INFO] Filtered to {len(dataset)} problems of types: {problem_types}")
    
    # Filter by difficulty level if specified
    if difficulty_levels:
        dataset = dataset.filter(lambda x: x['level'] in difficulty_levels)
        print(f"[INFO] Filtered to {len(dataset)} problems of levels: {difficulty_levels}")
    
    # Limit number of problems if specified
    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
        print(f"[INFO] Limited to {len(dataset)} problems")
    
    # Create agent
    agent = agent_factory()
    
    results = []
    correct = 0
    total = 0
    
    # Track by problem type and level
    stats_by_type = {}
    stats_by_level = {}
    
    # Initialize CSV file
    csv_path = output_dir / f"mode-{agent_config['mode']}_planner-{agent_config['use_planner']}_toolsel-{agent_config['use_tool_selector']}_computesel-{agent_config['use_compute_selector']}_fixedtool-{agent_config['fixed_tool']}_fixedcompute-{agent_config['fixed_compute']}_pf-{agent_config['planning_frequency']}_results_live.csv"
    csv_headers = ['problem_id', 'problem_type', 'level', 'gold_answer', 'predicted_answer', 
                   'is_correct', 'cumulative_accuracy', 'time_seconds']
    
    # Create CSV file and write header
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
    
    print(f"[INFO] CSV results will be saved to: {csv_path}")
    
    start_time = time.time()
    
    for idx in range(len(dataset)):
        problem_data = dataset[idx]
        
        problem = problem_data['problem']
        gold_answer = problem_data['solution']  # Full solution with answer
        problem_type = problem_data['type']
        level = problem_data['level']
        
        # Try to extract just the final answer from solution
        # MATH dataset typically has answer in \boxed{} format
        gold_answer_extracted = normalize_answer(gold_answer)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', gold_answer)
        if boxed_match:
            gold_answer_extracted = normalize_answer(boxed_match.group(1))
        
        print(f"\n{'='*60}")
        print(f"[INFO] Problem {idx + 1}/{len(dataset)}")
        print(f"Type: {problem_type}, Level: {level}")
        print(f"Problem: {problem[:150]}...")
        
        # Reset agent for new problem
        agent.reset()
        
        # Solve problem
        problem_start = time.time()
        response = agent.solve(problem)
        problem_time = time.time() - problem_start
        
        # Check answer
        is_correct = check_answer_equivalence(response.answer, gold_answer_extracted)
        
        if is_correct:
            correct += 1
        total += 1
        
        # Update stats by type
        if problem_type not in stats_by_type:
            stats_by_type[problem_type] = {'correct': 0, 'total': 0}
        stats_by_type[problem_type]['total'] += 1
        if is_correct:
            stats_by_type[problem_type]['correct'] += 1
        
        # Update stats by level
        if level not in stats_by_level:
            stats_by_level[level] = {'correct': 0, 'total': 0}
        stats_by_level[level]['total'] += 1
        if is_correct:
            stats_by_level[level]['correct'] += 1
        
        accuracy = correct / total * 100

        print(f"Reasoning: {response.reasoning}")
        print(f"Plan: {response.plan}")
        print(f"Predicted: {response.answer}")
        print(f"Gold: {gold_answer_extracted}")
        print(f"Correct: {'✅' if is_correct else '❌'}")
        print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Time: {problem_time:.2f}s")
        
        # Save result to results list
        result = {
            'problem_id': idx,
            'problem': problem,
            'problem_type': problem_type,
            'level': level,
            'gold_answer': gold_answer_extracted,
            'gold_solution': gold_answer,
            'predicted_answer': response.answer,
            'reasoning': response.reasoning,
            'plan': response.plan,
            'is_correct': is_correct,
            'time_seconds': problem_time,
            'tools_used': response.metadata.get('tools_used', []),
            'compute_configs_used': response.metadata.get('compute_configs_used', []),
            'metadata': response.metadata
        }
        results.append(result)
        
        # Append to CSV file immediately
        csv_row = {
            'problem_id': idx,
            'problem_type': problem_type,
            'level': level,
            'gold_answer': gold_answer_extracted,
            'predicted_answer': response.answer,
            'is_correct': is_correct,
            'cumulative_accuracy': f"{accuracy:.2f}",
            'time_seconds': f"{problem_time:.2f}"
        }
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(csv_row)
    
    total_time = time.time() - start_time
    
    # Compute final statistics
    final_accuracy = correct / total * 100 if total > 0 else 0
    
    # Calculate accuracy by type and level
    accuracy_by_type = {
        ptype: (stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0)
        for ptype, stats in stats_by_type.items()
    }
    
    accuracy_by_level = {
        level: (stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0)
        for level, stats in stats_by_level.items()
    }
    
    agent_stats = agent.get_stats()
    agent_stats.update({
        'total_problems': total,
        'correct': correct,
        'accuracy': final_accuracy,
        'total_time_seconds': total_time,
        'avg_time_per_problem': total_time / total if total > 0 else 0,
        'stats_by_type': stats_by_type,
        'stats_by_level': stats_by_level,
        'accuracy_by_type': accuracy_by_type,
        'accuracy_by_level': accuracy_by_level
    })
    
    # Save final results
    final_results = {
        'statistics': agent_stats,
        'results': results
    }
    
    final_path = output_dir / f"mode-{agent_config['mode']}_planner-{agent_config['use_planner']}_toolsel-{agent_config['use_tool_selector']}_computesel-{agent_config['use_compute_selector']}_fixedtool-{agent_config['fixed_tool']}_fixedcompute-{agent_config['fixed_compute']}_pf-{agent_config['planning_frequency']}_results_final.json"

    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total problems: {total}")
    print(f"Correct: {correct}")
    print(f"Overall Accuracy: {final_accuracy:.2f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per problem: {total_time / total:.2f}s")
    print(f"\nAccuracy by Problem Type:")
    for ptype, acc in accuracy_by_type.items():
        print(f"  {ptype}: {acc:.2f}% ({stats_by_type[ptype]['correct']}/{stats_by_type[ptype]['total']})")
    print(f"\nAccuracy by Difficulty Level:")
    for level, acc in accuracy_by_level.items():
        print(f"  {level}: {acc:.2f}% ({stats_by_level[level]['correct']}/{stats_by_level[level]['total']})")
    print(f"\nResults saved to: {final_path}")
    print(f"Live CSV results saved to: {csv_path}")
    
    return final_results

# Example usage with CLI-style config
def create_agent_from_config(config: Dict):
    """Create agent from configuration dict."""
    
    # Client factory
    def client_factory():
        from BALROG.balrog.client import create_llm_client
        return create_llm_client(DictConfig(config['client']))
    
    agent = MathAgent(
        client_factory=client_factory(),
        mode=config['agent']['mode'],
        planning_frequency=config['agent'].get('planning_frequency'),
        use_planner=config['agent']['use_planner'],
        use_tool_selector=config['agent']['use_tool_selector'],
        use_compute_selector=config['agent']['use_compute_selector'],
        fixed_tool=config['agent'].get('fixed_tool'),
        fixed_compute=config['agent'].get('fixed_compute'),
        remember_cot=config['agent']['remember_cot'],
        max_text_history=config['agent']['max_text_history'],
        max_image_history=config['agent']['max_image_history'],
        domain=config['agent'].get('domain', 'math')  # Default to 'math'
    )
    
    return agent


def main():
    """Main evaluation function matching CLI arguments."""
    
    # Configuration matching your CLI args
    config = {
        'agent': {
            'mode': 'dynamic',
            'use_planner': True,
            'use_tool_selector': True,
            'use_compute_selector': True,
            'remember_cot': True,
            'max_text_history': 16,
            'max_image_history': 0,
            'fixed_tool': None,  # or 'cot' / 'reactive_actor'
            'fixed_compute': None,  # or {'strategy': 'best_of_n', 'param': 3}
            'planning_frequency': None,  # Only for fixed mode
            'domain': 'math'  # 'math' for MATH dataset, 'crafter' for Crafter game
        },
        'eval': {
            'num_workers': 16,
            'output_dir': 'multi_tool_results',
            'max_problems': 250,  # Set to small number for testing
            'split': 'train',
            'problem_types': None,
            'difficulty_levels': None
        },
        'client': {
            'client_name': 'ollama',
            'base_url': 'http://localhost:11434/v1',
            'model_id': 'gemma3n:e4b',
            'generate_kwargs': {
                'temperature': 0.7,
                'max_tokens': 4096
            },
            'timeout': 60,
            'max_retries': 5,
            'delay': 2,
            'alternate_roles': False
        }
    }
    
    # Create agent factory
    agent_factory = lambda: create_agent_from_config(config)
    
    # Run evaluation
    results = evaluate_math_dataset(
        agent_factory=agent_factory,
        output_dir=config['eval']['output_dir'],
        num_workers=config['eval']['num_workers'],
        max_problems=config['eval'].get('max_problems'),
        split=config['eval']['split'],
        problem_types=config['eval'].get('problem_types'),
        difficulty_levels=config['eval'].get('difficulty_levels'),
        agent_config=config['agent']
    )
    
    return results


if __name__ == '__main__':
    main()