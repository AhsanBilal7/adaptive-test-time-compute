import re
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Callable

from src.help_functions.schemas import FinalAnswer, ToolsPayload, DecisionPayload
from src.help_functions.tool_selector import ToolSelector
from src.help_functions.compute_selector import ComputeSelector
from src.help_functions.prm_model import PRMModel
from src.help_functions.self_reflection import SelfReflectionReasoner
from src.help_functions.cot_reasoner import CoTReasoner
from src.help_functions.compute_strategies import get_compute_strategy
from src.help_functions.tools import NumericVerifier, VerifierTool, SummarizerTool, ReframeTool, WebSearchTool


def get_field_from_completion(completion_text, field, default=None):
    data = json.loads(completion_text)
    current = data
    for part in field.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _get(obj, key, default=None):
    if obj is None:
        return default
    val = getattr(obj, key, None)
    if val is not None:
        return val
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return default


def _provided(x):
    return x is not None and not (isinstance(x, str) and x.strip() == "")


@dataclass
class UniversalResponse:
    reasoning: Optional[str]  # Concatenated reasoning for backward compatibility
    reasoning_steps: List[str]  # Individual reasoning steps for PRM
    answer: str
    answer_unstructured: str
    plan: Optional[str]
    metadata: Dict[str, Any]
    compute_strategy_outputs: List[Dict[str, Any]]  # Outputs from compute strategies
    reasoner_outputs: List[Dict[str, Any]]  # Outputs from reasoners


class UniversalPromptBuilder:
    def __init__(self, max_text_history=16, remember_cot=True):
        self.max_text_history = max_text_history
        self.remember_cot = remember_cot
        self.problem = None
        self.history = []
    
    def set_problem(self, problem):
        self.problem = problem
        self.history = []
    
    def update_step(self, reasoning, answer):
        if self.remember_cot and reasoning:
            self.history.append({
                "role": "assistant",
                "content": f"Reasoning: {reasoning}\nAnswer: {answer}"
            })
            if len(self.history) > self.max_text_history:
                self.history = self.history[-self.max_text_history:]
    
    def get_prompt(self, plan, system_prompt):
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        
        problem_msg = f"Problem: {self.problem}"
        if plan:
            problem_msg += f"\n\nCurrent plan: {plan}"
        
        messages.append({"role": "user", "content": problem_msg})
        messages.extend(self.history)
        
        return messages


class UniversalAgent:
    def __init__(
        self,
        client_factory,
        use_planner=False,
        use_tool_selector=False,
        use_compute_selector=False,
        fixed_tool=None,
        fixed_compute=None,
        remember_cot=True,
        max_text_history=16,
    ):
        self.client = client_factory() if callable(client_factory) else client_factory
        self.prompt_builder = UniversalPromptBuilder(max_text_history, remember_cot)
        
        self.use_planner = use_planner
        self.fixed_tool = fixed_tool
        
        if fixed_tool is not None:
            valid_tools = ["self_reflection", "cot", "heuristic_script"]
            if fixed_tool not in valid_tools:
                raise ValueError(f"fixed_tool must be one of {valid_tools}, got {fixed_tool}")
            self.use_tool_selector = False
        else:
            self.use_tool_selector = use_tool_selector
        
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
        self.compute_metadata_history = []
        self.tools_used = []
        self.compute_configs_used = []
        
        # New tracking for outputs
        self.compute_strategy_outputs = []
        self.reasoner_outputs = []
        
        self.tool_selector = None
        self.compute_selector = None
        self.prm_model = None
        self.reasoners = {}
        
        if self.use_tool_selector or self.use_compute_selector or fixed_tool or fixed_compute:
            self._initialize_selectors_and_reasoners()
    
    def _initialize_selectors_and_reasoners(self):
        if self.use_tool_selector:
            self.tool_selector = ToolSelector(self.client)
        
        if self.use_compute_selector:
            self.compute_selector = ComputeSelector(self.client)
        
        self.prm_model = PRMModel(self.client)
        
        self_reflection_reasoner = SelfReflectionReasoner(self.client)
        cot_reasoner = CoTReasoner(self.client)
        
        self.numeric_verifier = NumericVerifier(self.prm_model)
        self.verifier = VerifierTool(self.prm_model)
        self.summarizer = SummarizerTool(self.client)
        self.reframe_tool = ReframeTool(self.client)
        self.web_tool = WebSearchTool(self.client)
        
        self.reasoners = {
            "self_reflection": self_reflection_reasoner,
            "cot": cot_reasoner,
        }
        
    def solve(
        self,
        problem: str,
        prompts: Dict[str, str],
    ) -> UniversalResponse:
        self.prompt_builder.set_problem(problem)
        
        # Reset tracking
        self.compute_strategy_outputs = []
        self.reasoner_outputs = []
        
        if self.use_planner and prompts.get("planning_prompt_template"):
            self.plan = self._make_plan(problem, prompts["planning_prompt_template"])
        else:
            self.plan = None
        
        messages = self.prompt_builder.get_prompt(self.plan, prompts["system_prompt"])
        
        if not self.use_tool_selector and not self.fixed_tool:
            reasoning, reasoning_steps, answer_structured, answer_unstructured = self._execute_direct(
                problem, prompts
            )
        else:
            if self.use_tool_selector:
                reasoning, reasoning_steps, answer_structured, answer_unstructured = self._execute_with_tools(
                    problem, messages, prompts
                )
            else:
                reasoning, reasoning_steps, answer_structured, answer_unstructured = self._execute_simple(
                    problem, messages, prompts
                )
        
        metadata = {
            "plan": self.plan,
            "use_planner": self.use_planner,
            "use_tool_selector": self.use_tool_selector,
            "use_compute_selector": self.use_compute_selector,
            "fixed_tool": self.fixed_tool,
            "fixed_compute": self.fixed_compute,
            "tools_used": self.tools_used,
            "compute_configs_used": self.compute_configs_used,
            "compute_metadata": self.compute_metadata_history,
        }
        
        return UniversalResponse(
            reasoning=reasoning,
            reasoning_steps=reasoning_steps,
            answer=answer_structured,
            answer_unstructured=answer_unstructured,
            plan=self.plan,
            metadata=metadata,
            compute_strategy_outputs=self.compute_strategy_outputs,
            reasoner_outputs=self.reasoner_outputs
        )
    
    def _make_plan(self, problem, planning_prompt_template):
        plan_prompt = planning_prompt_template.format(problem=problem)
        plan_messages = [{"role": "user", "content": plan_prompt}]
        response = self.client.generate(plan_messages)
        completion = response.completion if hasattr(response, "completion") else str(response)
        plan, _, _ = self._extract_plan(completion)
        return plan
    
    def _execute_direct(self, problem, prompts):
        direct_messages = [
            {"role": "system", "content": prompts["direct_solve_system_prompt"]},
            {"role": "user", "content": prompts["direct_solve_prompt"].format(problem=problem)}
        ]
        
        structured_response = self.client.generate_with_structured(
            messages=direct_messages,
            schema=FinalAnswer.model_json_schema()
        )
        
        final_answer_structured = get_field_from_completion(structured_response.completion, "answer")
        
        unstructured_messages = [
            {"role": "system", "content": prompts["unstructured_final_answer_system_prompt"]},
            {
                "role": "user",
                "content": prompts["unstructured_final_answer_user_prompt"].format(
                    problem=problem,
                    plan="No plan was created.",
                    full_reasoning="Direct solution approach used."
                )
            }
        ]
        
        unstructured_response = self.client.generate(unstructured_messages)
        final_answer_unstructured = unstructured_response.completion if hasattr(unstructured_response, "completion") else str(unstructured_response)
        
        metadata = {
            "tool": "direct",
            "compute_strategy": "direct",
            "compute_config": {"strategy": "none", "param": 0},
        }
        
        self.tools_used = ["direct"]
        self.compute_configs_used.append({"strategy": "none", "param": 0})
        self.compute_metadata_history.append(metadata)
        
        # For direct mode, create a single reasoning step
        reasoning_text = "Direct solution approach used."
        reasoning_steps = [reasoning_text]
        
        # Track outputs
        self.compute_strategy_outputs.append({
            "strategy": "none",
            "output": "Direct solve without compute strategy"
        })
        self.reasoner_outputs.append({
            "reasoner": "direct",
            "output": final_answer_structured
        })
        
        return "", reasoning_steps, final_answer_structured, final_answer_unstructured

    def _execute_simple(self, problem, messages, prompts):
        tool_name = self.fixed_tool if self.fixed_tool else "cot"
        self.tools_used = [tool_name]
        
        reasoner = self.reasoners.get(tool_name)
        
        if tool_name == "self_reflection":
            instruction_prompt = prompts["self_reflection_instruction_prompt"]
        elif tool_name == "cot":
            instruction_prompt = prompts["cot_instruction_prompt"]
        else:
            instruction_prompt = prompts["cot_instruction_prompt"]
        
        if self.use_compute_selector:
            compute_config = self.compute_selector.select_compute_strategy(
                problem,
                self.plan,
                tool_name,
                prompts["compute_selector_prompt"],
                messages,
                DecisionPayload.model_json_schema(),
                prompts["compute_selector_system_prompt"]
            )
        elif self.fixed_compute:
            compute_config = self.fixed_compute
        else:
            compute_config = {"strategy": "best_of_n", "param": 1}
        
        self.compute_configs_used.append(compute_config)
        
        if compute_config["param"] == 1:
            action, reasoning = reasoner.generate_action(messages, instruction_prompt)
            metadata = {
                "tool": tool_name,
                "compute_strategy": "direct",
                "compute_config": compute_config,
            }
            
            # Track reasoner output
            self.reasoner_outputs.append({
                "reasoner": tool_name,
                "action": action,
                "reasoning": reasoning,
                "messages": messages,
                "instruction_prompt": instruction_prompt
            })
            
            # Track compute strategy output (direct)
            self.compute_strategy_outputs.append({
                "strategy": "direct",
                "param": 1,
                "output": {
                    "action": action,
                    "reasoning": reasoning
                }
            })
        else:
            compute_strategy = get_compute_strategy(compute_config["strategy"], self.prm_model)
            action, metadata = compute_strategy.execute(
                reasoner,
                messages,
                problem,
                self.plan,
                compute_config,
                instruction_prompt,
                prompts["prm_scoring_prompt"],
            )
            metadata["tool"] = tool_name
            reasoning = metadata.get('chosen_reasoning', '')
            
            # Track compute strategy output
            self.compute_strategy_outputs.append({
                "strategy": compute_config["strategy"],
                "param": compute_config["param"],
                "output": {
                    "action": action,
                    "metadata": metadata,
                    "all_candidates": metadata.get('candidates', []),
                    "all_reasoning_texts": metadata.get('reasoning_texts', []),
                    "scores": metadata.get('scores', []),
                    "chosen_index": metadata.get('chosen_index', 0),
                    "chosen_reasoning": reasoning
                }
            })
            
            # Track reasoner output (best from compute strategy)
            self.reasoner_outputs.append({
                "reasoner": tool_name,
                "action": action,
                "reasoning": reasoning,
                "compute_enhanced": True,
                "num_candidates": metadata.get('num_scored', 1)
            })
        
        self.compute_metadata_history.append(metadata)
        
        # Create reasoning steps list
        reasoning_steps = []
        if reasoning:
            steps = [s.strip() for s in reasoning.split('\n') if s.strip()]
            reasoning_steps.extend(steps)
        
        if not reasoning_steps:
            reasoning_steps = [reasoning] if reasoning else [""]
        
        full_reasoning = f"\n--- {tool_name.upper()} OUTPUT ---\nAction: {action}\nReasoning: {reasoning}"
        
        answer_structured, answer_unstructured = self._extract_final_answers(
            problem, full_reasoning, prompts
        )
        
        return full_reasoning, reasoning_steps, answer_structured, answer_unstructured
    
    def _execute_with_tools(self, problem, messages, prompts):
        tool_selection = self.tool_selector.select_tool(
            problem,
            self.plan,
            prompts["tool_selector_prompt"],
            messages,
            ToolsPayload.model_json_schema(),
            prompts["tool_selector_system_prompt"]
        )
        
        selected_tools = tool_selection.get("tools", ["self_reflection"])
        self.tools_used = selected_tools
        
        accumulated_reasoning = []
        reasoning_steps = []
        
        for tool_name in selected_tools:
            if tool_name in ["self_reflection", "cot"]:
                reasoner = self.reasoners.get(tool_name)
                
                if tool_name == "self_reflection":
                    instruction_prompt = prompts["self_reflection_instruction_prompt"]
                elif tool_name == "cot":
                    instruction_prompt = prompts["cot_instruction_prompt"]
                
                if self.use_compute_selector:
                    compute_config = self.compute_selector.select_compute_strategy(
                        problem,
                        self.plan,
                        tool_name,
                        prompts["compute_selector_prompt"],
                        messages,
                        DecisionPayload.model_json_schema(),
                        prompts["compute_selector_system_prompt"]
                    )
                elif self.fixed_compute:
                    compute_config = self.fixed_compute
                else:
                    compute_config = {"strategy": "best_of_n", "param": 1}
                
                self.compute_configs_used.append(compute_config)
                
                if compute_config["param"] == 1:
                    action, reasoning = reasoner.generate_action(messages, instruction_prompt)
                    accumulated_reasoning.append(f"\n--- {tool_name.upper()} OUTPUT ---")
                    accumulated_reasoning.append(f"Action: {action}")
                    if reasoning:
                        accumulated_reasoning.append(f"Reasoning: {reasoning}")
                        steps = [s.strip() for s in reasoning.split('\n') if s.strip()]
                        reasoning_steps.extend(steps)
                    
                    metadata = {
                        "tool": tool_name,
                        "compute_strategy": "direct",
                        "compute_config": compute_config,
                    }
                    
                    # Track reasoner output
                    self.reasoner_outputs.append({
                        "reasoner": tool_name,
                        "action": action,
                        "reasoning": reasoning,
                        "messages": messages,
                        "instruction_prompt": instruction_prompt
                    })
                    
                    # Track compute strategy output (direct)
                    self.compute_strategy_outputs.append({
                        "strategy": "direct",
                        "param": 1,
                        "output": {
                            "action": action,
                            "reasoning": reasoning
                        }
                    })
                else:
                    compute_strategy = get_compute_strategy(compute_config["strategy"], self.prm_model)
                    action, metadata = compute_strategy.execute(
                        reasoner,
                        messages,
                        problem,
                        self.plan,
                        compute_config,
                        instruction_prompt,
                        prompts["prm_scoring_prompt"],
                    )
                    
                    accumulated_reasoning.append(f"\n--- {tool_name.upper()} OUTPUT (with {compute_config['strategy']}) ---")
                    accumulated_reasoning.append(f"Best action: {action}")
                    chosen_reasoning = metadata.get('chosen_reasoning', '')
                    accumulated_reasoning.append(f"Best Reasoning: {chosen_reasoning}")
                    
                    if chosen_reasoning:
                        steps = [s.strip() for s in chosen_reasoning.split('\n') if s.strip()]
                        reasoning_steps.extend(steps)
                    
                    metadata["tool"] = tool_name
                    
                    # Track compute strategy output
                    self.compute_strategy_outputs.append({
                        "strategy": compute_config["strategy"],
                        "param": compute_config["param"],
                        "output": {
                            "action": action,
                            "metadata": metadata,
                            "all_candidates": metadata.get('candidates', []),
                            "all_reasoning_texts": metadata.get('reasoning_texts', []),
                            "scores": metadata.get('scores', []),
                            "chosen_index": metadata.get('chosen_index', 0),
                            "chosen_reasoning": chosen_reasoning,
                            # Include beam/lookahead specific data
                            "beam": metadata.get('beam', []),
                            "beam_reasoning": metadata.get('beam_reasoning', []),
                            "beam_scores": metadata.get('beam_scores', []),
                            "all_rollouts": metadata.get('all_rollouts', []),
                        }
                    })
                    
                    # Track reasoner output (best from compute strategy)
                    self.reasoner_outputs.append({
                        "reasoner": tool_name,
                        "action": action,
                        "reasoning": chosen_reasoning,
                        "compute_enhanced": True,
                        "num_candidates": metadata.get('num_scored', 1)
                    })
                
                self.compute_metadata_history.append(metadata)
            
            elif tool_name == "numeric_verifier":
                self.compute_configs_used.append({"strategy": "direct", "param": 1})
                result = self.numeric_verifier.verify(
                    messages,
                    problem,
                    self.plan,
                    accumulated_reasoning[-1] if accumulated_reasoning else "",
                    prompts["prm_scoring_prompt"],
                )
                accumulated_reasoning.append(f"\n--- NUMERIC VERIFICATION ---")
                accumulated_reasoning.append(f"Verification result: {result}")
                reasoning_steps.append(f"Numeric verification: {result}")
                self.compute_metadata_history.append({"tool": "numeric_verifier", "result": result})
                
                # Track tool output
                self.reasoner_outputs.append({
                    "reasoner": "numeric_verifier",
                    "output": result
                })
            
            elif tool_name == "verifier":
                self.compute_configs_used.append({"strategy": "direct", "param": 1})
                result = self.verifier.verify(
                    messages,
                    problem,
                    self.plan,
                    accumulated_reasoning[-1] if accumulated_reasoning else "",
                    prompts["prm_scoring_prompt"],
                )
                accumulated_reasoning.append(f"\n--- GENERAL VERIFICATION ---")
                accumulated_reasoning.append(f"Verification result: {result}")
                reasoning_steps.append(f"General verification: {result}")
                self.compute_metadata_history.append({"tool": "verifier", "result": result})
                
                # Track tool output
                self.reasoner_outputs.append({
                    "reasoner": "verifier",
                    "output": result
                })
            
            elif tool_name == "summarizer":
                self.compute_configs_used.append({"strategy": "direct", "param": 1})
                summary = self.summarizer.summarize("\n".join(accumulated_reasoning))
                accumulated_reasoning.append(f"\n--- SUMMARY ---")
                accumulated_reasoning.append(f"{summary}")
                reasoning_steps.append(f"Summary: {summary}")
                self.compute_metadata_history.append({"tool": "summarizer", "summary": summary})
                
                # Track tool output
                self.reasoner_outputs.append({
                    "reasoner": "summarizer",
                    "output": summary
                })
            
            elif tool_name == "reframe":
                self.compute_configs_used.append({"strategy": "direct", "param": 1})
                reframed = self.reframe_tool.reframe(problem, self.plan)
                accumulated_reasoning.append(f"\n--- REFRAMED PROBLEM ---")
                accumulated_reasoning.append(f"{reframed}")
                reasoning_steps.append(f"Reframed: {reframed}")
                self.compute_metadata_history.append({"tool": "reframe", "reframed": reframed})
                
                # Track tool output
                self.reasoner_outputs.append({
                    "reasoner": "reframe",
                    "output": reframed
                })
            
        
        if not reasoning_steps:
            reasoning_steps = ["\n".join(accumulated_reasoning)]
        
        full_reasoning = "\n".join(accumulated_reasoning)
        
        answer_structured, answer_unstructured = self._extract_final_answers(
            problem, full_reasoning, prompts
        )
        
        return full_reasoning, reasoning_steps, answer_structured, answer_unstructured
    
    def _extract_final_answers(self, problem, full_reasoning, prompts):
        final_messages_structured = [
            {"role": "system", "content": prompts["final_answer_system_prompt"]},
            {
                "role": "user",
                "content": prompts["final_answer_user_prompt"].format(
                    problem=problem,
                    plan=self.plan if self.plan else "No plan was created.",
                    full_reasoning=full_reasoning
                )
            }
        ]
        
        structured_response = self.client.generate_with_structured(
            messages=final_messages_structured,
            schema=FinalAnswer.model_json_schema()
        )

        final_answer_structured = get_field_from_completion(structured_response.completion, "answer")
        
        final_messages_unstructured = [
            {"role": "system", "content": prompts["unstructured_final_answer_system_prompt"]},
            {
                "role": "user",
                "content": prompts["unstructured_final_answer_user_prompt"].format(
                    problem=problem,
                    plan=self.plan if self.plan else "No plan was created.",
                    full_reasoning=full_reasoning
                )
            }
        ]
        
        unstructured_response = self.client.generate(final_messages_unstructured)
        final_answer_unstructured = unstructured_response.completion if hasattr(unstructured_response, "completion") else str(unstructured_response)
        
        return final_answer_structured, final_answer_unstructured
    
    def _extract_plan(self, text):
        plan_match = re.search(r"<plan>(.*?)</plan>", text, re.IGNORECASE | re.DOTALL)
        
        if plan_match:
            plan = plan_match.group(1).strip()
            remaining = text[plan_match.end():].strip()
            return plan, remaining, True
        
        return None, text, False
    
    def get_stats(self):
        stats = {
            "use_planner": self.use_planner,
            "use_tool_selector": self.use_tool_selector,
            "use_compute_selector": self.use_compute_selector,
            "fixed_tool": self.fixed_tool,
            "fixed_compute": self.fixed_compute,
        }
        
        if self.use_tool_selector and self.tool_selector:
            stats["tool_selection"] = self.tool_selector.get_tool_distribution()
        
        if self.use_compute_selector and self.compute_selector:
            stats["compute_selection"] = self.compute_selector.get_compute_distribution()
        
        if self.prm_model:
            stats["prm_scoring"] = self.prm_model.get_scoring_stats()
        
        return stats
    
    def reset(self):
        self.plan = None
        self.compute_metadata_history = []
        self.tools_used = []
        self.compute_configs_used = []
        self.compute_strategy_outputs = []
        self.reasoner_outputs = []
        
        if self.tool_selector:
            self.tool_selector.reset()
        if self.compute_selector:
            self.compute_selector.reset()
        if self.prm_model:
            self.prm_model.reset()