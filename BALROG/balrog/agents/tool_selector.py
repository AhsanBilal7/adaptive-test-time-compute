import json
from typing import Dict, Any, List

_VALID_TOOLS = (
    "reactive_actor",
    "cot",
    "heuristic_script",
    "numeric_verifier",
    "verifier",
    "summarizer",
    "reframe",
    "web_search",
)

_DEFAULT = {"tools": ["reactive_actor"]}
from balrog.schemas import ToolsPayload

class ToolSelector:
    
    def __init__(self, client):
        self.client = client
        self.tool_selection_history = []  # list[Dict[str, Any]]
    
    def _build_prompt(self, obs, plan, history_messages) -> list:
        """
        Returns a messages list for JSON-only multi-tool selection.
        """
        sys = {
            "role": "system",
            "content": (
                "You are a tool selector that returns STRICT JSON for multi-tool chains.\n"
                "Respond with a single JSON object ONLY, no prose, no markdown.\n"
                "Schema: {\"tools\": [\"tool1\", \"tool2\", ...]}.\n"
                "Available tools: reactive_actor, cot, heuristic_script, numeric_verifier, verifier, summarizer, reframe, web_search."
            )
        }
        user = {
            "role": "user",
            "content": (
                "Select one or more tools to execute SEQUENTIALLY for this step.\n"
                f"Plan: {plan if plan else '(none)'}\n"
                f"Observation: {str(obs)}\n\n"
                "Available tools:\n"
                "reactive_actor  – Fast, direct action selection\n"
                "cot             – Step-by-step reasoning\n"
                "heuristic_script– Rule-based domain policy\n"
                "numeric_verifier– PRM-based numeric checks\n"
                "verifier        – General PRM correctness check\n"
                "summarizer      – Compress long reasoning chains\n"
                "reframe         – Reformulate question/plan\n"
                "web_search      – Retrieve external information\n\n"
                "DECISION RULES (apply in order; collect matches; cap to 3 tools):\n"
                "1) FACT-GAP: If Plan/Observation indicates missing facts, recency, URLs, or uncertainty about world knowledge → include web_search first.\n"
                "2) AMBIGUITY/POOR SPEC: If question is unclear, contradictory, or underspecified (markers: 'unclear', '?', 'maybe', multiple interpretations) → include reframe before any reasoning.\n"
                "3) MULTI-STEP/DERIVATION: If solving requires multi-step logic, decomposition, proofs, or algorithm design → include cot.\n"
                "4) DOMAIN RULES: If a known rule-based policy applies (e.g., fixed heuristics/workflows) → include heuristic_script (before cot if it can prune the space).\n"
                "5) NUMERIC RISK: If arithmetic, units, thresholds, probabilities, or quantitative constraints appear → include numeric_verifier after the generator (cot/heuristic_script/reactive_actor).\n"
                "6) GENERAL CORRECTNESS: If the output must satisfy constraints/specs or prior errors are likely → include verifier after generation (and after numeric_verifier if both are used).\n"
                "7) LONG CONTEXT: If Plan+Observation or expected chain > 600 tokens or multiple sub-answers → include summarizer last to compress.\n"
                "8) TRIVIALITY: ONLY if none of rules 1–7 fired and the task is single-step with explicit action → choose [reactive_actor] alone.\n\n"
                "ORDERING RULES:\n"
                "- If reframe selected, it must be first.\n"
                "- If web_search selected, it comes right after reframe (or first if no reframe).\n"
                "- heuristic_script precedes cot if both selected.\n"
                "- numeric_verifier precedes verifier; both follow the generator (reactive_actor/heuristic_script/cot).\n"
                "- summarizer is always last.\n\n"
                "HARD CONSTRAINTS:\n"
                "- Do NOT output only [\"reactive_actor\"] if any of rules 1–7 matched.\n"
                "- If any numeric terms are present, numeric_verifier is mandatory.\n"
                "- If external facts are referenced or freshness matters, web_search is mandatory.\n"
                "- Max sequence length is 3 tools; prefer the most impactful ones per rules above.\n\n"
                "TEMPLATES (examples, not prescriptive):\n"
                "- Simple, unambiguous action → [\"reactive_actor\"]\n"
                "- Needs facts then reasoning with checks → [\"web_search\", \"cot\", \"verifier\"]\n"
                "- Rule-based pruning then numeric check → [\"heuristic_script\", \"numeric_verifier\"]\n"
                "- Ambiguous prompt then plan+verify → [\"reframe\", \"cot\", \"verifier\"]\n"
                "- Long chain to compress → [\"cot\", \"summarizer\"]\n\n"
                "Return JSON ONLY. Example: {\"tools\": [\"web_search\", \"cot\", \"numeric_verifier\"]}"
            )
        }

        msgs = [sys] + (history_messages[-3:] if history_messages else []) + [user]
        # msgs = [sys] + [user]
        return msgs
    
    def _parse_and_validate(self, text: str) -> Dict[str, Any]:
        """
        Parse and validate multi-tool selection with backward compatibility.
        """
        # print("Raw tool selection text:", text)  # Debugging line
        try:
            obj = json.loads(text.strip())
            tools = obj.get("tools", [])
            
            # Backward compatibility: if "tool" (singular) is provided
            if not tools and "tool" in obj:
                tools = [obj["tool"]]
            
            # Ensure it's a list
            if isinstance(tools, str):
                tools = [tools]
            
        except Exception as e:
            print("Error parsing tool selection JSON in the parse and validation function:", e)  # Debugging line
            return dict(_DEFAULT)
        
        # Filter to valid tools only
        validated = [t for t in tools if t in _VALID_TOOLS]
        
        # Fallback if empty
        if not validated:
            validated = list(_DEFAULT["tools"])
        
        return {"tools": validated}
    
    def select_tool(self, obs, plan, history_messages):
        """
        Ask the master LLM to select which tools to use (supports multiple).
        
        Args:
            obs: Current observation
            plan: Current plan (if any)
            history_messages: Conversation history
            
        Returns:
            dict: {"tools": [str, ...]} - List of tool names to execute sequentially
        """
        messages = self._build_prompt(obs, plan, history_messages or [])

        # print("The message is:", messages)  # Debugging line
        try:
            # print("Sending messages to LLM for tool selection...", messages)
            # resp = self.client.generate(messages)
            resp = self.client.generate_with_structured(messages, schema=ToolsPayload.model_json_schema())
            text = resp.completion if hasattr(resp, "completion") else str(resp)
            print("Tool selection response text:", text)  # Debugging line
        except Exception as e:
            selection = dict(_DEFAULT)
            # print(f"Exception {e} occurred during tool selection; using default:", selection)
        else:
            selection = self._parse_and_validate(text)
            # print("Tool selection response selection:", selection)  # Debugging line
        
        # Record immutable snapshot
        self.tool_selection_history.append({
            "plan_present": bool(plan),
            "selection": dict(selection),
        })
        
        return selection
    
    def get_tool_distribution(self):
        """
        Get distribution of tool selections (flattened across all selections).
        
        Returns:
            dict: Tool selection statistics
        """
        counts = {k: 0 for k in _VALID_TOOLS}
        
        for ev in self.tool_selection_history:
            tools = ev.get("selection", {}).get("tools", [])
            for t in tools:
                if t in counts:
                    counts[t] += 1
        
        total = sum(counts.values())
        dist = {k: (counts[k] / total if total else 0.0) for k in counts}
        
        return {
            "tool_counts": counts,
            "tool_distribution": dist,
            "total_tool_invocations": total,
            "total_selections": len(self.tool_selection_history)
        }
    
    def reset(self):
        """
        Reset tool selection history.
        """
        self.tool_selection_history = []