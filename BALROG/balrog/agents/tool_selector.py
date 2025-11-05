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
                "Select one or more tools to execute sequentially for this step.\n"
                f"Plan: {plan if plan else '(none)'}\n"
                f"Observation: {str(obs)[:800]}\n\n"
                "Tool descriptions:\n"
                "1. reactive_actor - Fast, direct action selection\n"
                "2. cot - Chain-of-thought step-by-step reasoning\n"
                "3. heuristic_script - Rule-based domain-specific policy\n"
                "4. numeric_verifier - PRM-based numeric verification\n"
                "5. verifier - General PRM-based correctness verification\n"
                "6. summarizer - Compress long reasoning chains\n"
                "7. reframe - Reformulate question/plan for clarity\n"
                "8. web_search - Retrieve external information\n\n"
                "Guidelines:\n"
                "- Single tool for simple tasks: [\"reactive_actor\"]\n"
                "- Reasoning + verification: [\"cot\", \"numeric_verifier\"]\n"
                "- Complex multi-step: [\"cot\", \"verifier\", \"summarizer\"]\n"
                "- Meta-reasoning: [\"reframe\", \"cot\"]\n\n"
                "Return JSON ONLY. Example: {\"tools\": [\"cot\", \"numeric_verifier\"]}"
            )
        }
        msgs = [sys] + (history_messages[-3:] if history_messages else []) + [user]
        return msgs
    
    def _parse_and_validate(self, text: str) -> Dict[str, Any]:
        """
        Parse and validate multi-tool selection with backward compatibility.
        """
        try:
            obj = json.loads(text.strip())
            tools = obj.get("tools", [])
            
            # Backward compatibility: if "tool" (singular) is provided
            if not tools and "tool" in obj:
                tools = [obj["tool"]]
            
            # Ensure it's a list
            if isinstance(tools, str):
                tools = [tools]
            
        except Exception:
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
        
        try:
            resp = self.client.generate(messages)
            text = resp.completion if hasattr(resp, "completion") else str(resp)
        except Exception:
            selection = dict(_DEFAULT)
        else:
            selection = self._parse_and_validate(text)
        
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