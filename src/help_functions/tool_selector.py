"""Tool selector (A_T): chooses an ordered subset of reasoning tools for a problem."""

import json


_VALID_TOOLS = (
    "self_reflection",
    "cot",
    "heuristic_script",
    "numeric_verifier",
    "verifier",
    "summarizer",
    "reframe",
)

_DEFAULT = {"tools": ["cot"]}


class ToolSelector:
    """Selects an ordered list of reasoning tools with the LLM (A_T)."""
    
    def __init__(self, client):
        self.client = client
        self.tool_selection_history = []
    
    def _build_prompt(self, obs, plan, history_messages, tool_selector_prompt, tool_selector_system_prompt):
        sys = {"role": "system", "content": tool_selector_system_prompt}
        
        formatted_prompt = tool_selector_prompt.format(
            plan=plan if plan else "(none)",
            obs=str(obs),
        )
        
        user = {"role": "user", "content": formatted_prompt}
        msgs = [sys] + (history_messages[-3:] if history_messages else []) + [user]
        return msgs


    def _parse_and_validate(self, text):
        obj = json.loads(text.strip())
        print("ToolSelector raw output:", obj)
        
        # Handle both list and dict responses
        if isinstance(obj, list):
            tools = obj
        elif isinstance(obj, dict):
            tools = obj.get("tools", [])
            # Fallback to "tool" key if "tools" doesn't exist
            if not tools and "tool" in obj:
                tools = [obj["tool"]]
        else:
            tools = []
        
        # Ensure tools is a list
        if isinstance(tools, str):
            tools = [tools]
        
        # Validate against allowed tools
        validated = [t for t in tools if t in _VALID_TOOLS]
        
        # Default fallback
        if not validated:
            validated = list(_DEFAULT["tools"])
        
        return {"tools": validated}
        
    def select_tool(self, obs, plan, tool_selector_prompt, history_messages, schema, tool_selector_system_prompt):
        """Return {'tools': [...]} for the given problem and plan."""
        messages = self._build_prompt(obs, plan, history_messages or [], tool_selector_prompt, tool_selector_system_prompt)
        
        resp = self.client.generate_with_structured(messages, schema=schema)

        print("🧩🧩🧩🧩🧩 ToolSelector raw response:", resp)

        text = resp.completion if hasattr(resp, "completion") else str(resp)
        selection = self._parse_and_validate(text)
        
        self.tool_selection_history.append({
            "plan_present": bool(plan),
            "selection": dict(selection),
        })
        
        return selection
    
    def get_tool_distribution(self):
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
            "total_selections": len(self.tool_selection_history),
        }
    
    def reset(self):
        self.tool_selection_history = []