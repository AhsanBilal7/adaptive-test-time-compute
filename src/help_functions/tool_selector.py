import json


_VALID_TOOLS = (
    "self_reflection",
    "cot",
    "heuristic_script",
    "numeric_verifier",
    "verifier",
    "summarizer",
    "reframe",
    "web_search",
)

_DEFAULT = {"tools": ["cot"]}


class ToolSelector:
    
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
        tools = obj.get("tools", [])
        
        if not tools and "tool" in obj:
            tools = [obj["tool"]]
        
        if isinstance(tools, str):
            tools = [tools]
        
        validated = [t for t in tools if t in _VALID_TOOLS]
        
        if not validated:
            validated = list(_DEFAULT["tools"])
        
        return {"tools": validated}
    
    def select_tool(self, obs, plan, tool_selector_prompt, history_messages, schema, tool_selector_system_prompt):
        messages = self._build_prompt(obs, plan, history_messages or [], tool_selector_prompt, tool_selector_system_prompt)
        
        resp = self.client.generate_with_structured(messages, schema=schema)
        text = resp.completion if hasattr(resp, "completion") else str(resp)
        selection = self._parse_and_validate(text)
        
        self.tool_selection_history.append({
            "plan_present": bool(plan),
            "selection": dict(selection),
        })
        
        # print("======================================")
        # print("ToolSelector Prompt Messages:", messages)
        # print("ToolSelector selected tools:", selection)
        # print("======================================")
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