import json


_VALID_STRATEGIES = ("best_of_n", "beam_search", "lookahead")
_DEFAULT = {"strategy": "best_of_n", "param": 1}
_CLAMPS = {
    "best_of_n": (1, 64),
    "beam_search": (2, 16),
    "lookahead": (1, 4),
}


class ComputeSelector:
    
    def __init__(self, client):
        self.client = client
        self.compute_selection_history = []
    
    def _build_prompt(self, obs, plan, tool_name, compute_selector_prompt, history_messages):
        sys = {
            "role": "system",
            "content": (
                "You are a selector that returns STRICT JSON for test-time compute.\n"
                "Respond with a single JSON object ONLY, no prose, no markdown.\n"
                "Schema: {\"strategy\": \"best_of_n|beam_search|lookahead\", \"param\": int}."
            )
        }
        
        user_content = compute_selector_prompt.format(
            tool=tool_name,
            plan=plan if plan else "(none)",
            obs=str(obs),
        )
        
        user = {"role": "user", "content": user_content}
        msgs = [sys] + (history_messages[-3:] if history_messages else []) + [user]
        return msgs
    
    def _parse_and_validate(self, text):
        obj = json.loads(text.strip())
        strategy = str(obj.get("strategy", _DEFAULT["strategy"])).strip()
        param_raw = obj.get("param", _DEFAULT["param"])
        
        if isinstance(param_raw, bool):
            param_raw = int(param_raw)
        
        if isinstance(param_raw, (int, float)):
            param = int(param_raw)
        elif isinstance(param_raw, str):
            param = int(param_raw) if param_raw.isdigit() else _DEFAULT["param"]
        else:
            param = _DEFAULT["param"]
        
        if strategy not in _VALID_STRATEGIES:
            strategy = _DEFAULT["strategy"]
        
        lo, hi = _CLAMPS[strategy]
        if not (lo <= param <= hi):
            param = max(lo, min(hi, max(1, param)))
        
        return {"strategy": strategy, "param": param}
    
    def select_compute_strategy(self, obs, plan, tool_name, compute_selector_prompt, history_messages, schema):
        messages = self._build_prompt(obs, plan, tool_name, compute_selector_prompt, history_messages or [])
        resp = self.client.generate_with_structured(messages, schema)
        text = resp.completion if hasattr(resp, "completion") else str(resp)
        selection = self._parse_and_validate(text)
        
        self.compute_selection_history.append({
            "tool": tool_name,
            "plan_present": bool(plan),
            "selection": dict(selection),
        })
        
        return selection
    
    def get_compute_distribution(self):
        counts = {k: 0 for k in _VALID_STRATEGIES}
        for ev in self.compute_selection_history:
            s = ev.get("selection", {}).get("strategy")
            if s in counts:
                counts[s] += 1
        
        total = sum(counts.values())
        dist = {k: (counts[k] / total if total else 0.0) for k in counts}
        
        return {
            "strategy_counts": counts,
            "strategy_distribution": dist,
            "total_selections": total
        }
    
    def reset(self):
        self.compute_selection_history = []