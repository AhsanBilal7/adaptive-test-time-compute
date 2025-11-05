import json
from typing import Dict, Any

_VALID_STRATEGIES = ("best_of_n", "beam_search", "lookahead")
# Sensible global clamps; beam width and lookahead depth are interpreted per strategy.
_DEFAULT = {"strategy": "best_of_n", "param": 1}
_CLAMPS = {
    "best_of_n": (1, 64),     # N samples
    "beam_search": (2, 16),   # beam width
    "lookahead": (1, 4),      # rollout depth k
}


class ComputeSelector:
    
    def __init__(self, client):
        self.client = client
        self.compute_selection_history = []  # list[Dict[str,Any]]
    
    def _build_prompt(self, obs, plan, tool_name, history_messages) -> list:
        """
        Returns a messages list for a JSON-only reply.
        """
        sys = {
            "role": "system",
            "content": (
                "You are a selector that returns STRICT JSON for test-time compute.\n"
                "Respond with a single JSON object ONLY, no prose, no markdown.\n"
                "Schema: {\"strategy\": \"best_of_n|beam_search|lookahead\", \"param\": int}."
            )
        }
        user = {
            "role": "user",
            "content": (
                "Decide a compute strategy for the next action.\n"
                f"Tool: {tool_name}\n"
                f"Plan: {plan if plan else '(none)'}\n"
                f"Observation: {str(obs)[:800]}\n\n"
                "Guidelines:\n"
                "- If easy/clear path: best_of_n with param in [2, 8].\n"
                "- If branching and verifier guidance needed: beam_search with beam in [2, 8].\n"
                "- If misranking risk and deeper evaluation helps: lookahead with k in [1, 3].\n"
                "Return JSON ONLY. Example: {\"strategy\":\"best_of_n\",\"param\":4}"
            )
        }
        # Keep only the last message block from history to avoid bloat; you can expand if you like.
        msgs = [sys] + (history_messages[-3:] if history_messages else []) + [user]
        return msgs
    
    def _parse_and_validate(self, text: str) -> Dict[str, Any]:
        try:
            obj = json.loads(text.strip())
            strategy = str(obj.get("strategy", _DEFAULT["strategy"])).strip()
            param = int(obj.get("param", _DEFAULT["param"]))
        except Exception:
            return dict(_DEFAULT)

        if strategy not in _VALID_STRATEGIES:
            strategy = _DEFAULT["strategy"]
        lo, hi = _CLAMPS[strategy]
        if not (lo <= param <= hi):
            # clamp into allowed window
            param = max(lo, min(hi, max(1, param)))

        return {"strategy": strategy, "param": param}
    
    def select_compute_strategy(self, obs, plan, tool_name, history_messages):
        """
        Returns {'strategy': str, 'param': int} (validated + clamped).
        
        Args:
            obs: Current observation
            plan: Current plan (if any)
            tool_name: Selected tool/reasoner
            history_messages: Conversation history
            
        Returns:
            dict: Compute strategy config with 'strategy' and 'param'
        """
        messages = self._build_prompt(obs, plan, tool_name, history_messages or [])
        try:
            resp = self.client.generate(messages)
            text = resp.completion if hasattr(resp, "completion") else str(resp)
        except Exception:
            selection = dict(_DEFAULT)
        else:
            selection = self._parse_and_validate(text)

        # record immutable snapshot
        self.compute_selection_history.append({
            "tool": tool_name,
            "plan_present": bool(plan),
            "selection": dict(selection),
        })
        return selection
    
    def get_compute_distribution(self):
        """
        Get distribution of compute strategy selections.
        
        Returns:
            dict: Compute strategy statistics
        """
        counts = {k: 0 for k in _VALID_STRATEGIES}
        for ev in self.compute_selection_history:
            s = ev.get("selection", {}).get("strategy")
            if s in counts:
                counts[s] += 1
        total = sum(counts.values())
        dist = {k: (counts[k] / total if total else 0.0) for k in counts}
        return {"strategy_counts": counts, "strategy_distribution": dist, "total_selections": total}
    
    def reset(self):
        """
        Reset compute selection history.
        """
        self.compute_selection_history = []