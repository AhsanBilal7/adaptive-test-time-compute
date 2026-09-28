"""Compute selector for BALROG agents: chooses Best-of-N, beam search, or lookahead and its parameter."""

import json
from typing import Dict, Any
from balrog.schemas import DecisionPayload
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
                f"Observation: {str(obs)}\n\n"
                "Guidelines (READ CAREFULLY, THEN OUTPUT JSON ONLY):\n"
                "1) Derive signals from Plan+Observation (string checks are fine):\n"
                "   - branching_signals: count of terms { 'branch', 'option', 'alternative', 'path', 'fork', 'subtask', 'search', 'explore' } + patterns like lists (', and', ';', numbered steps >1).\n"
                "   - verifier_signal: 1 if Plan mentions 'verify', 'verifier', 'check', 'constraint', or if a verifier/scorer tool is available; else 0.\n"
                "   - ranking_risk: 1 if the task is to choose/compare/order/rank/evaluate candidates or mentions 'tie', 'score', 'tradeoff'; else 0.\n"
                "   - clarity: 1 if instructions are single-step and unambiguous (no branching_signals, no question marks, no 'maybe', 'unsure', 'unclear'); else 0.\n"
                "\n"
                "2) Decide strategy by this table (first rule that matches wins):\n"
                "   A) IF branching_signals >= 2 OR (branching_signals >= 1 AND verifier_signal == 1) → strategy='beam_search' (use beam ∈ [2,8]).\n"
                "   B) ELSE IF ranking_risk == 1 → strategy='lookahead' (use k ∈ [1,3]).\n"
                "   C) ELSE IF clarity == 1 → strategy='best_of_n' (use n ∈ [2,8]).\n"
                "   D) ELSE → strategy='beam_search'.\n"
                "\n"
                "3) Parameter selection (be conservative by default):\n"
                "   - beam_search: beam = 2 if candidates <= 3; 4 if 4–6; 6–8 if >6 or high uncertainty.\n"
                "   - best_of_n: n = 2 if trivial; 4 if moderate; 6–8 if errors are costly or observation is noisy.\n"
                "   - lookahead: k = 1 for light re-ranking; 2 if close call; 3 if ties/near-ties persist.\n"
                "\n"
                "4) Hard constraints to avoid defaulting to best_of_n:\n"
                "   - You MAY NOT choose best_of_n if branching_signals >= 1.\n"
                "   - Prefer beam_search over best_of_n whenever verifier_signal == 1.\n"
                "   - Prefer lookahead over best_of_n whenever ranking_risk == 1.\n"
                "\n"
                "Return JSON ONLY with no explanation. Example: {\"strategy\":\"beam_search\",\"param\":4}\n"
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
            resp = self.client.generate_with_structured(messages, DecisionPayload.model_json_schema())
            text = resp.completion if hasattr(resp, "completion") else str(resp)
            print("Compute selection response text:", text)
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