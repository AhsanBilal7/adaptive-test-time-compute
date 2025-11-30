MATH_SYSTEM_PROMPT = """You are a math solver. Provide the answer to the user's specific question in the required format."""


PLANNING_PROMPT_TEMPLATE = """Review the problem carefully and create a high-level plan for solving it.

Output your plan in this format:
<plan>YOUR SOLUTION APPROACH</plan>

Be specific about the steps you will take, but do NOT solve the problem yet.

You are a mathematical problem planner.

Your task is to produce a high-level solution approach for the following problem.

Problem:
{problem}

* STRICT REQUIREMENTS *
1. DO NOT solve the problem.
2. DO NOT perform algebra, arithmetic, or simplification.
3. Write 1-3 sentence plan only.
4. Wrap the plan EXACTLY inside <plan>...</plan>
5. No text is allowed outside the <plan> tags.

Output format (mandatory):
<plan>STEP-BY-STEP HIGH-LEVEL STRATEGY ONLY</plan>"""


TOOL_SELECTOR_PROMPT = (
    "Select one or more tools to execute SEQUENTIALLY for this step.\n"
    "Plan: {plan}\n"
    "Observation: {obs}\n\n"
    "Available tools:\n"
    "reactive_actor  - Fast, direct action selection\n"
    "cot             - Step-by-step reasoning\n"
    "heuristic_script- Rule-based domain policy\n"
    "numeric_verifier- PRM-based numeric checks\n"
    "verifier        - General PRM correctness check\n"
    "summarizer      - Compress long reasoning chains\n"
    "reframe         - Reformulate question/plan\n"
    "web_search      - Retrieve external information\n\n"
    "DECISION RULES (apply in order; collect matches; cap to 3 tools):\n"
    "1) FACT-GAP: If Plan/Observation indicates missing facts, recency, URLs, or uncertainty about world knowledge -> include web_search first.\n"
    "2) AMBIGUITY/POOR SPEC: If question is unclear, contradictory, or underspecified (markers: 'unclear', '?', 'maybe', multiple interpretations) -> include reframe before any reasoning.\n"
    "3) MULTI-STEP/DERIVATION: If solving requires multi-step logic, decomposition, proofs, or algorithm design -> include cot.\n"
    "4) DOMAIN RULES: If a known rule-based policy applies (e.g., fixed heuristics/workflows) -> include heuristic_script (before cot if it can prune the space).\n"
    "5) NUMERIC RISK: If arithmetic, units, thresholds, probabilities, or quantitative constraints appear -> include numeric_verifier after the generator (cot/heuristic_script/reactive_actor).\n"
    "6) GENERAL CORRECTNESS: If the output must satisfy constraints/specs or prior errors are likely -> include verifier after generation (and after numeric_verifier if both are used).\n"
    "7) LONG CONTEXT: If Plan+Observation or expected chain > 600 tokens or multiple sub-answers -> include summarizer last to compress.\n"
    "8) TRIVIALITY: ONLY if none of rules 1-7 fired and the task is single-step with explicit action -> choose [reactive_actor] alone.\n\n"
    "ORDERING RULES:\n"
    "- If reframe selected, it must be first.\n"
    "- If web_search selected, it comes right after reframe (or first if no reframe).\n"
    "- heuristic_script precedes cot if both selected.\n"
    "- numeric_verifier precedes verifier; both follow the generator (reactive_actor/heuristic_script/cot).\n"
    "- summarizer is always last.\n\n"
    "HARD CONSTRAINTS:\n"
    "- Do NOT output only [\"reactive_actor\"] if any of rules 1-7 matched.\n"
    "- If any numeric terms are present, numeric_verifier is mandatory.\n"
    "- If external facts are referenced or freshness matters, web_search is mandatory.\n"
    "- Max sequence length is 3 tools; prefer the most impactful ones per rules above.\n\n"
    "TEMPLATES (examples, not prescriptive):\n"
    "- Simple, unambiguous action -> [\"reactive_actor\"]\n"
    "- Needs facts then reasoning with checks -> [\"web_search\", \"cot\", \"verifier\"]\n"
    "- Rule-based pruning then numeric check -> [\"heuristic_script\", \"numeric_verifier\"]\n"
    "- Ambiguous prompt then plan+verify -> [\"reframe\", \"cot\", \"verifier\"]\n"
    "- Long chain to compress -> [\"cot\", \"summarizer\"]\n\n"
    'Return JSON ONLY. Example: {"tools": ["web_search", "cot", "numeric_verifier"]}'
)


COMPUTE_SELECTOR_PROMPT = (
    "Decide a compute strategy for the next action.\n"
    "Tool: {tool}\n"
    "Plan: {plan}\n"
    "Observation: {obs}\n\n"
    "Guidelines (READ CAREFULLY, THEN OUTPUT JSON ONLY):\n"
    "1) Derive signals from Plan+Observation (string checks are fine):\n"
    "   - branching_signals: count of terms { 'branch', 'option', 'alternative', 'path', 'fork', 'subtask', 'search', 'explore' } + patterns like lists (', and', ';', numbered steps >1).\n"
    "   - verifier_signal: 1 if Plan mentions 'verify', 'verifier', 'check', 'constraint', or if a verifier/scorer tool is available; else 0.\n"
    "   - ranking_risk: 1 if the task is to choose/compare/order/rank/evaluate candidates or mentions 'tie', 'score', 'tradeoff'; else 0.\n"
    "   - clarity: 1 if instructions are single-step and unambiguous (no branching_signals, no question marks, no 'maybe', 'unsure', 'unclear'); else 0.\n"
    "\n"
    "2) Decide strategy by this table (first rule that matches wins):\n"
    "   A) IF branching_signals >= 2 OR (branching_signals >= 1 AND verifier_signal == 1) -> strategy='beam_search' (use beam in [2,8]).\n"
    "   B) ELSE IF ranking_risk == 1 -> strategy='lookahead' (use k in [1,3]).\n"
    "   C) ELSE IF clarity == 1 -> strategy='best_of_n' (use n in [2,8]).\n"
    "   D) ELSE -> strategy='beam_search'.\n"
    "\n"
    "3) Parameter selection (be conservative by default):\n"
    "   - beam_search: beam = 2 if candidates <= 3; 4 if 4-6; 6-8 if >6 or high uncertainty.\n"
    "   - best_of_n: n = 2 if trivial; 4 if moderate; 6-8 if errors are costly or observation is noisy.\n"
    "   - lookahead: k = 1 for light re-ranking; 2 if close call; 3 if ties/near-ties persist.\n"
    "\n"
    "4) Hard constraints to avoid defaulting to best_of_n:\n"
    "   - You MAY NOT choose best_of_n if branching_signals >= 1.\n"
    "   - Prefer beam_search over best_of_n whenever verifier_signal == 1.\n"
    "   - Prefer lookahead over best_of_n whenever ranking_risk == 1.\n"
    "\n"
    'Return JSON ONLY with no explanation. Example: {"strategy":"beam_search","param":4}\n'
)


PRM_SCORING_PROMPT = """
Evaluate this response based on:
1. Alignment with the plan and goals
2. Appropriateness for the current observation
3. Likelihood of making progress toward task completion
4. Correctness and safety of the proposed action

Provide a score between 0.0 and 1.0:
- 0.0-0.3: Poor response (unhelpful, incorrect, or counterproductive)
- 0.3-0.5: Below average (some issues, suboptimal)
- 0.5-0.7: Average (reasonable, acceptable)
- 0.7-0.9: Good (appropriate, effective)
- 0.9-1.0: Excellent (optimal, highly aligned)

Output your score in the format:
<score>X.XX</score>
Replace X.XX with your numeric score.
"""


REACTIVE_INSTRUCTION_PROMPT = """Solve it directly and efficiently.

Use your mathematical intuition to quickly identify the solution approach and execute it.
Focus on the most direct path to the answer.

You are about to solve a math problem. Your task RIGHT NOW is to choose exactly next action from the list below.

Action set:
- ParseProblem: Extract key variables, conditions and constraints from the problem statement.
- ClassifySubjectDifficulty: Determine the subject area (Algebra, Geometry, etc.) and difficulty level.
- IdentifyKeywordsHeuristics: Scan for cues or heuristics that suggest specific techniques.
- ReformulateProblem: Restate the problem in a clearer or more formal mathematical notation.
- CheckAssumptions: Identify implicit domain constraints or assumptions.
- SelectStrategy: Choose a problem-solving strategy or heuristic to apply.
- DecomposeSubproblems: Break the main problem into smaller sub-tasks or cases.
- IdentifyToolsFormulas: List relevant formulas, theorems or tools required.
- EstimateFeasibilityCheck: Do a quick check of plausibility, bounds or magnitudes.
- PerformComputation: Execute an algebraic or numeric computation step.
- CaseAnalysis: Carry out one case in a case-by-case analysis.
- ConstructDiagramOrAuxiliary: For geometry or spatial problems, create an auxiliary construction or diagram.
- CombineResults: Combine results from sub-tasks or cases into an aggregate expression.
- SimplifyFinalizeExpression: Simplify the final expression into a standard form.
- SanityCheckFinalAnswer: Plug in special cases or check boundary values to verify reasonableness.
- BoxFinalAnswer: Format the final answer in the expected output style.
- ReviewSolution: Review the full solution chain for logic or hidden assumptions.
- GeneraliseOrEdgeCaseCheck: Consider extreme or boundary cases to ensure full correctness.
- AnnotateHeuristicUsed: Record which heuristic(s) were applied.
- FormatSolutionText: Prepare the full derivation or solution text for output or training.

Rules:
- Output ONLY the chosen action inside <action>...</action> tags.
- Inside the tags, write exactly: ActionName: one-line description.
- Do NOT solve the problem, do NOT include any other text before or after the tags.

Output format (exact):
<action>ActionName: one-line description</action>"""


COT_INSTRUCTION_PROMPT = """Solve it step by step.

You are about to solve a math problem. Your task RIGHT NOW is to choose next action from the list below.

Action set:
- ParseProblem: Extract key variables, conditions and constraints from the problem statement.
- ClassifySubjectDifficulty: Determine the subject area (Algebra, Geometry, etc.) and difficulty level.
- IdentifyKeywordsHeuristics: Scan for cues or heuristics that suggest specific techniques.
- ReformulateProblem: Restate the problem in a clearer or more formal mathematical notation.
- CheckAssumptions: Identify implicit domain constraints or assumptions.
- SelectStrategy: Choose a problem-solving strategy or heuristic to apply.
- DecomposeSubproblems: Break the main problem into smaller sub-tasks or cases.
- IdentifyToolsFormulas: List relevant formulas, theorems or tools required.
- EstimateFeasibilityCheck: Do a quick check of plausibility, bounds or magnitudes.
- PerformComputation: Execute an algebraic or numeric computation step.
- CaseAnalysis: Carry out one case in a case-by-case analysis.
- ConstructDiagramOrAuxiliary: For geometry or spatial problems, create an auxiliary construction or diagram.
- CombineResults: Combine results from sub-tasks or cases into an aggregate expression.
- SimplifyFinalizeExpression: Simplify the final expression into a standard form.
- SanityCheckFinalAnswer: Plug in special cases or check boundary values to verify reasonableness.
- BoxFinalAnswer: Format the final answer in the expected output style.
- ReviewSolution: Review the full solution chain for logic or hidden assumptions.
- GeneraliseOrEdgeCaseCheck: Consider extreme or boundary cases to ensure full correctness.
- AnnotateHeuristicUsed: Record which heuristic(s) were applied.
- FormatSolutionText: Prepare the full derivation or solution text for output or training.

Rules:
- Output ONLY the chosen action inside <action>...</action> tags.
- Inside the tags, write exactly: ActionName: one-line description.
- Do NOT solve the problem, do NOT include any other text before or after the tags

Output format (exact):
<action>ActionName: one-line description</action>"""


FINAL_ANSWER_SYSTEM_PROMPT = "You are a mathematical problem solver. Extract the final numerical answer from the reasoning below."