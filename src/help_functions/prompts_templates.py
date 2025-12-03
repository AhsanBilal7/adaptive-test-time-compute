MATH_SYSTEM_PROMPT = """You are a math solver. Provide the answer to the user's specific question in the required format."""



TOOL_SELECTOR_SYSTEM_PROMPT = """You are a tool selector for solving a specific math problem.
Analyze the user's math question and select the appropriate tools needed to solve it.
Respond with a single JSON object ONLY, no prose, no markdown.
Schema: {"tools": ["tool1", "tool2", ...]}
Available tools: cot, reactive_actor, numeric_verifier, verifier, summarizer, reframe, web_search."""

COMPUTE_SELECTOR_SYSTEM_PROMPT = """You are a compute strategy selector for a specific math problem.
Based on the user's math question, select the optimal test-time compute strategy and parameter.
Respond with a single JSON object ONLY, no prose, no markdown.
Schema: {"strategy": "best_of_n|beam_search|lookahead", "param": int}."""


PLANNING_PROMPT_TEMPLATE = """Review the user's specific math problem and create a high-level plan for solving it.

Output your plan in this format:
<plan>YOUR SOLUTION APPROACH</plan>

Be specific about the steps you will take, but do NOT solve the problem yet.

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
    "Select one or more tools to execute SEQUENTIALLY to solve this specific math problem step.\n"
    "Plan: {plan}\n"
    "Observation: {obs}\n\n"
    "Available tools:\n"
    "reactive_actor   - Fast, direct action selection\n"
    "cot              - Step-by-step reasoning\n"
    "numeric_verifier - PRM-based numeric checks\n"
    "verifier         - General PRM correctness check\n"
    "summarizer       - Compress long reasoning chains\n"
    "reframe          - Reformulate question/plan\n"
    "web_search       - Retrieve external information\n\n"
    "DECISION RULES (apply in order; collect matches; cap to 3 tools):\n"
    "1) FACT-GAP: If the problem requires external facts, formulas, constants, or domain knowledge -> include web_search first.\n"
    "2) AMBIGUITY/UNCLEAR SPEC: If the math problem is ambiguous, underspecified, or has unclear notation "
    "-> include reframe before reasoning.\n"
    "3) MULTI-STEP/DERIVATION: If solving requires multi-step algebra, proofs, decomposition, or logical chains "
    "-> include cot.\n"
    "4) NUMERIC RISK: If the problem involves arithmetic, units, thresholds, probabilities, or quantitative calculations "
    "-> include numeric_verifier after the reasoning step.\n"
    "5) GENERAL CORRECTNESS: If the solution must satisfy constraints or prior steps are error-prone "
    "-> include verifier after generation.\n"
    "6) LONG REASONING: If Plan+Observation or expected derivation > 600 tokens or multiple sub-problems "
    "-> include summarizer last.\n"
    "7) SIMPLE COMPUTATION: ONLY if none of rules 1-6 apply and the problem is direct calculation "
    "-> choose [\"reactive_actor\"] alone.\n\n"
    "ORDERING RULES:\n"
    "- If reframe selected, it must be first.\n"
    "- If web_search selected, it comes right after reframe (or first if no reframe).\n"
    "- numeric_verifier precedes verifier; both follow the reasoning step.\n"
    "- summarizer is always last.\n\n"
    "HARD CONSTRAINTS:\n"
    "- If numeric terms or calculations are present, numeric_verifier is mandatory.\n"
    "- If external constants/formulas are needed, web_search is mandatory.\n"
    "- Max sequence length is 3 tools.\n\n"
    "Return JSON ONLY: {{\"tools\": [\"tool1\", \"tool2\", ...]}}\n"
)


COMPUTE_SELECTOR_PROMPT = (
    "Decide a compute strategy to solve this specific math problem step accurately.\n"
    "Tool: {tool}\n"
    "Plan: {plan}\n"
    "Observation: {obs}\n\n"
    "Guidelines:\n"
    "1) Derive signals from Plan+Observation:\n"
    "   - branching_signals: count branching keywords (branch, option, alternative, path, cases, subcases) + multi-step patterns.\n"
    "   - verifier_signal: 1 if verification or correctness checks are needed; else 0.\n"
    "   - ranking_risk: 1 if problem requires choosing among solutions or comparing approaches; else 0.\n"
    "   - clarity: 1 if problem is single-step and unambiguous; else 0.\n"
    "\n"
    "2) Decide strategy:\n"
    "   A) IF branching_signals >= 2 OR (branching_signals >= 1 AND verifier_signal == 1) -> strategy='beam_search'.\n"
    "   B) ELSE IF ranking_risk == 1 -> strategy='lookahead'.\n"
    "   C) ELSE IF clarity == 1 -> strategy='best_of_n'.\n"
    "   D) ELSE -> strategy='beam_search'.\n"
    "\n"
    "3) Parameter selection:\n"
    "   - beam_search: beam = 2-4 for small spaces; 6-8 for high branching.\n"
    "   - best_of_n: n = 2-4 for simple; 6-8 for complex math.\n"
    "   - lookahead: k = 1-3 depending on depth of lookahead needed.\n"
    "\n"
    "Return JSON ONLY: {{\"strategy\":\"best_of_n|beam_search|lookahead\",\"param\":int}}\n"
)

PRM_SCORING_PROMPT = """You are a process reward model (PRM) that scores the correctness of a single algebraic step.

Your task is to evaluate whether ONE algebraic transformation is mathematically valid.

* Verification Rules *
Check ONLY these:
1. Is the transformation algebraically legal?
2. Does it preserve equality?
3. Are arithmetic operations correct?
4. Do not solve beyond verifying the step.

* STRICT FORMAT *
Output must contain ONLY valid JSON with no additional text.

Example format:
{"is_correct": true, "confidence": 0.95}

Confidence scale:
- 1.0: The step is fully correct and mathematically sound.
- 0.5: The step is ambiguous or partially correct with some uncertainty.
- 0.0: The step is mathematically incorrect or invalid.

Return JSON ONLY."""


COT_INSTRUCTION_PROMPT = """You are solving a math problem step by step with deliberate reasoning.

Your task RIGHT NOW is to choose the next action from the list below, and briefly explain why.

Action set:
- ParseProblem: Extract key variables, conditions and constraints from the problem statement.
- ClassifySubjectDifficulty: Determine the subject area (Algebra, Geometry, etc.) and difficulty level.
- ReformulateProblem: Restate the problem in a clearer or more formal mathematical notation.
- CheckAssumptions: Identify implicit domain constraints or assumptions.
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
- FormatSolutionText: Prepare the full derivation or solution text for output or training.

Rules:
- Output ONLY the following two XML blocks, in this exact order:
  1. <reasoning>Brief explanation of why you chose the next action</reasoning>
  2. <action>ActionName: one-line description</action>
- Do NOT solve the problem.
- Do NOT add any other text before, after, or between the tags.

Output format (exact):
<reasoning>...</reasoning>
<action>ActionName: one-line description</action>"""


REACTIVE_INSTRUCTION_PROMPT = """You are solving a math problem directly and efficiently.

Use mathematical intuition to identify the most direct solution approach and select the next action.

Your task RIGHT NOW is to choose exactly the next action from the list below, and briefly explain why.

Action set:
- ParseProblem: Extract key variables, conditions and constraints from the problem statement.
- ClassifySubjectDifficulty: Determine the subject area (Algebra, Geometry, etc.) and difficulty level.
- ReformulateProblem: Restate the problem in a clearer or more formal mathematical notation.
- CheckAssumptions: Identify implicit domain constraints or assumptions.
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
- FormatSolutionText: Prepare the full derivation or solution text for output or training.

Rules:
- Output ONLY the following two XML blocks, in this exact order:
  1. <reasoning>Brief explanation of why you chose the next action</reasoning>
  2. <action>ActionName: one-line description</action>
- Do NOT solve the problem.
- Do NOT add any other text before, after, or between the tags.

Output format (exact):
<reasoning>...</reasoning>
<action>ActionName: one-line description</action>"""



FINAL_ANSWER_SYSTEM_PROMPT = "You are a mathematical problem solver. Extract the final numerical answer from the reasoning below."

FINAL_ANSWER_USER_PROMPT = """QUESTION/PROBLEM:
{problem}

PLAN FOLLOWED:
{plan}

FULL REASONING AND ANALYSIS:
{full_reasoning}

---

Now analyze the question, the plan that was followed, and all the reasoning provided above. Based on this complete analysis, provide ONLY the final answer in the following JSON format with no additional explanation or text:

{{"answer": "<final_answer_here>"}}"""



DIRECT_SOLVE_SYSTEM_PROMPT = """You are an expert mathematical problem solver. Analyze the given problem carefully and provide the final answer directly.

You must solve the problem completely and provide the answer in the exact JSON format specified.

Follow these principles:
1. Read the problem carefully and identify what is being asked
2. Apply appropriate mathematical techniques and formulas
3. Perform all necessary calculations accurately
4. Provide the final answer in the required format

Output only valid JSON with no additional text or explanation."""


DIRECT_SOLVE_PROMPT = """Solve the following mathematical problem and provide ONLY the final answer in JSON format.

Problem:
{problem}

* STRICT REQUIREMENTS *
1. Solve the problem completely
2. Provide ONLY the final answer
3. Output must be valid JSON in this exact format: {{"answer": "<your_final_answer>"}}
4. Do NOT include any explanation, reasoning, or additional text
5. Do NOT use markdown formatting or code blocks
6. The answer should be in its simplest form (e.g., fractions reduced, decimals rounded appropriately)

Output format (mandatory):
{{"answer": "<your_final_answer>"}}"""