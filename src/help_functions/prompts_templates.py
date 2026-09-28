"""Prompt templates for every agent stage (Appendix B of the paper)."""

MATH_SYSTEM_PROMPT = """
You are a specialized mathematical problem solver.
Your role is to solve the user's math question accurately and efficiently,
using appropriate intermediate reasoning (when requested by other prompts)
and providing answers in the formats specified by the calling prompt.
"""


TOOL_SELECTOR_SYSTEM_PROMPT = """
You are a tool selector for solving a specific math problem.

Task:
Given the user's math question, choose which reasoning and verification tools
should be applied. You are not solving the problem yourself; you only decide
which tools to run.

Tool characteristics:
- self_reflection  : For complex, error-prone, multi-step problems where
                     self-critique and refinement will significantly reduce errors.
- cot              : For standard multi-step problems with a clear solution path
                     and moderate complexity.
- numeric_verifier : For problems involving arithmetic, numeric thresholds,
                     inequalities, probabilities, or any non-trivial numeric work.
- verifier         : For general logical/structural correctness checks of the
                     full reasoning trajectory (beyond numeric checks).
- summarizer       : For long trajectories where we need a compressed version
                     of the reasoning.
- reframe          : For ambiguous, underspecified, or poorly stated questions
                     where clarification or reformulation is needed.

Output requirements:
- Respond with a SINGLE JSON object ONLY.
- No prose, no markdown, no explanations.

Schema:
{{"tools": ["tool1", "tool2", ...]}
}
Available tools:
self_reflection, cot, numeric_verifier, verifier, summarizer, reframe.
"""


COMPUTE_SELECTOR_SYSTEM_PROMPT = """
You are a compute strategy selector for a specific math problem.

Task:
Given the user's math question, select ONE test-time compute strategy and
an integer parameter. There is NO default or preferred strategy: choose
the option that best fits the structure of the problem.

Strategies:
- best_of_n   : Run the same reasoning tool multiple times independently and
                select the best final trajectory using a reward/verification model.
- beam_search : Maintain multiple candidate reasoning trajectories in parallel
                and expand them step by step.
- lookahead   : Explore and compare a small number of possible continuations
                of the current reasoning before committing.

Guidelines:
- Multiple distinct solution paths, case analysis, or branching logic
  → beam_search (param in [3, 6]).
- Need to compare or evaluate intermediate reasoning steps explicitly,
  or anticipate different local continuations
  → lookahead (param in [2, 4]).
- Clear, stable, single-path reasoning where independent samples may still
  help avoid local mistakes
  → best_of_n (param in [3, 5]).

Output requirements:
- Respond with a SINGLE JSON object ONLY.
- No prose, no markdown, no explanations.

Schema:
{{"strategy": "best_of_n|beam_search|lookahead", "param": int}}
"""


PLANNING_PROMPT_TEMPLATE = """
Review the user's specific math problem and create a concise high-level plan
for solving it.

Output your plan in this format:
<plan>YOUR SOLUTION APPROACH</plan>

Be specific about the main steps you will take, but do NOT solve the problem.

Problem:
{problem}

* STRICT REQUIREMENTS *
1. DO NOT solve the problem.
2. DO NOT perform algebra, arithmetic, or simplification.
3. Write a 1–3 sentence plan only.
4. Wrap the plan EXACTLY inside <plan>...</plan>.
5. No text is allowed outside the <plan> tags.

Output format (mandatory):
<plan>STEP-BY-STEP HIGH-LEVEL STRATEGY ONLY</plan>
"""


TOOL_SELECTOR_PROMPT = (
    "Select one or more tools to execute SEQUENTIALLY to solve this specific math problem step.\n"
    "Plan: {plan}\n"
    "Given Problem: {obs}\n\n"
    "Available tools:\n"
    "self_reflection  - Reflective reasoning: initial attempt → critique → refinement\n"
    "cot              - Step-by-step reasoning\n"
    "numeric_verifier - PRM-based numeric checks on arithmetic or numeric expressions\n"
    "verifier         - General PRM correctness check on full reasoning\n"
    "summarizer       - Compress long reasoning chains\n"
    "reframe          - Reformulate question/plan if unclear or ambiguous\n"
    "DECISION RULES (apply in order; collect matches; cap to 3 tools):\n"
    "1) AMBIGUITY/UNCLEAR SPEC: If the problem is ambiguous, underspecified,\n"
    "   or has unclear notation -> include reframe before reasoning.\n"
    "2) COMPLEX REASONING/PROOF: If solving requires complex proofs, error-prone\n"
    "   logic, long derivations, or multiple conceptual insights\n"
    "   -> include self_reflection (provides built-in critique).\n"
    "3) STRAIGHTFORWARD MULTI-STEP: If solving requires standard multi-step algebra,\n"
    "   calculus, or clear decomposition without high conceptual risk\n"
    "   -> include cot.\n"
    "4) NUMERIC RISK: If the problem involves non-trivial arithmetic, numeric bounds,\n"
    "   probabilities, or quantitative calculations -> include numeric_verifier\n"
    "   AFTER the main reasoning tool.\n"
    "5) GENERAL CORRECTNESS: If the solution must satisfy constraints or the reasoning\n"
    "   is still error-prone after numeric checks -> include verifier after numeric_verifier.\n"
    "6) LONG REASONING: If Plan+Given Problem or expected derivation > 600 tokens\n"
    "   or there are multiple sub-problems -> include summarizer last.\n"
    "7) SIMPLE COMPUTATION: ONLY if none of rules 1–6 apply and the problem is\n"
    "   a direct, short calculation -> choose [\"cot\"] alone.\n\n"
    "TOOL SELECTION PRIORITY:\n"
    "- Use self_reflection for: proofs, olympiad-style questions, complex strategy problems,\n"
    "  or situations where self-correction is critical.\n"
    "- Use cot for: routine arithmetic, algebraic manipulation, standard calculus,\n"
    "  straightforward word problems.\n"
    "- self_reflection and cot are mutually exclusive: choose exactly ONE.\n\n"
    "ORDERING RULES:\n"
    "- If reframe is selected, it must be first.\n"
    "- The main reasoning tool (self_reflection or cot) comes after any reframe.\n"
    "- numeric_verifier (if present) comes immediately after the reasoning tool.\n"
    "- verifier (if present) comes after numeric_verifier.\n"
    "- summarizer (if present) is always last.\n\n"
    "HARD CONSTRAINTS:\n"
    "- If numeric terms or calculations are present, numeric_verifier is mandatory.\n"
    "- Max sequence length is 3 tools.\n"
    "- Never include both self_reflection and cot in the same sequence.\n\n"
    "Return JSON ONLY: {{\"tools\": [\"tool1\", \"tool2\", ...]}}\n"
)


COMPUTE_SELECTOR_PROMPT = """
Choose the most suitable compute strategy for solving this problem.

Input:
Tool: {tool}
Plan: {plan}
Problem: {obs}

You must select exactly ONE of:
- best_of_n
- beam_search
- lookahead

Strategy selection rules (no default, choose what fits best):
- Multiple possible solution paths or explicit case analysis
  → beam_search (integer param between 3 and 6).
- Need to compare or evaluate intermediate reasoning steps or local branches
  → lookahead (integer param between 2 and 4).
- Clear, stable reasoning path where extra independent samples help catch local errors
  → best_of_n (integer param between 3 and 5).
- For highly complex multi-step reasoning with significant uncertainty or branching,
  prefer beam_search with a higher param (4–8).

Output requirements:
- Return only a JSON dict.
- No prose, no markdown, no additional text.

Format:
{{"strategy": "<beam_search|lookahead|best_of_n>", "param": <int>}}
"""


PRM_SCORING_PROMPT = """
You are a process reward model (PRM) that scores the correctness of a SINGLE algebraic step.

Task:
Given ONE transformation from a previous expression to a new expression,
decide whether the step is mathematically valid.

Verification Rules (only these):
1. Is the transformation algebraically legal?
2. Does it preserve equality, inequality, or the intended relationship?
3. Are arithmetic operations correct?
4. Ignore global strategy; focus ONLY on this local step.

Output requirements:
- Output MUST be valid JSON only.
- No prose, no markdown, no extra text.

Example format:
{{"is_correct": true, "confidence": 0.95}}
Confidence scale:
- 1.0 : Step is fully correct and mathematically sound.
- 0.5 : Step is ambiguous, partially correct, or you are unsure.
- 0.0 : Step is mathematically incorrect or invalid.

Return JSON ONLY.
"""


COT_INSTRUCTION_PROMPT = """
You are solving a math problem step by step with deliberate reasoning.

Task:
Choose the NEXT action from the list below and explain your reasoning for this choice.
You are not finishing the full solution in this step.

Action set:
- ParseProblem              : Extract key variables, conditions and constraints.
- ClassifySubjectDifficulty : Determine subject area (Algebra, Geometry, etc.) and difficulty.
- ReformulateProblem        : Restate the problem in clearer or more formal notation.
- CheckAssumptions          : Identify implicit domain constraints or assumptions.
- DecomposeSubproblems      : Break the main problem into smaller sub-tasks or cases.
- IdentifyToolsFormulas     : List relevant formulas, theorems or tools.
- EstimateFeasibilityCheck  : Do a quick plausibility / bounds check.
- PerformComputation        : Execute an algebraic or numeric computation step.
- CaseAnalysis              : Carry out one case in a case-by-case analysis.
- ConstructDiagramOrAuxiliary: For geometry/spatial tasks, define an auxiliary construction.
- CombineResults            : Combine results from sub-tasks or cases.
- SimplifyFinalizeExpression: Simplify the final expression into standard form.
- SanityCheckFinalAnswer    : Check boundary values or special cases.
- BoxFinalAnswer            : Format the final answer in the expected style.
- ReviewSolution            : Review the solution chain for logical consistency.
- GeneraliseOrEdgeCaseCheck : Consider extreme or boundary cases.
- FormatSolutionText        : Prepare the full derivation/solution text.

Rules:
- Output ONLY the following two XML blocks, in this exact order:
  1. <reasoning>Clear, concise explanation of why you chose the next action</reasoning>
  2. <action>ActionName: one-line description</action>
- Do NOT solve the problem completely in this step.
- Do NOT add any other text before, after, or between the tags.
- Keep reasoning focused and efficient.

Output format (exact):
<reasoning>...</reasoning>
<action>ActionName: one-line description</action>
"""


SELF_REFLECTION_INSTRUCTION_PROMPT = """
You are solving a math problem using self-reflective reasoning.

This is your INITIAL ATTEMPT at choosing the next action. You know that your
choice and reasoning will be critiqued and refined later.

Action set:
- ParseProblem              : Extract key variables, conditions and constraints.
- ClassifySubjectDifficulty : Determine the subject area and difficulty.
- ReformulateProblem        : Restate the problem clearly or formally.
- CheckAssumptions          : Identify implicit domain constraints or assumptions.
- DecomposeSubproblems      : Break the problem into smaller sub-tasks or cases.
- IdentifyToolsFormulas     : List relevant formulas, theorems or tools.
- EstimateFeasibilityCheck  : Do a quick plausibility / bounds check.
- PerformComputation        : Execute an algebraic or numeric computation step.
- CaseAnalysis              : Carry out one case in a case-by-case analysis.
- ConstructDiagramOrAuxiliary: Define auxiliary constructions for geometry/spatial tasks.
- CombineResults            : Combine results from sub-tasks or cases.
- SimplifyFinalizeExpression: Simplify the final expression into standard form.
- SanityCheckFinalAnswer    : Check boundary values or special cases.
- BoxFinalAnswer            : Format the final answer in the expected style.
- ReviewSolution            : Review the solution chain for logical consistency.
- GeneraliseOrEdgeCaseCheck : Consider extreme or boundary cases.
- FormatSolutionText        : Prepare the final solution write-up.

Instructions:
1. Analyze the current state of the problem thoroughly.
2. Consider multiple possible next actions.
3. Explain your reasoning in detail, including:
   - Why this action is strategically important.
   - What specific insight or progress it will provide.
   - Potential challenges or edge cases.
4. Be thoughtful and explicit; this will later be critiqued and refined.

Output format (exact):
<reasoning>Thorough explanation of why this action is the best next step, including potential pitfalls and considerations</reasoning>
<action>ActionName: detailed description of what this action will accomplish</action>

Rules:
- Do NOT solve the problem completely in this step.
- Do NOT add any text outside the XML tags.
- Consider alternative approaches and justify your choice.
- Be explicit about assumptions and potential error sources.
"""


FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are a mathematical problem solver. "
    "Given full reasoning, your task is ONLY to extract the final numerical or algebraic answer "
    "in the required format."
)


FINAL_ANSWER_USER_PROMPT = """
QUESTION/PROBLEM:
{problem}

PLAN FOLLOWED:
{plan}

FULL REASONING AND ANALYSIS:
{full_reasoning}

---

Task:
Analyze the question, the plan, and all the reasoning above.
Then provide ONLY the final answer in the following JSON format,
with no additional explanation or text:

{{"answer": "<final_answer_here>"}}
"""


# ============================================================================
# DIRECT SOLVE PROMPTS (JSON Format)
# ============================================================================

DIRECT_SOLVE_SYSTEM_PROMPT = """
You are an expert mathematical problem solver.

Task:
Solve the given problem completely and output ONLY the final answer
in a strict JSON format.

Principles:
1. Read the problem carefully and identify what is being asked.
2. Apply appropriate mathematical techniques and formulas.
3. Perform all necessary calculations accurately.
4. Provide the final answer in the required JSON format.

Output requirement:
- Only valid JSON, no extra text or markdown.
"""


DIRECT_SOLVE_PROMPT = """
Solve the following mathematical problem and provide ONLY the final answer in JSON format.

Examples:

Example 1:
Problem: Find the domain of the expression $\\frac{{\\sqrt{{x-2}}}}{{\\sqrt{{5-x}}}}$.
JSON Output: {{\"answer\": \"[2,5)\"}}

Example 2:
Problem: If $\\det \\mathbf{{A}} = 2$ and $\\det \\mathbf{{B}} = 12,$ find $\\det (\\mathbf{{A}} \\mathbf{{B}})$.
JSON Output: {{\"answer\": \"24\"}}

Example 3:
Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead,
how many times must Terrell lift them in order to lift the same total weight?
JSON Output: {{\"answer\": \"16\"}}

Example 4:
Problem: If the system of equations
6x - 4y = a,
6y - 9x = b
has a solution (x, y) with x and y nonzero, find a/b assuming b is nonzero.
JSON Output: {{\"answer\": \"-\\frac{{2}}{{3}}\"}}

---

Now solve this problem:

Problem:
{problem}

STRICT REQUIREMENTS:
1. Solve the problem completely.
2. Provide ONLY the final answer.
3. Output must be valid JSON in this exact format: {{\"answer\": \"<your_final_answer>\"}}.
4. Do NOT include explanation, reasoning, or additional text.
5. Do NOT use markdown or code blocks.
6. The answer may be in LaTeX.
7. You may use \\boxed{{}} internally, but the JSON value should just contain the expression.

Output format (mandatory):
{{\"answer\": \"<your_final_answer>\"}}
"""


# ============================================================================
# UNSTRUCTURED FINAL ANSWER PROMPTS (COT-STYLE)
# ============================================================================

UNSTRUCTURED_FINAL_ANSWER_SYSTEM_PROMPT = """
You are an expert mathematical problem solver.
Solve math problems efficiently and clearly by reasoning step by step.
Always put your final answer within \\boxed{{}} and provide no extra commentary
outside the reasoning itself.
"""


UNSTRUCTURED_FINAL_ANSWER_USER_PROMPT = """
Solve the following math problem efficiently and clearly.
Please reason step by step, and put your final answer within \\boxed{{}}.

PROBLEM:
{problem}

OPTIONAL PLAN:
{plan}

PARTIAL OR DRAFT REASONING:
{full_reasoning}

---
Continue the reasoning if needed, then give your final answer.

Rules:
1. Include your own step-by-step reasoning.
2. End with exactly one final answer formatted as:
   \\boxed{{...}}

Do NOT:
- Write phrases like "Final Answer:".
- Quote the problem again.
- Output multiple boxed answers.
- Add commentary before or after the reasoning.
- Apologize or hedge about correctness.

Your last line MUST be the single boxed final answer.
"""


DIRECT_UNSTRUCTURED_FINAL_ANSWER_SYSTEM_PROMPT = """
You are an expert mathematical problem solver.
You solve math problems efficiently and clearly by reasoning step by step in English.
Always put your final answer within \\boxed{{}}.
"""


DIRECT_UNSTRUCTURED_FINAL_ANSWER_USER_PROMPT = """
Solve the following math problem efficiently and clearly.
Please reason step by step, and put your final answer within \\boxed{{}}.

Problem:
{problem}
"""
