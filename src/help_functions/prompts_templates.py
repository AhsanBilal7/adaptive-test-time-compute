MATH_SYSTEM_PROMPT = """You are a math solver. Provide the answer to the user's specific question in the required format."""


TOOL_SELECTOR_SYSTEM_PROMPT = """You are a tool selector for solving a specific math problem.
Analyze the user's math question and select the appropriate tools needed to solve it.

Tool characteristics:
- self_reflection: Best for complex, error-prone problems; includes automatic critique and refinement
- cot: Best for straightforward multi-step problems with clear solution paths

Respond with a single JSON object ONLY, no prose, no markdown.
Schema: {"tools": ["tool1", "tool2", ...]}
Available tools: self_reflection, cot, numeric_verifier, verifier, summarizer, reframe, web_search."""


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
    "Given Problem: {obs}\n\n"
    "Available tools:\n"
    "self_reflection  - Reflective reasoning: initial attempt → critique → refinement\n"
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
    "3) COMPLEX REASONING/PROOF: If solving requires complex proofs, error-prone logic, multi-step derivations, "
    "or strategic decision-making where mistakes are likely -> include self_reflection (provides built-in critique).\n"
    "4) STRAIGHTFORWARD MULTI-STEP: If solving requires standard multi-step algebra or clear decomposition "
    "without significant risk of conceptual errors -> include cot.\n"
    "5) NUMERIC RISK: If the problem involves arithmetic, units, thresholds, probabilities, or quantitative calculations "
    "-> include numeric_verifier after the reasoning step.\n"
    "6) GENERAL CORRECTNESS: If the solution must satisfy constraints or prior steps are error-prone "
    "-> include verifier after generation.\n"
    "7) LONG REASONING: If Plan+Given Problem or expected derivation > 600 tokens or multiple sub-problems "
    "-> include summarizer last.\n"
    "8) SIMPLE COMPUTATION: ONLY if none of rules 1-7 apply and the problem is direct calculation "
    "-> choose [\"cot\"] alone.\n\n"
    "TOOL SELECTION PRIORITY:\n"
    "- Use self_reflection for: proofs, complex strategy problems, olympiad-level questions, problems requiring "
    "multiple conceptual insights, or when error correction is critical.\n"
    "- Use cot for: standard arithmetic, algebraic manipulation, routine calculus, straightforward word problems.\n"
    "- self_reflection and cot are mutually exclusive - choose ONE based on complexity.\n\n"
    "ORDERING RULES:\n"
    "- If reframe selected, it must be first.\n"
    "- If web_search selected, it comes right after reframe (or first if no reframe).\n"
    "- Reasoning tool (self_reflection or cot) comes after any reframe/web_search.\n"
    "- numeric_verifier precedes verifier; both follow the reasoning step.\n"
    "- summarizer is always last.\n\n"
    "HARD CONSTRAINTS:\n"
    "- If numeric terms or calculations are present, numeric_verifier is mandatory.\n"
    "- If external constants/formulas are needed, web_search is mandatory.\n"
    "- Max sequence length is 3 tools.\n"
    "- Never include both self_reflection and cot in the same sequence.\n\n"
    "Return JSON ONLY: {{\"tools\": [\"tool1\", \"tool2\", ...]}}\n"
)

COMPUTE_SELECTOR_PROMPT = """Choose compute strategy for this problem.

Tool: {tool}
Plan: {plan}
Problem: {obs}

Strategy selection:
- Multiple solution paths or case analysis → beam_search (param: 3-6)
- Need to compare approaches → lookahead (param: 2-4)
- Single clear path → best_of_n (param: 3-5)
- Complex multi-step → beam_search (param: 4-8)

Output JSON only: {{"strategy": "best_of_n", "param": 5}}"""

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

Your task is to choose the next action from the list below and explain your reasoning.

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
  1. <reasoning>Clear, concise explanation of why you chose the next action</reasoning>
  2. <action>ActionName: one-line description</action>
- Do NOT solve the problem completely in this step.
- Do NOT add any other text before, after, or between the tags.
- Keep reasoning focused and efficient.

Output format (exact):
<reasoning>...</reasoning>
<action>ActionName: one-line description</action>"""


SELF_REFLECTION_INSTRUCTION_PROMPT = """You are solving a math problem using self-reflective reasoning.

This is your INITIAL ATTEMPT. Propose the next action, knowing that you will critique and refine it.

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

Instructions:
1. Analyze the current state of the problem thoroughly
2. Consider multiple possible next actions
3. Explain your reasoning in detail, including:
   - Why this action is strategically important
   - What specific insights or progress it will provide
   - Potential challenges or edge cases to watch for
4. Be thoughtful - your reasoning will be critiqued and refined

Output format (exact):
<reasoning>Thorough explanation of why this action is the best next step, including potential pitfalls and considerations</reasoning>
<action>ActionName: detailed description of what this action will accomplish</action>

Rules:
- Do NOT solve the problem completely in this step.
- Do NOT add any text outside the XML tags.
- Provide substantial reasoning that can be meaningfully critiqued.
- Consider alternative approaches and explain why you chose this one.
- Be explicit about assumptions and potential error sources."""

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



# ============================================================================
# DIRECT SOLVE PROMPTS (JSON Format)
# ============================================================================

DIRECT_SOLVE_SYSTEM_PROMPT = """You are an expert mathematical problem solver. Analyze the given problem carefully and provide the final answer directly.

You must solve the problem completely and provide the answer in the exact JSON format specified.

Follow these principles:
1. Read the problem carefully and identify what is being asked
2. Apply appropriate mathematical techniques and formulas
3. Perform all necessary calculations accurately
4. Provide the final answer in the required format

Here are some examples of problems and their solutions:

Example 1:
Problem: Find the domain of the expression $\\frac{{\\sqrt{{x-2}}}}{{\\sqrt{{5-x}}}}$.
Solution: The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{{[2,5)}}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Example 2:
Problem: If $\\det \\mathbf{{A}} = 2$ and $\\det \\mathbf{{B}} = 12,$ then find $\\det (\\mathbf{{A}} \\mathbf{{B}}).$
Solution: We have that $\\det (\\mathbf{{A}} \\mathbf{{B}}) = (\\det \\mathbf{{A}})(\\det \\mathbf{{B}}) = (2)(12) = \\boxed{{24}}$.
Final Answer: The final answer is $24$. I hope it is correct.

Example 3:
Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution: If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:
\\begin{{align*}}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{{16}}
\\end{{align*}}
Final Answer: The final answer is $16$. I hope it is correct.

Example 4:
Problem: If the system of equations
\\begin{{align*}}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.
Solution: If we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain
$$6y-9x=-\\frac{{3}}{{2}}a.$$Since we also know that $6y-9x=b$, we have
$$-\\frac{{3}}{{2}}a=b\\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$$
Final Answer: The final answer is $-\\frac{{2}}{{3}}$. I hope it is correct.

Output only valid JSON with no additional text or explanation."""


DIRECT_SOLVE_PROMPT = """Solve the following mathematical problem and provide ONLY the final answer in JSON format.

Here are some examples:

Example 1:
Problem: Find the domain of the expression $\\frac{{\\sqrt{{x-2}}}}{{\\sqrt{{5-x}}}}$.
Solution: The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{{[2,5)}}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.
JSON Output: {{"answer": "[2,5)"}}

Example 2:
Problem: If $\\det \\mathbf{{A}} = 2$ and $\\det \\mathbf{{B}} = 12,$ then find $\\det (\\mathbf{{A}} \\mathbf{{B}}).$
Solution: We have that $\\det (\\mathbf{{A}} \\mathbf{{B}}) = (\\det \\mathbf{{A}})(\\det \\mathbf{{B}}) = (2)(12) = \\boxed{{24}}$.
Final Answer: The final answer is $24$. I hope it is correct.
JSON Output: {{"answer": "24"}}

Example 3:
Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution: If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:
\\begin{{align*}}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{{16}}
\\end{{align*}}
Final Answer: The final answer is $16$. I hope it is correct.
JSON Output: {{"answer": "16"}}

Example 4:
Problem: If the system of equations
\\begin{{align*}}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.
Solution: If we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain
$$6y-9x=-\\frac{{3}}{{2}}a.$$Since we also know that $6y-9x=b$, we have
$$-\\frac{{3}}{{2}}a=b\\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$$
Final Answer: The final answer is $-\\frac{{2}}{{3}}$. I hope it is correct.
JSON Output: {{"answer": "-\\frac{{2}}{{3}}"}}

---

Now solve this problem:

Problem:
{problem}

* STRICT REQUIREMENTS *
1. Solve the problem completely
2. Provide ONLY the final answer
3. Output must be valid JSON in this exact format: {{"answer": "<your_final_answer>"}}
4. Do NOT include any explanation, reasoning, or additional text
5. Do NOT use markdown formatting or code blocks
6. The answer can be in LaTeX format and will be normalized later
7. You may use \\boxed{{}} or not - both are acceptable

Output format (mandatory):
{{"answer": "<your_final_answer>"}}"""


# ============================================================================
# UNSTRUCTURED FINAL ANSWER PROMPTS
# ============================================================================

UNSTRUCTURED_FINAL_ANSWER_SYSTEM_PROMPT = """You are an expert mathematical problem solver. Your task is to provide the final answer to a math problem in the exact format used in mathematical competitions and textbooks.

The answer must follow this specific format:
Final Answer: The final answer is <answer>. I hope it is correct.

Where <answer> is the mathematical answer in its natural form. You can include:
- Plain numbers: 42, 3.14159, -7
- LaTeX fractions: \\frac{{2}}{{3}}, -\\frac{{2}}{{3}}
- LaTeX expressions: \\sqrt{{2}}, 2\\pi, e^{{\\pi i}}
- Sets or intervals: [2,5), \\{{1,2,3\\}}
- With or without \\boxed{{}}: both \\boxed{{42}} and 42 are acceptable

The answer will be automatically normalized later, so focus on mathematical correctness.

Here are examples of the expected format:

Example 1:
Problem: Find the domain of the expression $\\frac{{\\sqrt{{x-2}}}}{{\\sqrt{{5-x}}}}$.
Solution: The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{{[2,5)}}$.
Final Answer: The final answer is $\\boxed{{[2,5)}}$. I hope it is correct.

Example 2:
Problem: If $\\det \\mathbf{{A}} = 2$ and $\\det \\mathbf{{B}} = 12,$ then find $\\det (\\mathbf{{A}} \\mathbf{{B}}).$
Solution: We have that $\\det (\\mathbf{{A}} \\mathbf{{B}}) = (\\det \\mathbf{{A}})(\\det \\mathbf{{B}}) = (2)(12) = \\boxed{{24}}$.
Final Answer: The final answer is $\\boxed{{24}}$. I hope it is correct.

Example 3:
Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution: If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:
\\begin{{align*}}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{{16}}
\\end{{align*}}
Final Answer: The final answer is $\\boxed{{16}}$. I hope it is correct.

Example 4:
Problem: If the system of equations
\\begin{{align*}}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.
Solution: If we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain
$$6y-9x=-\\frac{{3}}{{2}}a.$$Since we also know that $6y-9x=b$, we have
$$-\\frac{{3}}{{2}}a=b\\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$$
Final Answer: The final answer is $\\boxed{{-\\frac{{2}}{{3}}}}$. I hope it is correct.

Do NOT include any explanation, reasoning, or additional text before or after this format."""


UNSTRUCTURED_FINAL_ANSWER_USER_PROMPT = """Based on the following problem, plan, and reasoning, provide the final answer in the required format.

Here are examples of the expected output format:

Example 1:
Final Answer: The final answer is $\\boxed{{[2,5)}}$. I hope it is correct.

Example 2:
Final Answer: The final answer is $\\boxed{{24}}$. I hope it is correct.

Example 3:
Final Answer: The final answer is $\\boxed{{16}}$. I hope it is correct.

Example 4:
Final Answer: The final answer is $\\boxed{{-\\frac{{2}}{{3}}}}$. I hope it is correct.

---

PROBLEM:
{problem}

PLAN:
{plan}

REASONING:
{full_reasoning}

---

Provide ONLY the final answer in this exact format:
Final Answer: The final answer is <answer>. I hope it is correct.

Remember:
- The answer will be automatically normalized, so focus on correctness
- You can use LaTeX notation (\\frac{{}}{{}}, \\sqrt{{}}, \\pi, etc.)
- You may use \\boxed{{}} to highlight the answer, but it's optional
- No explanation or additional text allowed
- Just provide the answer in the exact format shown in the examples"""