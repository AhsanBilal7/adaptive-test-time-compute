# src/templates.py
MATH_FEWSHOT = """You are a careful mathematician. Solve the problem step by step.

STRICT OUTPUT FORMAT (the grader parses this exactly):
Reasoning: <your steps across multiple lines>
Answer: <final answer on ONE line only>

Rules:
- Output exactly one 'Reasoning:' block followed by exactly one 'Answer:' line.
- Do NOT add any text before 'Reasoning:' or after the 'Answer:' line.
- Do NOT use code fences, bullets, or extra sections.
- The 'Answer:' line must contain only the final value (include units only if essential).

Example 1
Question: If you have 3 apples and buy 4 more, how many apples do you have?
Reasoning: Start with 3. Add 4. 3 + 4 = 7.
Answer: 7

Example 2
Question: A train travels 60 km in 1.5 hours. What is its speed in km/h?
Reasoning: Speed = distance / time = 60 / 1.5 = 40.
Answer: 40 km/h

Example 3
Question: Janet has 5 books. She buys 3 more books and then gives 2 books to her friend. How many books does she have now?
Reasoning: Start with 5 books. Buy 3 more: 5 + 3 = 8. Give away 2: 8 - 2 = 6.
Answer: 6
"""


QA_FEWSHOT = """You are a precise reasoner. Provide concise, grounded steps, then the final answer.

STRICT OUTPUT FORMAT (parsed exactly):
Reasoning: <your steps across multiple lines>
Answer: <final short answer on ONE line only>

Rules:
- Output exactly one 'Reasoning:' block followed by exactly one 'Answer:' line.
- Do NOT add any text before 'Reasoning:' or after the 'Answer:' line.
- Do NOT use code fences, bullets, or extra sections.
- The 'Answer:' line must be a single short word/phrase (e.g., 'yes', 'no', or a brief noun phrase).

Example 1
Question: Do penguins live at the North Pole?
Reasoning: Penguins are native to the Southern Hemisphere, mainly Antarctica. The North Pole is in the Arctic, which has no penguins.
Answer: no

Example 2
Question: Is water wet?
Reasoning: "Wet" refers to being covered by liquid water; water itself can be described as making things wet.
Answer: yes

Example 3
Question: Can a person run faster than a cheetah?
Reasoning: Cheetahs can reach speeds up to 70 mph. The fastest human speed recorded is around 28 mph. Cheetahs are much faster.
Answer: no
"""


MATH_ITERATIVE = """You are a careful mathematician solving a problem step by step.

Continue the reasoning below. Provide the NEXT step(s) in your solution.

Rules:
- If you can determine the final answer, output ONLY: Answer: <value>
- Otherwise, provide 1-3 sentences of reasoning for the next logical step
- Do NOT repeat previous reasoning
- Do NOT start from scratch
- Be concise and focused

Question: {question}

Previous reasoning:
{previous_reasoning}

Next step:"""


QA_ITERATIVE = """You are a precise reasoner working through a question step by step.

Continue the reasoning below. Provide the NEXT step(s) toward your conclusion.

Rules:
- If you can determine the final answer, output ONLY: Answer: <value>
- Otherwise, provide 1-3 sentences of reasoning for the next logical step
- Do NOT repeat previous reasoning
- Do NOT start from scratch
- Be concise and focused

Question: {question}

Previous reasoning:
{previous_reasoning}

Next step:"""


MATH_FINAL_ANSWER = """You are a careful mathematician. Based on the reasoning below, provide the final answer.

Question: {question}

Reasoning:
{reasoning}

CRITICAL: Output ONLY the final answer in this exact format with NO additional text, explanations, or comments:
Answer: <value>

Do NOT add anything before or after this line. Just the answer.

Final Answer:"""


QA_FINAL_ANSWER = """You are a precise reasoner. Based on the reasoning below, provide the final answer.

Question: {question}

Reasoning:
{reasoning}

CRITICAL: Output ONLY the final answer in this exact format with NO additional text, explanations, or comments:
Answer: <value>

Do NOT add anything before or after this line. Just the answer.

Final Answer:"""


def build_prompt(question: str, kind: str = "math") -> str:
    head = MATH_FEWSHOT if kind == "math" else QA_FEWSHOT
    return f"""{head}

Question: {question}
Reasoning:"""


def build_iterative_prompt(question: str, previous_reasoning: str, kind: str = "math") -> str:
    """
    Build a prompt for iterative reasoning where we continue from previous steps.
    """
    template = MATH_ITERATIVE if kind == "math" else QA_ITERATIVE
    
    if not previous_reasoning.strip():
        previous_reasoning = "(none yet - this is your first step)"
    
    return template.format(question=question, previous_reasoning=previous_reasoning)


def build_final_answer_prompt(question: str, reasoning: str, kind: str = "math") -> str:
    """
    Build a prompt to extract ONLY the final answer from complete reasoning.
    Forces clean output with no additional comments.
    """
    template = MATH_FINAL_ANSWER if kind == "math" else QA_FINAL_ANSWER
    return template.format(question=question, reasoning=reasoning)