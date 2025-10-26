# src/templates.py
MATH_FEWSHOT = """You are a careful mathematician. Solve the problem step by step, showing clear reasoning.
Format strictly:
Reasoning: <your steps across multiple lines>
Answer: <final numeric answer>

Example 1
Question: If you have 3 apples and buy 4 more, how many apples do you have?
Reasoning: Start with 3. Add 4. 3 + 4 = 7.
Answer: 7

Example 2
Question: A train travels 60 km in 1.5 hours. What is its speed in km/h?
Reasoning: Speed = distance / time = 60 / 1.5 = 40 km/h.
Answer: 40

Example 3
Question: Janet has 5 books. She buys 3 more books and then gives 2 books to her friend. How many books does she have now?
Reasoning: Start with 5 books. Buy 3 more: 5 + 3 = 8. Give away 2: 8 - 2 = 6.
Answer: 6
"""

QA_FEWSHOT = """You are a precise reasoner. Provide concise, grounded steps, then the final answer.
Format strictly:
Reasoning: <your steps across multiple lines>
Answer: <final short answer yes/no or short phrase>

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


def build_prompt(question: str, kind: str = "math") -> str:
    head = MATH_FEWSHOT if kind == "math" else QA_FEWSHOT
    return f"""{head}

Question: {question}
Reasoning:"""