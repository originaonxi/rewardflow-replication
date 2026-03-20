"""
Rollout Agent: generates N diverse solution attempts per problem.
Each attempt = a sequence of reasoning steps.
Some will be correct. Some wrong. Some partially correct.
This diversity is what makes the state graph meaningful.
"""

from __future__ import annotations
import re
import anthropic
from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_rollouts(problem: dict, n_rollouts: int = 6) -> list[dict]:
    """
    Generate N diverse solution attempts.
    Uses temperature > 0 for diversity.
    Returns list of {steps: [...], answer: str, is_correct: bool}
    """
    rollouts = []

    for i in range(n_rollouts):
        # Vary temperature for diversity
        temperature = 0.3 + (i * 0.15)  # 0.3, 0.45, 0.6, 0.75, 0.9, 1.0
        temperature = min(temperature, 1.0)

        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": f"""Solve this step by step:

{problem['problem']}

Format EXACTLY as:
Step 1: [reasoning]
Step 2: [reasoning]
...
ANSWER: [final answer as a number or short phrase]"""
                }]
            )

            text = msg.content[0].text
            steps = _parse_steps(text)
            answer = _parse_answer(text)
            is_correct = _check_answer(answer, problem["answer_value"])

            rollouts.append({
                "rollout_id": i,
                "steps": steps,
                "answer": answer,
                "is_correct": is_correct,
                "temperature": temperature,
            })

        except Exception as e:
            rollouts.append({
                "rollout_id": i,
                "steps": [],
                "answer": "",
                "is_correct": False,
                "temperature": temperature,
                "error": str(e),
            })

    return rollouts

def _parse_steps(text: str) -> list[str]:
    """Extract numbered steps from response."""
    steps = []
    pattern = r'Step\s+\d+:\s*(.+?)(?=Step\s+\d+:|ANSWER:|$)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    for m in matches:
        step = re.sub(r'\s+', ' ', m).strip()
        if step:
            steps.append(step)
    if not steps:
        lines = [l.strip() for l in text.split('\n')
                 if l.strip() and not l.startswith('ANSWER')]
        steps = lines[:6]
    return steps

def _parse_answer(text: str) -> str:
    """Extract ANSWER: value."""
    match = re.search(r'ANSWER:\s*(.+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Take last number found
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else ""

def _check_answer(response_answer: str, correct_value: float) -> bool:
    """Check if answer is correct within tolerance."""
    try:
        nums = re.findall(r'-?\d+\.?\d*', str(response_answer))
        if nums:
            val = float(nums[0])
            tolerance = max(abs(correct_value) * 0.02, 0.1)
            return abs(val - correct_value) <= tolerance
    except Exception:
        pass
    return False
