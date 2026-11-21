import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gateway.ask import LocalModel, is_grounded, extract_numbers, extract_candidate_answer, N_SAMPLES

prompt = "A bank account starts at $50. A withdrawal of $80 is made, then a deposit of $25, then another withdrawal of $10. What is the final balance?"

print("Prompt numbers:", extract_numbers(prompt))
print()

local = LocalModel()
for i in range(N_SAMPLES):
    r = local(prompt)
    text = r["text"]
    grounded = is_grounded(prompt, text)
    candidate = extract_candidate_answer(text)
    reply_nums = extract_numbers(text)
    print(f"--- Sample {i+1} ---")
    print(f"grounded = {grounded}")
    print(f"candidate_answer = {candidate!r}")
    print(f"numbers_in_reply = {reply_nums}")
    print(f"full_text = {text}")
    print()