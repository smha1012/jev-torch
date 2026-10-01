"""Inference examples for a trained JEV model.

    python examples/inference.py --model smha1012/jev-9b      # from the Hugging Face Hub
    python examples/inference.py --model runs/jev-9b/best     # from a local checkpoint
"""

import argparse

from jev import JEVExample, JEVPredictor, load_jsonl

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="Hub repo id or local checkpoint directory")
p.add_argument("--device", default="auto")
args = p.parse_args()

jev = JEVPredictor(args.model, device=args.device)


def show(title, probs):
    print(f"\n## {title}")
    for option, prob in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {prob:6.3f}  {'#' * round(prob * 30):30s}  {option}")


# 1) Three question kinds -------------------------------------------------------------------

# noul: yes/no. Options are always [false-like, true-like].
show("noul: pause the rollout?", jev.predict(
    kind="noul",
    state="The deploy passed all CI checks, but the canary shows p99 latency up 40% versus baseline.",
    question="Should the rollout be paused?",
    options=["false", "true"],
))

# choice: pick one of 2-16 options; any wording works.
show("choice: root cause", jev.predict(
    kind="choice",
    state="The nightly feed is byte-identical to yesterday; additionally arrived 3 hours late. "
          "Context: shortly after a software update. Reported by the automated check.",
    question="Root cause for this scenario.",
    options=["producer_change", "schema_drift", "infrastructure", "expected_variation"],
))

# score: ordered levels 0-5. The expected value is often more useful than the argmax.
severity = jev.predict(
    kind="score",
    state="Customers report the export button silently fails for files over 2 GB; a CLI workaround exists.",
    question="Rate severity for this scenario on a 0-5 scale.",
    options=["0", "1", "2", "3", "4", "5"],
)
show("score: severity 0-5", severity)
print(f"  expected severity = {sum(int(k) * v for k, v in severity.items()):.2f}")


# 2) Batch scoring from a JSONL file ----------------------------------------------------------

examples = load_jsonl("examples/example.jsonl")
print("\n## batch: examples/example.jsonl")
for ex, probs in zip(examples, jev.predict_batch(examples)):
    best = max(probs, key=probs.get)
    print(f"  [{ex.kind:6s}] {ex.question[:50]:50s} -> {best} ({probs[best]:.2f})")


# 3) Using calibration: decide when confident, escalate otherwise ------------------------------
# Probabilities are temperature-calibrated, so a threshold is a real accuracy/coverage dial:
# at 0.85 you act on the cases where the model is right ~85%+ of the time and hand off the rest.

THRESHOLD = 0.85
tickets = [
    "Login page returns HTTP 500 for every user since the 14:00 deploy.",
    "A user asks whether the dark-mode toggle could be moved to the top bar.",
    "Invoices for one enterprise customer show the wrong VAT rate.",
]
teams = ["infrastructure", "frontend", "billing", "product_feedback"]
batch = [JEVExample(kind="choice", state=t, question="Which team should handle this ticket?",
                    options=teams, label=0) for t in tickets]

print(f"\n## routing with a {THRESHOLD} confidence threshold")
for ticket, probs in zip(tickets, jev.predict_batch(batch)):
    team = max(probs, key=probs.get)
    action = f"route to {team}" if probs[team] >= THRESHOLD else "escalate to a human / larger model"
    print(f"  {probs[team]:.2f}  {action:40s}  <- {ticket[:60]}")
