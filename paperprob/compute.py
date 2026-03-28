from __future__ import annotations

import json

import anthropic

from paperprob.extract import Claim


def jeffrey_conditionalization(
    pr_evidence: float,
    pr_claim_given_evidence: float,
    pr_claim_given_not_evidence: float,
) -> float:
    """Compute Pr(claim) using Jeffrey conditionalization.

    Pr(C) = Pr(C|E) * Pr(E) + Pr(C|~E) * (1 - Pr(E))
    """
    return (
        pr_claim_given_evidence * pr_evidence
        + pr_claim_given_not_evidence * (1.0 - pr_evidence)
    )


PROBABILITY_SYSTEM_PROMPT = """\
You are a calibrated probabilistic reasoner. Given a research paper's claim and \
its supporting reasons/evidence, estimate three probabilities.

Definitions:
- E = the conjunction of all listed reasons/evidence being true and sound.
- C = the claim being true.

You must estimate:
1. Pr(E): How likely is it that ALL the supporting reasons/evidence are true? \
Consider replication rates, methodological quality, and the strength of the evidence.
2. Pr(C | E): If all the evidence IS true and sound, how likely is the claim? \
Consider whether the evidence logically entails the claim or merely supports it.
3. Pr(C | ~E): If the evidence is NOT true (flawed, unreplicable, or unsound), \
how likely is the claim anyway? Consider base rates and alternative evidence.

Guidelines:
- Be well-calibrated. Avoid overconfidence.
- Consider base rates for the field.
- All values must be between 0 and 1 (exclusive — never use exactly 0 or 1).
- Provide a brief rationale (1-2 sentences) for each estimate.

Return ONLY valid JSON matching this schema (no other text):

```json
{
  "pr_evidence": 0.75,
  "pr_evidence_rationale": "Brief rationale...",
  "pr_claim_given_evidence": 0.85,
  "pr_claim_given_evidence_rationale": "Brief rationale...",
  "pr_claim_given_not_evidence": 0.20,
  "pr_claim_given_not_evidence_rationale": "Brief rationale..."
}
```\
"""


def estimate_probabilities(
    claim: Claim,
    paper_text: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
) -> Claim:
    """Use Claude to estimate the three probabilities for a claim."""
    reasons_text = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(claim.reasons))

    user_msg = (
        f"Paper context (may be truncated):\n{paper_text[:50_000]}\n\n"
        f"---\n\n"
        f"Claim: {claim.claim}\n\n"
        f"Supporting reasons/evidence:\n{reasons_text}\n\n"
        f"Estimate Pr(E), Pr(C|E), and Pr(C|~E) for this claim."
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=PROBABILITY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text

    # Strip markdown code fences if present
    if "```json" in raw:
        raw = raw.split("```json", 1)[1]
        raw = raw.split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1]
        raw = raw.split("```", 1)[0]

    data = json.loads(raw.strip())

    claim.pr_evidence = float(data["pr_evidence"])
    claim.pr_claim_given_evidence = float(data["pr_claim_given_evidence"])
    claim.pr_claim_given_not_evidence = float(data["pr_claim_given_not_evidence"])
    claim.pr_evidence_rationale = data.get("pr_evidence_rationale")
    claim.pr_claim_given_evidence_rationale = data.get("pr_claim_given_evidence_rationale")
    claim.pr_claim_given_not_evidence_rationale = data.get("pr_claim_given_not_evidence_rationale")

    return claim
