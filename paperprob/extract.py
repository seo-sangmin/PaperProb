from __future__ import annotations

import dataclasses
import json
import pathlib

import anthropic
import pdfplumber


@dataclasses.dataclass
class Claim:
    claim: str
    reasons: list[str]
    pr_evidence: float | None = None
    pr_claim_given_evidence: float | None = None
    pr_claim_given_not_evidence: float | None = None
    pr_evidence_rationale: str | None = None
    pr_claim_given_evidence_rationale: str | None = None
    pr_claim_given_not_evidence_rationale: str | None = None


def read_paper(path: str) -> str:
    """Read a paper from PDF or text file and return its text content."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".pdf":
        text_parts: list[str] = []
        with pdfplumber.open(p) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        text = "\n\n".join(text_parts)
    else:
        text = p.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"No text content extracted from: {path}")

    # Truncate very long papers with a warning
    max_chars = 500_000  # ~125k tokens
    if len(text) > max_chars:
        text = text[:max_chars]
        text += "\n\n[TRUNCATED — paper exceeds maximum length]"

    return text


EXTRACTION_SYSTEM_PROMPT = """\
You are a philosophy-of-science analyst. Your task is to identify the main \
claim(s) and their supporting reasons/evidence from a research paper.

Instructions:
- Identify 1-3 main claims — the central theses or hypotheses the paper argues for.
- Each claim must be a single, falsifiable proposition stated in one clear sentence.
- For each claim, list the supporting reasons/evidence the paper offers. Each reason \
should be independently statable (e.g., an empirical finding, a logical argument, \
or a cited result).
- Be precise: use the paper's own language where possible.

Return ONLY valid JSON matching this schema (no other text):

```json
{
  "claims": [
    {
      "claim": "The main thesis stated as a single sentence.",
      "reasons": [
        "First supporting reason or piece of evidence.",
        "Second supporting reason or piece of evidence."
      ]
    }
  ]
}
```

Example output for a hypothetical paper on exercise and cognition:

```json
{
  "claims": [
    {
      "claim": "Regular aerobic exercise improves working memory in adults over 60.",
      "reasons": [
        "Randomized controlled trial (N=200) showed a 15% improvement in n-back task scores after 12 weeks of aerobic exercise vs. control.",
        "fMRI data revealed increased activation in the dorsolateral prefrontal cortex post-intervention.",
        "Effect persisted at 6-month follow-up with no significant decay."
      ]
    }
  ]
}
```\
"""


def extract_claims(
    paper_text: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
) -> list[Claim]:
    """Extract claims and supporting reasons from paper text using Claude."""
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Analyze this research paper and extract its main claims and supporting reasons:\n\n{paper_text}",
            }
        ],
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

    claims = []
    for item in data["claims"]:
        claims.append(Claim(claim=item["claim"], reasons=item["reasons"]))

    if not claims:
        raise ValueError("No claims extracted from the paper.")

    return claims
