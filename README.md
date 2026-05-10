# paperprob

Estimate the probability of a research paper's claims using [Jeffrey conditionalization](https://plato.stanford.edu/entries/bayes-theorem/#4).

Given a paper, PaperProb:

1. **Extracts** the main claim(s) and supporting reasons/evidence (via Claude)
2. **Estimates** three probabilities per claim (via Claude):
   - **Pr(E)** — probability that all supporting evidence is true
   - **Pr(C | E)** — probability of the claim given the evidence holds
   - **Pr(C | ~E)** — probability of the claim if the evidence fails
3. **Computes** the overall probability of each claim:

```
Pr(C) = Pr(C|E) x Pr(E) + Pr(C|~E) x (1 - Pr(E))
```

## Installation

```bash
pip install -e .
```

Requires Python 3.10+ and an [Anthropic API key](https://console.anthropic.com/):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

```bash
# Analyze a PDF
paperprob paper.pdf

# Analyze a text file
paperprob paper.txt

# Use a different model
paperprob paper.pdf --model claude-opus-4-20250514

# Output as JSON
paperprob paper.pdf --json
```

## Example Output

```
╭──────── PaperProb ────────╮
│ paper.pdf  |  Model: claude-sonnet-4-20250514 │
╰───────────────────────────╯

──────────── Claim 1 of 1 ────────────

Claim: Transformer architectures outperform RNNs on machine translation.

Reasons/Evidence:
  1. BLEU score of 28.4 on EN-DE WMT 2014 (vs. 25.8 for best RNN)
  2. Training time reduced by an order of magnitude
  3. Self-attention captures long-range dependencies more effectively

┌─────────────┬───────┬──────────────────────────────────────────┐
│ Probability │ Value │ Rationale                                │
├─────────────┼───────┼──────────────────────────────────────────┤
│ Pr(E)       │ 0.850 │ Results widely replicated; benchmarks... │
│ Pr(C | E)   │ 0.900 │ Evidence directly demonstrates the...    │
│ Pr(C | ~E)  │ 0.300 │ Some independent evidence from other...  │
└─────────────┴───────┴──────────────────────────────────────────┘

╭─── Jeffrey Conditionalization ───╮
│ Pr(C) = Pr(C|E) x Pr(E) + Pr(C|~E) x (1 - Pr(E))  │
│       = 0.900 x 0.850 + 0.300 x 0.150               │
│       = 0.8100                                        │
╰──────────────────────────────────╯
```

## How It Works

PaperProb treats all supporting reasons as a single conjunction **E** (the evidence). This avoids dubious conditional independence assumptions -- instead, you assess "how likely is it that *all* of this evidence holds up?" as one judgment.

Each claim is evaluated independently. The LLM provides calibrated probability estimates with brief rationales for transparency.

### Jeffrey Conditionalization

Unlike simple Bayesian conditioning (which assumes you learn E with certainty), Jeffrey conditionalization handles uncertain evidence. You don't need to be certain the evidence is true -- you only need a probability Pr(E) reflecting your confidence in it. This makes it well-suited for evaluating research papers, where evidence is rarely beyond doubt.

## Project Structure

```
paperprob/
  __init__.py
  extract.py    # PDF/text reading + LLM claim/reason extraction
  compute.py    # Jeffrey conditionalization + LLM probability estimation
  cli.py        # CLI entry point, orchestration, rich output
```

## License

MIT
