from __future__ import annotations

import argparse
import json
import sys

import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from paperprob.compute import estimate_probabilities, jeffrey_conditionalization
from paperprob.extract import Claim, extract_claims, read_paper

console = Console()


def display_claim(index: int, total: int, claim: Claim) -> None:
    """Display a single claim with its reasons, probabilities, and Pr(claim)."""
    # Header
    console.rule(f"[bold]Claim {index} of {total}[/bold]")
    console.print()

    # Claim text
    console.print(f"[bold cyan]Claim:[/bold cyan] {claim.claim}")
    console.print()

    # Reasons
    console.print("[bold cyan]Reasons/Evidence:[/bold cyan]")
    for i, reason in enumerate(claim.reasons, 1):
        console.print(f"  {i}. {reason}")
    console.print()

    # Probabilities table
    table = Table(show_header=True, header_style="bold", expand=False)
    table.add_column("Probability", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Rationale", max_width=60)

    table.add_row(
        "Pr(E)",
        f"{claim.pr_evidence:.3f}",
        claim.pr_evidence_rationale or "",
    )
    table.add_row(
        "Pr(C | E)",
        f"{claim.pr_claim_given_evidence:.3f}",
        claim.pr_claim_given_evidence_rationale or "",
    )
    table.add_row(
        "Pr(C | ¬E)",
        f"{claim.pr_claim_given_not_evidence:.3f}",
        claim.pr_claim_given_not_evidence_rationale or "",
    )
    console.print(table)
    console.print()

    # Compute Pr(claim)
    pr_claim = jeffrey_conditionalization(
        claim.pr_evidence,
        claim.pr_claim_given_evidence,
        claim.pr_claim_given_not_evidence,
    )

    # Result panel
    formula = (
        f"Pr(C) = Pr(C|E) × Pr(E) + Pr(C|¬E) × (1 − Pr(E))\n"
        f"      = {claim.pr_claim_given_evidence:.3f} × {claim.pr_evidence:.3f}"
        f" + {claim.pr_claim_given_not_evidence:.3f} × {1.0 - claim.pr_evidence:.3f}\n"
        f"      = [bold green]{pr_claim:.4f}[/bold green]"
    )
    console.print(Panel(formula, title="[bold]Jeffrey Conditionalization[/bold]", expand=False))
    console.print()

    return pr_claim


def claims_to_json(claims: list[Claim], pr_claims: list[float]) -> str:
    """Serialize results to JSON."""
    results = []
    for claim, pr_c in zip(claims, pr_claims):
        results.append({
            "claim": claim.claim,
            "reasons": claim.reasons,
            "pr_evidence": claim.pr_evidence,
            "pr_evidence_rationale": claim.pr_evidence_rationale,
            "pr_claim_given_evidence": claim.pr_claim_given_evidence,
            "pr_claim_given_evidence_rationale": claim.pr_claim_given_evidence_rationale,
            "pr_claim_given_not_evidence": claim.pr_claim_given_not_evidence,
            "pr_claim_given_not_evidence_rationale": claim.pr_claim_given_not_evidence_rationale,
            "pr_claim": pr_c,
        })
    return json.dumps(results, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="paperprob",
        description="Compute the probability of research paper claims using Jeffrey conditionalization.",
    )
    parser.add_argument("paper", help="Path to a PDF or text file containing the research paper")
    parser.add_argument(
        "-m", "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    # Validate API key
    try:
        client = anthropic.Anthropic()
    except anthropic.AuthenticationError:
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    # Read paper
    if not args.output_json:
        console.print(f"[dim]Reading paper:[/dim] {args.paper}")
    try:
        paper_text = read_paper(args.paper)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    # Extract claims
    if not args.output_json:
        console.print(f"[dim]Extracting claims using {args.model}...[/dim]")
    try:
        claims = extract_claims(paper_text, client, model=args.model)
    except Exception as e:
        console.print(f"[bold red]Error extracting claims:[/bold red] {e}")
        sys.exit(1)

    if not args.output_json:
        console.print(f"[dim]Found {len(claims)} claim(s). Estimating probabilities...[/dim]")
        console.print()

    # Estimate probabilities for each claim
    for claim in claims:
        try:
            estimate_probabilities(claim, paper_text, client, model=args.model)
        except Exception as e:
            console.print(f"[bold red]Error estimating probabilities:[/bold red] {e}")
            sys.exit(1)

    # Display results
    pr_claims = []
    if args.output_json:
        for claim in claims:
            pr_c = jeffrey_conditionalization(
                claim.pr_evidence,
                claim.pr_claim_given_evidence,
                claim.pr_claim_given_not_evidence,
            )
            pr_claims.append(pr_c)
        print(claims_to_json(claims, pr_claims))
    else:
        console.print(Panel(f"[bold]{args.paper}[/bold]  |  Model: {args.model}", title="PaperProb"))
        console.print()
        for i, claim in enumerate(claims, 1):
            pr_c = display_claim(i, len(claims), claim)
            pr_claims.append(pr_c)


if __name__ == "__main__":
    main()
