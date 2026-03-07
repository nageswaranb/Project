"""
main.py
────────────────────────────────────────────────────────────────
SwimCoach RAG System — Main Entry Point

Usage:
  # 1. Seed the knowledge base with sample coaching data
  python main.py seed

  # 2. Ingest a new coaching video
  python main.py ingest --video path/to/video.mp4 --video-id V-0999 \
                        --tags butterfly "dolphin kick" --difficulty advanced

  # 3. Run the primary sample query from the assignment
  python main.py query

  # 4. Run a custom query
  python main.py query --q "Which dolphin kick videos improved male butterfly swimmers aged 18-22?"

  # 5. Interactive REPL
  python main.py repl
────────────────────────────────────────────────────────────────
"""

import sys
import argparse
from rich.console import Console
from rich.panel import Panel

console = Console()

SAMPLE_QUERIES = [
    # Primary query from assignment §3.1
    (
        "Am I missing any must-watch videos for this athlete profile? "
        "Athlete: 19M, 200m Butterfly, underwater dolphin kick weakness, "
        "6 weeks to State Championships."
    ),
    # Additional queries from §3.2
    "Which dolphin kick videos produced measurable improvement for male butterfly swimmers aged 18-22?",
    "What is the optimal video sequence for a butterfly swimmer 5-6 weeks out from a championship?",
    "Which videos are typically introduced too early and later revisited for this profile?",
]


def cmd_seed(_args):
    from ingestion.pipeline import seed_knowledge_base
    seed_knowledge_base()


def cmd_ingest(args):
    from ingestion.pipeline import ingest_video
    if not args.video:
        console.print("[red]Error:[/] --video path required for ingest command.")
        sys.exit(1)
    ingest_video(args.video, args.video_id, args.tags, args.difficulty)


def cmd_query(args):
    from query.pipeline import run_rag_query
    q = args.q or SAMPLE_QUERIES[0]
    run_rag_query(q, stream=not args.no_stream)


def cmd_demo(_args):
    """Run all 4 sample queries from the assignment document."""
    from query.pipeline import run_rag_query
    console.print(Panel(
        "Running all sample queries from Assignment §3",
        title="SwimCoach RAG · Demo Mode",
        style="magenta",
    ))
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        console.rule(f"[bold]Query {i}/{len(SAMPLE_QUERIES)}[/]")
        run_rag_query(q, stream=False)
        input("\nPress Enter for next query …\n")


def cmd_repl(_args):
    """Interactive coaching REPL."""
    from query.pipeline import run_rag_query
    console.print(Panel(
        "SwimCoachGPT Interactive Mode\nType 'exit' to quit.",
        title="SwimCoach RAG · REPL",
        style="cyan",
    ))
    while True:
        try:
            q = input("\n[Coach] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            run_rag_query(q, stream=True)
    console.print("\n[dim]Session ended.[/]")


def main():
    parser = argparse.ArgumentParser(
        description="SwimCoach RAG System — Azure OpenAI + Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # seed
    sub.add_parser("seed", help="Seed KB with sample coaching data")

    # ingest
    ing = sub.add_parser("ingest", help="Ingest a coaching video")
    ing.add_argument("--video",      required=True, help="Path to video file")
    ing.add_argument("--video-id",   default="V-0001")
    ing.add_argument("--tags",       nargs="*", default=[])
    ing.add_argument("--difficulty", default="intermediate",
                     choices=["beginner", "intermediate", "advanced"])

    # query
    qry = sub.add_parser("query", help="Run a RAG query")
    qry.add_argument("--q",         default=None, help="Query string (default: assignment §3.1 query)")
    qry.add_argument("--no-stream", action="store_true")

    # demo
    sub.add_parser("demo", help="Run all 4 sample queries from the assignment")

    # repl
    sub.add_parser("repl", help="Interactive coaching session")

    args = parser.parse_args()

    dispatch = {
        "seed":   cmd_seed,
        "ingest": cmd_ingest,
        "query":  cmd_query,
        "demo":   cmd_demo,
        "repl":   cmd_repl,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
