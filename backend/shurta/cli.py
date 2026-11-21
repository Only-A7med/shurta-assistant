"""The ``python -m shurta`` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from shurta.config import Settings, get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shurta",
        description=(
            "Build and query a local, bilingual knowledge base of public service information."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    ingest = commands.add_parser(
        "ingest",
        help="collect the selected services from the official site",
        description=(
            "Collects the selected public services from the official site in English and "
            "Arabic, writes data/processed/services.json and a collection report. "
            "Responses are cached under data/raw, so re-running issues no requests."
        ),
    )
    ingest.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the cached responses and read the site again",
    )

    commands.add_parser(
        "index",
        help="build the search index from the collected corpus",
    )

    evaluate = commands.add_parser("eval", help="measure retrieval or answer quality")
    evaluate.add_argument(
        "target",
        choices=["prepare", "retrieval", "answers"],
        help=(
            "prepare: derive the key facts an answer must contain from the local corpus; "
            "retrieval: score retrieval; answers: score generated answers"
        ),
    )
    evaluate.add_argument(
        "--model",
        action="append",
        default=None,
        help="generation model to evaluate; repeat to compare several (answers only)",
    )
    evaluate.add_argument(
        "--force",
        action="store_true",
        help="re-run models that already have a scorecard (answers only)",
    )
    evaluate.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N questions of each group (answers only)",
    )
    evaluate.add_argument(
        "--subset",
        default=None,
        help=(
            "run only the questions named in a subset file: a path, or a name "
            "under eval/subsets (answers only)"
        ),
    )
    evaluate.add_argument(
        "--rescore",
        action="store_true",
        help=(
            "re-grade the answers already stored in each scorecard and rewrite the "
            "reports, without calling a model (answers only)"
        ),
    )
    evaluate.add_argument(
        "--speed-from-subset",
        default=None,
        metavar="NAME",
        help=(
            "take the speed component from each model's scorecard for this subset, "
            "measured with nothing else running, instead of from the run's own median "
            "(answers only)"
        ),
    )
    evaluate.add_argument(
        "--note",
        action="append",
        default=None,
        metavar="MODEL=TEXT",
        help="a note to print beside one model in the report (answers only)",
    )

    commands.add_parser(
        "doctor",
        help="check that Ollama, the models, the corpus and the index are in place",
    )

    serve = commands.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    return parser


def _ingest(settings: Settings, *, refresh: bool) -> int:
    from shurta.collect import dubai_police
    from shurta.collect.http import CollectionError
    from shurta.collect.ingest import collect

    if refresh:
        for cached in settings.raw_dir.glob("*.json"):
            cached.unlink()
        print(f"Cleared the response cache in {settings.raw_dir}")

    print(f"Reading {dubai_police.SITE} with at least 2 seconds between requests.")
    try:
        result = collect(settings.raw_dir)
    except CollectionError as error:
        print(f"Collection failed: {error}", file=sys.stderr)
        return 1

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.services_file.write_text(result.corpus.model_dump_json(indent=1), encoding="utf-8")
    report_path = settings.processed_dir / "collection-report.md"
    report_path.write_text(result.report, encoding="utf-8")

    print(f"Collected {len(result.corpus.services)} services -> {settings.services_file}")
    print(f"Validation report -> {report_path}")
    if result.missing_slugs:
        print(
            f"{len(result.missing_slugs)} selected services were not found: "
            f"{', '.join(result.missing_slugs)}",
            file=sys.stderr,
        )
    return 0


def _index(settings: Settings) -> int:
    from shurta.index.build import build_index

    if not settings.services_file.exists():
        print(
            f"No corpus at {settings.services_file}. Run `python -m shurta ingest` first.",
            file=sys.stderr,
        )
        return 1
    summary = build_index(settings)
    print(summary)
    return 0


def _doctor(settings: Settings) -> int:
    from shurta.doctor import run_doctor

    return run_doctor(settings)


def _notes(pairs: Sequence[str] | None) -> dict[str, str]:
    """``MODEL=TEXT`` arguments as a mapping."""
    notes: dict[str, str] = {}
    for pair in pairs or ():
        model, _, text = pair.partition("=")
        if text:
            notes[model.strip()] = text.strip()
    return notes


def _evaluate(settings: Settings, arguments: argparse.Namespace) -> int:
    target = arguments.target
    if target == "prepare":
        from shurta.evaluation.prepare import main as prepare_main

        return prepare_main(settings)

    if target == "retrieval":
        from shurta.evaluation.retrieval import main as retrieval_main

        return retrieval_main(settings)

    from shurta.evaluation.answers import main as answers_main

    return answers_main(
        settings,
        models=list(arguments.model) if arguments.model else None,
        force=arguments.force,
        limit=arguments.limit,
        subset=arguments.subset,
        notes=_notes(arguments.note),
        rescore_only=arguments.rescore,
        speed_from_subset=arguments.speed_from_subset,
    )


def _serve(settings: Settings, host: str, port: int, reload: bool) -> int:
    import uvicorn

    uvicorn.run(
        "shurta.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    settings = get_settings()

    if arguments.command == "ingest":
        return _ingest(settings, refresh=arguments.refresh)
    if arguments.command == "index":
        return _index(settings)
    if arguments.command == "eval":
        return _evaluate(settings, arguments)
    if arguments.command == "doctor":
        return _doctor(settings)
    if arguments.command == "serve":
        return _serve(settings, arguments.host, arguments.port, arguments.reload)

    parser.error(f"unknown command {arguments.command!r}")


__all__ = ["build_parser", "main"]
