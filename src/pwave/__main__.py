"""Command line: ``python -m pwave {fetch,eval,demo}``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CACHE = Path("cache")
DEFAULT_OUT = Path("docs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pwave", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, helptext in (
        ("fetch", "download catalogue picks and waveforms into the cache"),
        ("eval", "evaluate pickers on the cached data and write results + figures"),
        ("demo", "fetch if the cache is empty, then evaluate"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        s.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p


def cmd_eval(cache: Path, out: Path) -> dict:
    from . import data, evaluate, figures

    traces, noise = data.load(cache)
    out.mkdir(exist_ok=True)
    res = evaluate.run(traces, noise, out / "results.json")
    figures.make_all(traces, noise, res, out)
    return res


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "fetch":
        from . import fetch

        fetch.run(args.cache)
        return 0
    if args.cmd == "demo" and not (args.cache / "selected.json").exists():
        from . import fetch

        fetch.run(args.cache)
    res = cmd_eval(args.cache, args.out)
    brief = {k: v for k, v in res.items() if not k.startswith("_")}
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
