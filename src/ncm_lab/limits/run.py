"""Run limits experiments one at a time (a 6 GB GPU holds one model comfortably).

    python -m ncm_lab.limits.run --list
    python -m ncm_lab.limits.run batch-sweep output-length
    python -m ncm_lab.limits.run all              # ~1-2 h on an RTX 3050, then the report
    python -m ncm_lab.limits.run gpu-only         # everything that does not need Ollama
"""

import argparse
import importlib

# name -> (module, needs Ollama, default n, what it answers)
EXPERIMENTS = {
    "faithfulness": ("faithfulness", False, 500,
                     "how often outputs invent numbers/names or repeat (CPU, Phase 4 outputs)"),
    "batch-sweep": ("batch_sweep", False, 128,
                    "batch size where AR overtakes NCM; largest batch that fits in memory"),
    "output-length": ("output_length", False, 32,
                      "output length at which NCM's fixed step count beats AR decoding"),
    "input-length": ("input_length", False, 100,
                     "where NCM's fixed input window breaks translation quality"),
    "robustness": ("robustness", False, 200, "quality under typos, case and punctuation noise"),
    "sudoku-stress": ("sudoku_stress", True, 100,
                      "NCM vs exact solver vs LLM on unsolvable / sparse / shuffled sudoku"),
    "llm-vs-ncm": ("llm_vs_ncm", True, 100,
                   "150-260M NCM specialist vs 3B general LLM, same tasks"),
    "agent-reliability": ("agent_reliability", True, 10,
                          "how often the LLM agent uses the NCM tools correctly (sampled runs)"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", help="names, 'all' or 'gpu-only'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n", type=int, help="examples (or repeats) per condition")
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    if args.list or not args.experiments:
        for name, (_, llm, n, what) in EXPERIMENTS.items():
            print(f"{name:18s} n={n:<4d} {'[Ollama] ' if llm else '         '}{what}")
        return
    names = args.experiments
    if names == ["all"]:
        names = list(EXPERIMENTS)
    elif names == ["gpu-only"]:
        names = [k for k, v in EXPERIMENTS.items() if not v[1]]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiment(s): {unknown}; see --list")

    for name in names:
        module, _, n, what = EXPERIMENTS[name]
        print(f"===== {name}: {what}", flush=True)
        importlib.import_module(f"ncm_lab.limits.{module}").main(n=args.n or n,
                                                                 precision=args.precision)
    if not args.no_report:
        from .report import main as report

        report()


if __name__ == "__main__":
    main()
