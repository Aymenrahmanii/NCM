"""Limits suite on CPU with stand-in models: every experiment runs end to end, no downloads."""

import torch

from ncm_lab.api.registry import FakeStore, FakeSudoku, FakeTextModel
from ncm_lab.limits import (agent_reliability, batch_sweep, faithfulness, input_length,
                            llm_vs_ncm, output_length, report, robustness, sudoku_stress)
from ncm_lab.limits.common import read, write
from ncm_lab.rag.llm import LLMReply

WIKI_PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
WIKI_SOLUTION = ("534678912672195348198342567859761423426853791713924856961537284287419635"
                 "345286179")


class FakeSeqModel:
    """prepare -> run -> finish like the real wrappers; keeps the first `keep` words."""

    prefix_len = 8

    def __init__(self, keep: int = 6, oom_above: int | None = None):
        self.keep, self.oom_above = keep, oom_above
        self.src_tok = lambda t: {"input_ids": t.split()}

    def prepare(self, items):
        return list(items)

    def run(self, x):
        if self.oom_above and len(x) > self.oom_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        return torch.zeros(len(x), 12, dtype=torch.long)

    def generate(self, texts):
        return [" ".join(str(t).split()[: self.keep]) for t in texts]


# ---- 1. input length -------------------------------------------------------------------
def test_join_groups_keeps_sources_and_references_aligned():
    src, refs = [f"s{i}" for i in range(7)], [f"r{i}" for i in range(7)]
    ins, outs = input_length.join_groups(src, refs, 3, n=5)
    assert ins == ["s0 s1 s2", "s3 s4 s5"] and outs == ["r0 r1 r2", "r3 r4 r5"]


def test_input_length_shows_truncation_of_a_fixed_window():
    src = [f"Satz {i} hat genau fünf Wörter" for i in range(40)]
    rows, _ = input_length.evaluate(FakeSeqModel(keep=6), "fake", src, src, groups=(1, 4), n=5)
    one, four = rows
    assert one["share_truncated"] == 0.0 and four["share_truncated"] == 1.0
    assert four["length_ratio"] < one["length_ratio"]
    assert four["bleu"] < one["bleu"]


# ---- 2. batch sweep --------------------------------------------------------------------
def test_batch_sweep_stops_at_out_of_memory():
    rows = batch_sweep.sweep(FakeSeqModel(oom_above=4), "fake", "wmt14", ["a b c"] * 8,
                             batch_sizes=(1, 2, 4, 8, 16), examples=8)
    assert [r["batch_size"] for r in rows] == [1, 2, 4, 8]
    assert [r["oom"] for r in rows] == [False, False, False, True]
    assert all(r["examples_per_s"] > 0 for r in rows[:3])


def test_crossover_finds_first_batch_where_b_is_faster():
    rows = [{"model": m, "batch_size": bs, "examples_per_s": tp, "oom": "False"}
            for m, bs, tp in [("NCM", 1, 4), ("AR", 1, 3), ("NCM", 8, 5), ("AR", 8, 19),
                              ("NCM", 32, 5), ("AR", 32, 43)]]
    assert batch_sweep.crossover(rows, "NCM", "AR") == 8
    assert batch_sweep.crossover(rows, "AR", "NCM") is None  # NCM only wins at batch 1


# ---- 3. output length ------------------------------------------------------------------
def test_breakeven_from_a_linear_ar_cost():
    rows = [{"model": "NCM", "batch_size": 1, "output_tokens": 128, "ms_p50": 250.0}]
    rows += [{"model": "AR", "batch_size": 1, "output_tokens": t, "ms_p50": 20 + 5 * t}
             for t in (16, 32, 64, 128)]
    (b,) = output_length.breakeven(rows, "wmt14")
    assert b["ar_ms_fixed"] == 20.0 and b["ar_ms_per_token"] == 5.0
    assert b["breakeven_tokens"] == 46  # 20 + 5 * 46 = 250


def test_ncm_cost_reads_the_canvas_size():
    (row,) = output_length.ncm_cost(FakeSeqModel(), ["a"], batch_sizes=(1,), repeats=2)
    assert row["output_tokens"] == 12 and row["ms_p50"] >= 0


# ---- 4. faithfulness -------------------------------------------------------------------
def test_faithfulness_flags_invented_numbers_names_and_repeats():
    source = "The council in Cardiff approved a £20m bridge on Tuesday."
    ok = faithfulness.check("Cardiff approved a £20m bridge.", source)
    assert ok == {"invented_numbers": [], "invented_names": [], "repeats": False}
    bad = faithfulness.check("Plans for a £12m bridge in Swansea have been approved.", source)
    assert bad["invented_numbers"] == ["12"] and bad["invented_names"] == ["Swansea"]
    assert faithfulness.check("intention or coup d coup coup?", "")["repeats"]


def test_faithfulness_score_shares():
    row, flagged = faithfulness.score("m", "xsum", ["Paris 3", "fine"], ["Paris 3", "fine"])
    assert row["share_any"] == 0.0 and flagged == []
    row, flagged = faithfulness.score("m", "xsum", ["Rome 4", "fine"], ["Paris 3", "fine"])
    assert row["share_invented_number"] == 0.5 and row["share_invented_name"] == 0.0
    assert len(flagged) == 1


# ---- 5. robustness ---------------------------------------------------------------------
def test_perturbations_are_deterministic_and_change_text():
    text = ["Die Regierung beschloss gestern, die Steuern deutlich zu senken."]
    p = robustness.perturbations()
    assert p["typos_30pct"](text) == p["typos_30pct"](text) != text
    assert p["no_punctuation"](text)[0].endswith("senken")
    assert p["uppercase"](text)[0].isupper()


def test_robustness_reports_drop_vs_clean():
    texts = ["the cat sat on the mat today", "a dog ran in the park"]
    rows = robustness.evaluate(FakeSeqModel(keep=10), "fake", "wmt14", texts, texts)
    by = {r["perturbation"]: r for r in rows}
    assert by["clean"]["drop_vs_clean"] == 0 and by["clean"]["bleu"] == 100.0
    assert by["uppercase"]["drop_vs_clean"] > 0


# ---- 6. sudoku -------------------------------------------------------------------------
def test_exact_solver():
    assert sudoku_stress.solve_exact(WIKI_PUZZLE) == (WIKI_SOLUTION, "solved")
    assert sudoku_stress.solve_exact("55" + WIKI_PUZZLE[2:])[1] == "unsolvable"
    sol, status = sudoku_stress.solve_exact("0" * 81)
    assert status == "solved" and sudoku_stress.is_solution(sol, "0" * 81)


def test_puzzle_sets_have_the_intended_properties():
    sets = sudoku_stress.puzzle_sets([WIKI_PUZZLE])
    assert sudoku_stress.solve_exact(sets["shuffled"][0])[1] == "solved"
    assert sets["fewer_clues_5"][0].count("0") == WIKI_PUZZLE.count("0") + 5
    assert sudoku_stress.solve_exact(sets["unsolvable"][0])[1] == "unsolvable"
    assert sets["already_solved"] == [WIKI_SOLUTION]


def test_parse_llm_reply():
    reply = "Here is the solution:\n" + "\n".join(WIKI_SOLUTION[i : i + 9]
                                                   for i in range(0, 81, 9))
    assert sudoku_stress.parse_llm(reply) == (WIKI_SOLUTION, False)
    assert sudoku_stress.parse_llm("UNSOLVABLE: row 1 has two 5s") == (None, True)


def test_sudoku_evaluate_scores_unsolvable_by_saying_so():
    sets = sudoku_stress.puzzle_sets([WIKI_PUZZLE])
    fake = FakeSudoku("x")  # always returns the same grid, never says "unsolvable"
    rows, _ = sudoku_stress.evaluate(
        sets, {"exact": sudoku_stress.run_exact,
               "fake NCM": lambda ps: sudoku_stress.run_batched(fake.generate, ps)})
    by = {(r["solver"], r["set"]): r for r in rows}
    assert all(by[("exact", s)]["correct"] == 1.0 for s in sets)
    assert by[("fake NCM", "unsolvable")]["correct"] == 0.0
    assert by[("fake NCM", "unsolvable")]["full_grid_returned"] == 1.0


# ---- 7. specialist vs LLM --------------------------------------------------------------
def test_clean_llm_output():
    assert llm_vs_ncm.clean_llm('Summary: "A bridge was approved."\n\nNote: ...') == \
        "A bridge was approved."
    assert llm_vs_ncm.clean_llm("\nTranslation: Good morning.") == "Good morning."


# ---- 8. agent reliability --------------------------------------------------------------
class OneToolLLM:
    """Calls search_docs once (with broken JSON when seed is odd), then answers."""

    def __init__(self, temperature, seed):
        self.seed, self.calls = seed, 0

    def chat(self, messages, tools=None, temperature=None):
        self.calls += 1
        if self.calls == 1 and tools is not None:
            args = "{oops" if self.seed % 2 else {"query": "Rhine"}
            return LLMReply("", tool_calls=[{"function": {"name": "search_docs",
                                                          "arguments": args}}])
        return LLMReply("It flows through Germany.",
                        raw_message={"role": "assistant", "content": "x"})


def test_agent_reliability_counts_tool_errors_and_accuracy():
    from ncm_lab.agents.tools import Toolbox

    tools = Toolbox(store=FakeStore(), summarizer=FakeTextModel("s"),
                    translator=FakeTextModel("t"), sudoku=FakeSudoku("x"))
    tasks = [{"name": "search_answer", "task": "Where does the Rhine flow?",
              "gold": ["Germany"]}]
    rows, details = agent_reliability.evaluate(tasks, OneToolLLM, tools, repeats=4)
    greedy, sampled = rows
    assert greedy["runs"] == 1 and greedy["runs_with_tool_error"] == 0.0
    assert sampled["runs"] == 4 and sampled["runs_with_tool_error"] == 0.5
    assert sampled["accuracy"] == 1.0 and sampled["out_of_steps"] == 0.0
    assert len(details) == 5


# ---- report ----------------------------------------------------------------------------
def test_report_builds_summary_charts_and_latex(tmp_path):
    sweep = []
    for name, oom in (("NCM (xsum)", 64), ("bart-base-xsum", None)):
        sweep += batch_sweep.sweep(FakeSeqModel(oom_above=oom), name, "xsum", ["a"] * 4,
                                   batch_sizes=(1, 8, 128), examples=8)
    write("batch_sweep", sweep, results=tmp_path)
    write("llm_vs_ncm", [
        {"task": "xsum", "system": s, "params_M": p, "rougeL": q, "latency_p50_ms": ms,
         "latency_p95_ms": ms * 1.2} for s, p, q, ms in
        [("NCM (xsum)", 149, 27.8, 268), ("bart-base-xsum", 139, 33.5, 274),
         ("qwen2.5:3b (zero-shot)", 3090, 25.0, 1500)]], results=tmp_path)
    assert read("batch_sweep", tmp_path)[0]["oom"] == "False"

    out = report.main(results=tmp_path, tables=tmp_path / "tables")
    text = out.read_text()
    assert "Throughput and memory vs batch size" in text and "Specialist NCM" in text
    assert (tmp_path / "limits_batch_sweep.png").exists()
    assert (tmp_path / "limits_llm_vs_ncm.png").exists()
    assert "\\begin{tabular}" in (tmp_path / "tables" / "limits_batch_sweep.tex").read_text()


def test_runner_continues_after_a_failed_experiment(monkeypatch, capsys):
    import sys
    import types

    import pytest

    from ncm_lab.limits import run

    ran = []
    ok = types.SimpleNamespace(main=lambda n, precision: ran.append("ok"))
    bad = types.SimpleNamespace(main=lambda n, precision: sys.exit("Ollama is not reachable"))
    modules = {"ncm_lab.limits.faithfulness": bad, "ncm_lab.limits.batch_sweep": ok}
    monkeypatch.setattr(run.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(sys, "argv", ["run", "faithfulness", "batch-sweep", "--no-report"])
    with pytest.raises(SystemExit, match="failed: faithfulness"):
        run.main()
    assert ran == ["ok"]
    assert "faithfulness failed" in capsys.readouterr().out
