"""Limit 6: sudoku outside the training distribution: NCM vs an exact solver vs a general LLM.

NCM solves 84.5% of the official hard test puzzles (Phase 3). These sets ask what happens
around that:

- hard: the official test puzzles (reference point);
- shuffled: the same puzzles with digits relabelled and the grid transposed: logically
  identical, so a solver that reasons should score the same;
- fewer_clues_5 / _15: 5 or 15 clues removed: still solvable (often in several ways, any
  valid completion counts), but sparser than anything in training;
- unsolvable: one empty cell filled with a digit already in its row; the right answer is
  "no solution", which NCM has no way to say;
- already_solved: the full solution as the puzzle; the answer is the input itself.

The exact solver (backtracking, CPU) is the ground truth and the speed reference.
"""

import pickle
import random
import re
import time
from functools import partial
from pathlib import Path

from .common import free_gpu, mean, write

DATA = Path("data/sudoku")
DIGITS = "123456789"
UNSOLVABLE_WORDS = ("unsolvable", "no solution", "not solvable", "impossible", "invalid")


# ---- exact solver --------------------------------------------------------------------------
def _units(i: int) -> tuple[int, int, int]:
    r, c = divmod(i, 9)
    return r, c, (r // 3) * 3 + c // 3


def solve_exact(puzzle: str, max_nodes: int = 2_000_000) -> tuple[str | None, str]:
    """(solution, status) with status "solved", "unsolvable" or "gave_up"."""
    grid = [0 if ch in "0." else int(ch) for ch in puzzle]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(grid):
        if v:
            r, c, b = _units(i)
            bit = 1 << v
            if (rows[r] | cols[c] | boxes[b]) & bit:
                return None, "unsolvable"  # two equal clues in a row, column or box
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
    nodes = 0

    def search() -> bool | None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return None
        best, best_cands = -1, None
        for i, v in enumerate(grid):  # most constrained empty cell first
            if v == 0:
                r, c, b = _units(i)
                used = rows[r] | cols[c] | boxes[b]
                cands = [d for d in range(1, 10) if not used & (1 << d)]
                if best_cands is None or len(cands) < len(best_cands):
                    best, best_cands = i, cands
                    if len(cands) <= 1:
                        break
        if best < 0:
            return True
        r, c, b = _units(best)
        for d in best_cands:
            bit = 1 << d
            grid[best] = d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            found = search()
            if found or found is None:
                return found
            grid[best] = 0
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[b] &= ~bit
        return False

    found = search()
    if found is None:
        return None, "gave_up"
    return ("".join(map(str, grid)), "solved") if found else (None, "unsolvable")


# ---- puzzle sets ---------------------------------------------------------------------------
def load_puzzles(difficulty: str = "hard", n: int = 100) -> list[str]:
    raw = pickle.load(open(DATA / difficulty / "puzzles.pickle", "rb"))[:n]
    return ["".join(str(t) for t in ids if t < 10) for ids in raw]


def shuffled(p: str, rng: random.Random) -> str:
    perm = dict(zip(DIGITS, rng.sample(DIGITS, 9)))
    relabelled = [perm.get(ch, "0") for ch in p]
    return "".join(relabelled[9 * c + r] for r in range(9) for c in range(9))  # transpose


def fewer_clues(p: str, k: int, rng: random.Random) -> str:
    cells = list(p)
    clues = [i for i, ch in enumerate(cells) if ch not in "0."]
    for i in rng.sample(clues, min(k, len(clues))):
        cells[i] = "0"
    return "".join(cells)


def make_unsolvable(p: str, rng: random.Random) -> str:
    """Fill an empty cell with a digit that is already a clue in the same row."""
    cells = list(p)
    for r in rng.sample(range(9), 9):
        row = range(9 * r, 9 * r + 9)
        empty = [i for i in row if cells[i] in "0."]
        given = [cells[i] for i in row if cells[i] not in "0."]
        if empty and given:
            cells[rng.choice(empty)] = rng.choice(given)
            return "".join(cells)
    raise ValueError("puzzle has no row with both a clue and a blank")


def puzzle_sets(base: list[str], seed: int = 0) -> dict[str, list[str]]:
    rng = random.Random(seed)
    solved = [s for s, status in map(solve_exact, base) if status == "solved"]
    return {
        "hard": base,
        "shuffled": [shuffled(p, rng) for p in base],
        "fewer_clues_5": [fewer_clues(p, 5, rng) for p in base],
        "fewer_clues_15": [fewer_clues(p, 15, rng) for p in base],
        "unsolvable": [make_unsolvable(p, rng) for p in base],
        "already_solved": solved,
    }


# ---- solvers -------------------------------------------------------------------------------
def is_solution(sol: str | None, puzzle: str) -> bool:
    from ncm_lab.models.ncm import NCMSudokuSolver

    return sol is not None and NCMSudokuSolver.is_valid(sol, puzzle)


def full_grid(sol: str | None) -> bool:
    return sol is not None and len(sol) == 81 and all(ch in DIGITS for ch in sol)


def llm_prompt(puzzle: str) -> str:
    rows = "\n".join(puzzle[i : i + 9] for i in range(0, 81, 9))
    return ("Solve this sudoku. 0 means an empty cell. Reply with the 9 rows of the solution "
            "(9 digits each) and nothing else, or reply UNSOLVABLE if it has no solution.\n\n"
            + rows)


def parse_llm(text: str) -> tuple[str | None, bool]:
    """(the last 81 digits in the reply, whether it says the puzzle is unsolvable)."""
    said_unsolvable = any(w in text.lower() for w in UNSOLVABLE_WORDS)
    digits = re.findall(r"\d", text)
    return ("".join(digits[-81:]) if len(digits) >= 81 else None), said_unsolvable


def score(solver: str, set_name: str, puzzles: list[str], outputs: list[tuple],
          ms: list[float]) -> dict:
    """outputs: (solution or None, said "unsolvable") per puzzle."""
    solved = [is_solution(s, p) for (s, _), p in zip(outputs, puzzles)]
    said = [u for _, u in outputs]
    correct = said if set_name == "unsolvable" else solved
    return {"set": set_name, "solver": solver, "n": len(puzzles),
            "correct": round(sum(correct) / len(puzzles), 3),
            "valid_solution": round(sum(solved) / len(puzzles), 3),
            "full_grid_returned": round(sum(full_grid(s) for s, _ in outputs) / len(puzzles), 3),
            "said_unsolvable": round(sum(said) / len(puzzles), 3),
            "ms_per_puzzle": round(mean(ms), 2)}


def run_exact(puzzles: list[str]) -> tuple[list[tuple], list[float]]:
    outs, ms = [], []
    for p in puzzles:
        t = time.perf_counter()
        sol, status = solve_exact(p)
        ms.append((time.perf_counter() - t) * 1000)
        outs.append((sol, status == "unsolvable"))
    return outs, ms


def run_batched(generate, puzzles: list[str]) -> tuple[list[tuple], list[float]]:
    """A batched model (NCM): never says "unsolvable"; ms is the batched cost per puzzle."""
    from .common import timed

    sols, total = timed(generate, puzzles)
    return [(s, False) for s in sols], [total / len(puzzles)] * len(puzzles)


def run_llm(llm, puzzles: list[str]) -> tuple[list[tuple], list[float]]:
    outs, ms = [], []
    for p in puzzles:
        reply = llm.ask(llm_prompt(p))
        outs.append(parse_llm(reply.content))
        ms.append(reply.ms)
    return outs, ms


def evaluate(sets: dict[str, list[str]], solvers: dict, llm_n: int | None = None):
    """solvers: name -> function(puzzles) -> (outputs, ms). LLMs get the first llm_n only."""
    rows, details = [], []
    for name, run in solvers.items():
        for set_name, puzzles in sets.items():
            ps = puzzles[:llm_n] if (llm_n and name.startswith("LLM")) else puzzles
            outs, ms = run(ps)
            rows.append(score(name, set_name, ps, outs, ms))
            details += [{"solver": name, "set": set_name, "puzzle": p, "output": s,
                         "said_unsolvable": u, "valid": is_solution(s, p)}
                        for p, (s, u) in list(zip(ps, outs))[:3]]
    return rows, details


def main(n: int = 100, precision: str = "fp16", llm: bool = True, llm_n: int = 20) -> None:
    from ncm_lab.models.ncm import NCMSudokuSolver

    sets = puzzle_sets(load_puzzles("hard", n))
    rows, details = evaluate(sets, {"exact solver (CPU)": run_exact})

    ncm = NCMSudokuSolver("hard", precision=precision)
    ncm.generate(sets["hard"][:16])  # warm-up
    r, d = evaluate(sets, {"NCM sudoku-hard": partial(run_batched, ncm.generate)})
    rows, details = rows + r, details + d
    del ncm
    free_gpu()

    if llm:
        from ncm_lab.rag.llm import OllamaLLM

        client = OllamaLLM()
        if client.available():
            client.ask("hi")  # load the model before timing
            r, d = evaluate(sets, {f"LLM {client.model}": lambda ps: run_llm(client, ps)},
                            llm_n=llm_n)
            rows, details = rows + r, details + d
        else:
            print("Ollama not reachable: skipping the LLM solver")
    write("sudoku_stress", rows, details)
