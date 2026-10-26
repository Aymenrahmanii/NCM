from ncm_lab.agents.tools import Toolbox
from ncm_lab.api.registry import FakeStore, FakeSudoku, FakeTextModel
from ncm_lab.models.ncm import NCMSudokuSolver
from ncm_lab.rag.data import is_correct, normalize, split_sentences


def test_normalize_drops_case_punctuation_and_articles():
    assert normalize("The  Denver Broncos!") == "denver broncos"


def test_is_correct_is_lenient_containment():
    assert is_correct("It was the Denver Broncos.", ["Denver Broncos"])
    assert not is_correct("Carolina Panthers", ["Denver Broncos"])


def test_split_sentences():
    text = "Berlin ist die Hauptstadt. Sie hat 3,7 Millionen Einwohner! Stimmt das? Ja."
    assert split_sentences(text) == ["Berlin ist die Hauptstadt.",
                                     "Sie hat 3,7 Millionen Einwohner!", "Stimmt das?", "Ja."]


def test_sudoku_validator():
    good = FakeSudoku("s").generate(["x"])[0]
    assert NCMSudokuSolver.is_valid(good)
    assert not NCMSudokuSolver.is_valid(good[:-1] + "1")
    puzzle = "0" + good[1:]
    assert NCMSudokuSolver.is_valid(good, puzzle)
    assert not NCMSudokuSolver.is_valid(good, "9" + good[1:])  # clue not kept


def test_toolbox_reports_errors_instead_of_crashing():
    tools = Toolbox(store=FakeStore(), summarizer=FakeTextModel("s"),
                    translator=FakeTextModel("t"), sudoku=FakeSudoku("x"))
    result, ms = tools.call("solve_sudoku", {"wrong_arg": 1})
    assert result.startswith("error:") and ms >= 0
