from covt_persp.score import bootstrap_ci, extract_letter


def test_answer_tag_wins():
    assert extract_letter("Maybe A... <answer>C</answer>") == "C"


def test_answer_tag_with_text():
    assert extract_letter("<answer> (B) the converging lines </answer>") == "B"


def test_boxed():
    assert extract_letter("so the answer is \\boxed{D}") == "D"


def test_last_standalone_letter():
    assert extract_letter("Option A is wrong, so B") == "B"


def test_ignores_letters_inside_words():
    assert extract_letter("A Bridge And Dock") == "A"
    assert extract_letter("Because of the edges") is None


def test_custom_letters():
    assert extract_letter("<answer>E</answer>", list("ABCDE")) == "E"


def test_ci_bounds():
    lo, hi = bootstrap_ci([1, 0] * 100)
    assert 0.4 < lo < 0.5 < hi < 0.6


def test_placeholder_options_prompt():
    from covt_persp.prompts import build_prompt
    p = build_prompt("Fill the ?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "direct")
    assert "shown in the image" in p and "A. a" not in p
    p2 = build_prompt("Which line?", {"A": "Line A", "B": "Line B"}, "direct")
    assert "A. Line A" in p2


def test_mcnemar():
    from covt_persp.score import mcnemar_p
    assert mcnemar_p(0, 0) == 1.0
    assert abs(mcnemar_p(5, 5) - 1.0) < 1e-9
    assert mcnemar_p(0, 10) < 0.01
