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
