from tqdm import tqdm


def test_explicit_positive_total_takes_precedence_over_an_empty_iterable():
    with tqdm([], total=2, disable=True) as progress:
        assert bool(progress)
