from tqdm.contrib import tenumerate


def test_enumeration_respects_a_nonzero_start():
    assert list(tenumerate(["one", "two"], start=5, disable=True)) == [(5, "one"), (6, "two")]
