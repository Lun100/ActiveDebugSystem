from tqdm import tqdm


def test_custom_bar_shows_both_sides():
    result = tqdm.format_meter(3, 10, 2, ncols=64, ascii=True,
                               bar_format="left {bar} {n}/{total} done")
    assert result.startswith("left ")
    assert result.endswith(" 3/10 done")
