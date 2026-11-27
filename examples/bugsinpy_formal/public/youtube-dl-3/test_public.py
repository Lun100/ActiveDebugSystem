from youtube_dl.utils import unescapeHTML


def test_bare_ampersand_preserves_later_entity():
    assert unescapeHTML("rock & roll &amp; jazz") == "rock & roll & jazz"
