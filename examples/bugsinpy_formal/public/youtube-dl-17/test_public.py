from youtube_dl.utils import cli_bool_option


def test_unset_boolean_option_is_omitted():
    assert cli_bool_option({}, "--feature", "feature") == []
