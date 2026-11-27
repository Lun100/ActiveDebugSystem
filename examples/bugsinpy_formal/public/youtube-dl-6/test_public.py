from youtube_dl.utils import parse_dfxp_time_expr


def test_missing_timestamp_is_not_zero():
    assert parse_dfxp_time_expr(None) is None
