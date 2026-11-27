from tornado.httputil import url_concat


def test_no_query_args_preserves_url():
    url = "https://example.org/items?active=1"
    assert url_concat(url, None) == url
