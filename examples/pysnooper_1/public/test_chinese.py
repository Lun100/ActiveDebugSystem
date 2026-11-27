# -*- coding: utf-8 -*-
"""Public reproduction of BugsInPy PySnooper bug 1."""

import pysnooper


def test_chinese(tmp_path):
    path = tmp_path / "trace.log"

    @pysnooper.snoop(path)
    def traced():
        message = "失败"
        return message

    assert traced() == "失败"
    assert "失败" in path.read_text(encoding="utf-8")
