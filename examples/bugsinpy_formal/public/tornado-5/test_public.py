from tornado.ioloop import PeriodicCallback


def test_small_backward_clock_step_advances_timer():
    periodic = PeriodicCallback(lambda: None, 250)
    periodic._next_timeout = 10.0
    periodic._update_next(9.9)
    assert periodic._next_timeout == 10.25
