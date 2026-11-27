"""Run one trusted evaluator suite inside its task's container."""

import pytest


class SuiteOutcome:
    def __init__(self):
        self.unverified = set()

    def pytest_collection_finish(self, session):
        self.unverified.update(item.nodeid for item in session.items)

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed and not hasattr(report, "wasxfail"):
            self.unverified.discard(report.nodeid)


def main() -> int:
    outcome = SuiteOutcome()
    status = pytest.main(
        ["-q", "-c", "/dev/null", "--noconftest", "-p", "no:cacheprovider", "/suite"],
        plugins=[outcome],
    )
    if status == 0:
        return 1 if outcome.unverified else 10
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
