FROM python:3.8.1-slim-buster@sha256:73f3903470a6e55202a6bb989c23b047487eb1728feba655410076da24106838

RUN python -m pip --disable-pip-version-check install --no-cache-dir \
    pytest==7.4.4 exceptiongroup==1.3.2 iniconfig==2.1.0 packaging==26.2 \
    pluggy==1.5.0 tomli==2.5.0 typing-extensions==4.13.2

ENV LC_ALL=C \
    PYTHONCOERCECLOCALE=0 \
    PYTHONUTF8=0 \
    PYTHONIOENCODING=utf-8

COPY src/active_debug/evaluation/pytest_suite.py /opt/active-debug/pytest_suite.py
