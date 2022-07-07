# mypy: ignore-errors

import nox

ALL_PYTHON_VS = ["3.8", "3.9", "3.10"]


@nox.session(python=ALL_PYTHON_VS)
def test(session):
    session.install(".[test]")
    session.run("pytest", "-n", "auto", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def comparison(session):
    session.install(".[test,comparison]")
    session.run(
        "pytest",
        "-n",
        "auto",
        "tests/experimental/starry_test.py",
        *session.posargs,
        env={"JAX_ENABLE_X64": "1"}
    )
