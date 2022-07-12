# mypy: ignore-errors

import nox

ALL_PYTHON_VS = ["3.8", "3.9", "3.10"]


@nox.session(python=ALL_PYTHON_VS)
@nox.parametrize("x64", [True, False])
def test(session, x64):
    session.install(".[test]")
    if x64:
        env = {"JAX_ENABLE_X64": "1"}
    else:
        env = {}

    session.run("pytest", "-n", "auto", *session.posargs, env=env)


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
