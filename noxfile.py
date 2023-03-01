# mypy: ignore-errors

import nox

ALL_PYTHON_VS = ["3.9", "3.10", "3.11"]


@nox.session(python=ALL_PYTHON_VS)
@nox.parametrize("x64", [True, False])
def test(session, x64):
    session.install(".[test,test-math]")
    if x64:
        env = {"JAX_ENABLE_X64": "True"}
    else:
        env = {"JAX_ENABLE_X64": "False"}
    session.run("pytest", "-n", "auto", *session.posargs, env=env)


@nox.session(python=ALL_PYTHON_VS)
def comparison(session):
    session.install(".[test,comparison]", "numpy<1.22")
    session.run(
        "pytest",
        "-n",
        "auto",
        "tests/experimental/starry",
        *session.posargs,
        env={"JAX_ENABLE_X64": "True"}
    )
