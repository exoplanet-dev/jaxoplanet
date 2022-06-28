import nox

ALL_PYTHON_VS = ["3.8", "3.9", "3.10"]


@nox.session(python=ALL_PYTHON_VS)
def test(session):
    session.install(".[test]")
    session.run("pytest", "-v", "tests", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def test_comparison(session):
    session.install(".[test,comparison]")
    session.run(
        "pytest",
        "-n",
        "auto",
        "tests/ops/test_starry.py",
        *session.posargs,
        env={"JAX_ENABLE_X64": "1"}
    )


# @nox.session(python=ALL_PYTHON_VS)
# def test_comparison(session):
#     session.install(".[test,comparison]")
#     session.run("pytest", "-v", "tests/core_test.py", *session.posargs)


# @nox.session(python=ALL_PYTHON_VS)
# def test_pymc3(session):
#     session.install(".[test,pymc3]")
#     session.run("python", "-c", "import theano")
#     session.run("python", "-c", "import exoplanet_core.pymc3.ops")
#     session.run("pytest", "-v", "tests/pymc3_test.py", *session.posargs)


# @nox.session(python=ALL_PYTHON_VS)
# def test_pymc4(session):
#     session.install(".[test,pymc4]")
#     session.run("python", "-c", "import aesara")
#     session.run("python", "-c", "import exoplanet_core.pymc4.ops")
#     session.run("pytest", "-v", "tests/pymc4_test.py", *session.posargs)


# @nox.session(python=ALL_PYTHON_VS)
# def test_pymc4_jax(session):
#     session.install(".[test,pymc4,jax]")
#     session.run("python", "-c", "import jax")
#     session.run("python", "-c", "import aesara")
#     session.run("python", "-c", "import exoplanet_core.pymc4.ops")
#     session.run("pytest", "-v", "tests/pymc4_jax_test.py", *session.posargs)


# @nox.session(python=ALL_PYTHON_VS)
# def test_jax(session):
#     session.install(".[test,jax]")
#     session.run("python", "-c", "import jax")
#     session.run("python", "-c", "import exoplanet_core.jax.ops")
#     session.run("pytest", "-v", "tests/jax_test.py", *session.posargs)


# @nox.session(python=ALL_PYTHON_VS)
# def test_all(session):
#     session.install(".[test,pymc3,pymc4,jax,comparison]")
#     session.run("python", "-c", "import jax")
#     session.run("python", "-c", "import aesara")
#     session.run("python", "-c", "import theano")
#     session.run("pytest", "-v", "tests", *session.posargs)
