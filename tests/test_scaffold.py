"""Scaffold smoke tests.

These prove the package imports and its configuration surface behaves, so the
CI pipeline is genuinely green on the empty scaffold rather than green because
it ran nothing. Real tests arrive with each milestone's code.
"""

from lonestar import __version__, config


def test_package_imports_and_has_a_version():
    assert __version__ == "0.1.0"


def test_database_url_is_wellformed():
    url = config.database_url()
    assert url.startswith("postgresql+psycopg2://")
    assert config.DB_NAME in url


def test_seed_is_an_integer():
    # Reproducibility hinges on a fixed integer seed.
    assert isinstance(config.SEED, int)
