import os

import pytest

os.environ.setdefault("ARTH_HASH_SALT", "test-salt")


@pytest.fixture(scope="session")
def gold_split():
    from arth.gold import DEFAULT_GOLD, load_jsonl, split_gold

    return split_gold(load_jsonl(DEFAULT_GOLD))


@pytest.fixture(scope="session")
def bundle(gold_split):
    """A small but real student bundle: hashing embedder, 8k synthetic lines, gold calibration."""
    from arth.synth import generate
    from arth.train import build_bundle

    cal, _ = gold_split
    return build_bundle(generate(8000, seed=7), cal, version="t1", embedder_spec="hash", seed=7, epochs=5,
                        rank=64, n_random_views=40)


@pytest.fixture()
def service(bundle, tmp_path):
    from arth.api import Service
    from arth.engine import AuditMonitor
    from arth.registry import Registry
    from arth.store import Store

    reg = Registry(tmp_path / "models")
    reg.save(bundle)
    return Service(Store(":memory:"), reg, bundle=bundle, audit=AuditMonitor())


@pytest.fixture()
def client(service):
    from fastapi.testclient import TestClient

    from arth.api import create_app

    return TestClient(create_app(service))
