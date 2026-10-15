"""Readiness and bounded caches for CPU deployment."""

from collections import OrderedDict
from fastapi.testclient import TestClient

import server


def test_health_does_not_load_a_model(monkeypatch):
    def fail():
        raise AssertionError("Health must not load models")

    monkeypatch.setattr(server.zerotts_backend, "_load_engine", fail)
    monkeypatch.setattr(server.vtts_backend, "_load_engine", fail)
    response = TestClient(server.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_space_home_redirects_to_docs():
    response = TestClient(server.app).get("/", follow_redirects=False)
    assert response.headers["location"] == "/docs"


def test_edge_cache_remains_bounded(monkeypatch):
    monkeypatch.setattr(server, "_audio_cache", OrderedDict())
    monkeypatch.setattr(server, "_EDGE_RAM_MAX_BYTES", 10)
    server._remember_edge_audio("a", b"123456")
    server._remember_edge_audio("b", b"abcdef")
    assert list(server._audio_cache) == ["b"]
    server._remember_edge_audio("large", b"x" * 11)
    assert list(server._audio_cache) == ["b"]


def test_startup_loads_only_cache_files_that_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(server, "_audio_cache", OrderedDict())
    monkeypatch.setattr(server, "_EDGE_RAM_MAX_BYTES", 10)
    (tmp_path / "old.mp3").write_bytes(b"old-audio")
    (tmp_path / "new.mp3").write_bytes(b"new-audio")
    server._load_disk_cache()
    assert sum(map(len, server._audio_cache.values())) <= 10
