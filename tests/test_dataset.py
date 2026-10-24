"""Check the download of external benchmark datasets."""

import urllib.request

import pytest

from wren import dataset


def test_fetch_discards_an_interrupted_download(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "DATA", tmp_path)

    def interrupted(url, path):
        path.write_text("partial")
        raise ConnectionError

    monkeypatch.setattr(urllib.request, "urlretrieve", interrupted)
    with pytest.raises(ConnectionError):
        dataset.fetch("mobile_actions")
    assert not (tmp_path / "mobile_actions" / "dataset.jsonl").exists()

    monkeypatch.setattr(
        urllib.request, "urlretrieve", lambda url, path: path.write_text("ok")
    )
    [path] = dataset.fetch("mobile_actions")
    assert path.read_text() == "ok"
