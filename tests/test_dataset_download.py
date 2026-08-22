from __future__ import annotations

import io
import zipfile

import pytest

from recagent_eval.dataset import download_movielens_1m


def _archive_bytes(entries: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return stream.getvalue()


def test_download_extracts_a_bounded_movielens_archive_without_network(
    tmp_path, monkeypatch
) -> None:
    payload = _archive_bytes(
        {
            "ml-1m/movies.dat": b"1::Movie (2000)::Drama\n",
            "ml-1m/ratings.dat": b"1::1::5::1\n",
        }
    )
    requests = []

    def fake_urlopen(url, *, timeout):
        requests.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr("recagent_eval.dataset.urllib.request.urlopen", fake_urlopen)

    extracted = download_movielens_1m(tmp_path)

    assert requests == [
        ("https://files.grouplens.org/datasets/movielens/ml-1m.zip", 60)
    ]
    assert (extracted / "movies.dat").read_bytes().startswith(b"1::Movie")

    monkeypatch.setattr(
        "recagent_eval.dataset.urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("existing archive should be reused"),
    )
    assert download_movielens_1m(tmp_path) == extracted


def test_download_rejects_archive_path_traversal_before_extracting(
    tmp_path, monkeypatch
) -> None:
    payload = _archive_bytes({"../escaped.txt": b"unsafe"})
    monkeypatch.setattr(
        "recagent_eval.dataset.urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(payload),
    )

    with pytest.raises(ValueError, match="unsafe archive member"):
        download_movielens_1m(tmp_path)

    assert not (tmp_path.parent / "escaped.txt").exists()
