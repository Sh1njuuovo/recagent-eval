from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path


def read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot safely open {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"unsafe non-regular file: {path}")
        if info.st_size > max_bytes:
            raise ValueError(f"file exceeds maximum size: {path}")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes or len(payload) != info.st_size:
            raise ValueError(f"file changed or was partially read: {path}")
        return payload
    finally:
        os.close(descriptor)


def ensure_distinct_files(paths: Mapping[str, Path]) -> None:
    """Reject path aliases, including aliases through existing hard links."""
    resolved: dict[Path, str] = {}
    inodes: dict[tuple[int, int], str] = {}
    for label, path in paths.items():
        canonical = path.resolve(strict=False)
        if canonical in resolved:
            raise ValueError(
                f"{label} and {resolved[canonical]} paths must be distinct"
            )
        resolved[canonical] = label
        try:
            info = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        identity = (info.st_dev, info.st_ino)
        if identity in inodes:
            raise ValueError(f"{label} path aliases {inodes[identity]} path")
        inodes[identity] = label
