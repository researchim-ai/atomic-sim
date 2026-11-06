"""Fetch one pinned public ZIP member through bounded HTTPS range reads.

The 18 GB archive is not downloaded wholesale. ZIP CRC and the original CSV
SHA256 are verified before atomic publication. No upstream code is executed.
"""

import argparse
import hashlib
import io
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from atomic_model_phase_assessment.importer import RAW_PIN

URL = "https://zenodo.org/api/records/22234711/files/ned3-013_SubatmosphericBoiling.zip/content"
SIZE = 18849974237
MEMBER = "ned3-013_SubatmosphericBoiling/A-FlatCu/100kPa/Boiling-274_csv_data.csv"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect refused for pinned source")


class RangeReader(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.calls = 0
        self.total = 0
        self.cache = {}
        self.opener = urllib.request.build_opener(NoRedirect())

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek mode")
        target = offset if whence == 0 else self.pos + offset if whence == 1 else SIZE + offset
        if not 0 <= target <= SIZE:
            raise ValueError("Seek outside archive")
        self.pos = target
        return target

    def read(self, n=-1):
        if n < 0:
            n = SIZE - self.pos
        n = min(n, SIZE - self.pos)
        if n > 2_000_000:
            raise ValueError("Read budget exceeded")
        out = []
        while n:
            block = self.pos // 1_000_000
            begin = block * 1_000_000
            end = min(SIZE, begin + 1_000_000)
            if block not in self.cache:
                self.calls += 1
                if self.calls > 100 or self.total + (end - begin) > 70_000_000:
                    raise ValueError("Network budget exceeded")
                req = urllib.request.Request(URL, headers={"Range": f"bytes={begin}-{end - 1}"})
                with self.opener.open(req, timeout=30) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {begin}-{end - 1}/{SIZE}"
                    ):
                        raise ValueError("Pinned archive range mismatch")
                    raw = response.read(end - begin + 1)
                if len(raw) != end - begin:
                    raise ValueError("Incomplete range")
                self.cache[block] = raw
                self.total += len(raw)
                if len(self.cache) > 3:
                    del self.cache[next(iter(self.cache))]
            count = min(n, end - self.pos)
            out.append(self.cache[block][self.pos - begin : self.pos - begin + count])
            self.pos += count
            n -= count
        return b"".join(out)


def fetch(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        reader = RangeReader()
        with zipfile.ZipFile(reader) as archive:
            member = archive.getinfo(MEMBER)
            if (
                member.file_size != 518599832
                or member.CRC != 3284962502
                or member.compress_size > 64_000_000
            ):
                raise ValueError("Pinned ZIP member metadata mismatch")
            with (
                archive.open(member) as src,
                tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as dst,
            ):
                temporary = Path(dst.name)
                h = hashlib.sha256()
                size = 0
                while chunk := src.read(1_000_000):
                    size += len(chunk)
                    if size > member.file_size:
                        raise ValueError("Plaintext budget exceeded")
                    dst.write(chunk)
                    h.update(chunk)
            if size != member.file_size or h.hexdigest() != RAW_PIN:
                raise ValueError("Primary CSV identity mismatch")
        # Hard-link publication is atomic and refuses a concurrently created target.
        os.link(temporary, output)
        return dict(
            primary_sha256=RAW_PIN,
            bytes=size,
            zip_member_crc_verified=True,
            whole_zip_checksum_verified=False,
            network_bytes=reader.total,
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    print(fetch(p.parse_args().output))


if __name__ == "__main__":
    main()
