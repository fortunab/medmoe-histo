import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile

from .data import DATASETS


def main():
    parser = argparse.ArgumentParser(description="Fetch an official dataset archive")
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--directory", type=Path, default=Path("data"))
    args = parser.parse_args()
    url = DATASETS[args.dataset]["url"]
    args.directory.mkdir(parents=True, exist_ok=True)
    archive = args.directory / PurePosixPath(url).name
    if not archive.exists():
        with tempfile.TemporaryDirectory(dir=args.directory) as staging:
            partial = Path(staging) / archive.name
            print(f"Downloading {url}", flush=True)
            with urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            with ZipFile(partial) as bundle:
                if not bundle.namelist():
                    raise ValueError(f"Empty ZIP: {url}")
            partial.rename(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"{archive}\nSHA-256 {digest}")


if __name__ == "__main__":
    main()
