"""Install the unmodified RESCO v1.0 scenario into an ignored local directory."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

REVISION = "fc65e91f6711d53296dee824e684b201a3e6808d"
BASE = f"https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/{REVISION}/resco_benchmark/environments/ingolstadt21/"
FILES = {
    "LICENSE": (35149, "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7"),
    "ingolstadt21.net.xml": (1860867, "6b01f05c388efe60df218c9ae99c8b639d41dfa9"),
    "ingolstadt21.rou.xml": (425223, "3560385ee08da71364ae6076f9df82bfaacbcc6b"),
    "ingolstadt21.sumocfg": (215, "e87b8c6d226c483b5b35a53c06e135ad87defc14"),
}
DESTINATION = Path(__file__).resolve().parents[1] / "src/networks/ingolstadt21"


def verify(directory):
    for name, (size, expected) in FILES.items():
        data = (directory / name).read_bytes()
        blob = f"blob {len(data)}\0".encode() + data
        if len(data) != size or hashlib.sha1(blob).hexdigest() != expected:
            raise ValueError(f"Integrity check failed for {name}")
    print(f"Verified {len(FILES)} RESCO v1.0 files in {directory}")


def main():
    if DESTINATION.exists():
        verify(DESTINATION)
        return

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=DESTINATION.parent) as temporary:
        staged = Path(temporary) / "ingolstadt21"
        staged.mkdir()
        for name, (size, _) in FILES.items():
            with urlopen(BASE + name, timeout=30) as response:
                (staged / name).write_bytes(response.read(size + 1))
        verify(staged)
        staged.rename(DESTINATION)


if __name__ == "__main__":
    main()
