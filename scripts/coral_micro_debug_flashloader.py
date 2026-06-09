#!/usr/bin/env python3
"""Run the Coral Micro flashloader load step with visible NXP blhost output."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BD_FILE = """options {
    flags = 0x00;
    startAddress = 0x2024ff00;
    ivtOffset = 0x0;
    initialLoadSize = 0x2000;
    entryPointAddress = 0x20250000;
}
sources {
    elfFile = extern(0);
}
section (0) {
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coralmicro", type=Path, default=Path("../coralmicro"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/profiles/coral/flashloader_debug")
    )
    parser.add_argument("--skip-load", action="store_true", help="Only generate the IVT binary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sdk = args.coralmicro.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    elftosb = sdk / "third_party/nxp/elftosb/linux/amd64/elftosb"
    blhost = sdk / "third_party/nxp/blhost/bin/linux/amd64/blhost"
    srec = sdk / "build/libs/nxp/flashloader/image.srec"
    for path in (elftosb, blhost, srec):
        if not path.exists():
            raise FileNotFoundError(path)

    bdfile = output_dir / "flashloader.bd"
    ivt = output_dir / "ivt_flashloader.bin"
    bdfile.write_text(BD_FILE, encoding="utf-8")

    subprocess.run(
        [str(elftosb), "-f", "imx", "-V", "-c", str(bdfile), "-o", str(ivt), str(srec)],
        check=True,
    )
    print(f"Wrote {ivt}")

    if args.skip_load:
        return

    subprocess.run(
        [str(blhost), "-V", "-u", "0x1fc9,0x13d", "--", "load-image", str(ivt)],
        check=True,
    )


if __name__ == "__main__":
    main()
