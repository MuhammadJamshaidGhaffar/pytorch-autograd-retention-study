from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    raise SystemExit(subprocess.call([sys.executable, str(root / "pilot/measure.py"),
                                    "--case", Path(__file__).parent.name, *sys.argv[1:]]))
