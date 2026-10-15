"""Check vulnerable/fixed numerical equivalence for Phase 2-4 cases C001-C009."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.phase_2_runtime.measure import validate

if __name__ == '__main__':
    validate()
