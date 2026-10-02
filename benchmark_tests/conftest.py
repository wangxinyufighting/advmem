import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if os.getenv("BENCHMARK_CONTRACT_TESTS") == "1":
    from _legacy_contract import install
    install()
