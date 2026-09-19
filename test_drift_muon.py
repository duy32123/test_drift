"""Convenience entry point for the NEW numerical and integration test suite."""
import sys
from pathlib import Path
import pytest
if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).parent/"tests"),"-q",*sys.argv[1:]]))
