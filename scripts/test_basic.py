#!/usr/bin/env python3
"""
Basic pytest tests for the Turkish AI project.
These tests verify core functionality and project structure.
"""

import pytest
import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_project_structure():
    """Test that essential project files exist."""
    project_root = Path(__file__).parent.parent

    # Check essential directories
    assert (project_root / "services").exists(), "Services directory should exist"
    assert (project_root / "configs").exists(), "Configs directory should exist"
    assert (project_root / "scripts").exists(), "Scripts directory should exist"

    # Check essential files
    assert (
        project_root / "requirements_benchmark.txt"
    ).exists(), "Requirements file should exist"
    assert (
        project_root / "docker-compose.yml"
    ).exists(), "Docker compose file should exist"


def test_basic_math():
    """Basic test to ensure pytest is working correctly."""
    assert 2 + 2 == 4
    assert 10 * 5 == 50
    assert "hello" + " world" == "hello world"


def test_python_version():
    """Test that we are running on a supported Python version."""
    import sys

    version = sys.version_info
    assert version.major == 3, "Should be running Python 3"
    assert version.minor >= 9, "Should be running Python 3.9 or higher"


if __name__ == "__main__":
    pytest.main([__file__])
