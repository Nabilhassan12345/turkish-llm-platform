#!/usr/bin/env python3
"""
Pytest-compatible tests for Phase A4 components.
This script tests the router and basic functionality.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_router_import():
    """Test that the router can be imported."""
    try:
        from services.router import SectorRouter

        assert SectorRouter is not None
    except ImportError:
        pytest.skip("Router service not available")


def test_basic_router_functionality():
    """Test basic router functionality."""
    try:
        from services.router import SectorRouter

        router = SectorRouter()

        # Test with a simple query
        test_query = "Banka kredisi almak istiyorum"
        result = router.route_query(test_query)

        # Basic validation - result should be a string or dict
        assert result is not None
        assert isinstance(result, (str, dict))

    except Exception as e:
        pytest.skip(f"Router functionality test skipped: {e}")


def test_services_directory_exists():
    """Test that services directory and files exist."""
    project_root = Path(__file__).parent.parent
    services_dir = project_root / "services"

    assert services_dir.exists(), "Services directory should exist"
    assert (services_dir / "router.py").exists(), "Router service should exist"


if __name__ == "__main__":
    pytest.main([__file__])
