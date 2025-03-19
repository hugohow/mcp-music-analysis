import pytest
from mcp_music_analysis.server import main


def test_server_creation():
    """Test that we can launch a server"""
    main()
    assert main is not None
