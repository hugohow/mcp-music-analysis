# server.py

from fastmcp import FastMCP
import librosa

# Create an MCP server
mcp = FastMCP(
    "Audio analysis of audio",
    dependencies=["librosa"],
)


@mcp.tool()
def beat(file_path: str) -> int:
    """Get the beat of an audio file"""
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo


@mcp.prompt()
def analyze_audio() -> str:
    """Create a prompt template for analyzing audio"""
    return f""
