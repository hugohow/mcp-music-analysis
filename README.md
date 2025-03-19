# MCP Music Analysis

This repository contains a **Model Context Provider (MCP)** that uses MCP and [librosa](https://librosa.org/) for audio analysis on audio in local, youtube link, or audio link.

## Usage with Claude Desktop

<div style="display: flex; gap: 1rem;">
  <img src="public/screen.png" alt="alt text" width="40%">
  <img src="public/screen1.png" alt="alt text" width="40%">
</div>

## Installation

```bash
# Clone repository
git clone git@github.com:hugohow/mcp-music-analysis.git
cd mcp-music-analysis

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Usage with Claude Desktop

To use this with Claude Desktop, add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "music-analysis": {
      "command": "uvx",
      "args": ["-n", "mcp-music-analysis"]
    }
  }
}
```

## Example Prompts

Here are some sample prompts you might use in a conversational or chat-based context once the server is running. The MCP will understand these requests and execute the relevant tools:

```
Can you analyze the beat of /Users/hugohow-choong/Desktop/sample-6s.mp3?
Could you give me the duration of https://download.samplelib.com/mp3/sample-15s.mp3 ?
Please compute the MFCC for this file: /path/to/another_audio.mp3
What are the spectral centroid values for /path/to/music.wav?
I'd like to know the onset times for https://www.youtube.com/watch?v=8HFiFd9vx1c
```

## To-Do List

- [x] Add URL to audio file download
- [x] Add YouTube to audio file transformation
- [ ] Experiment with multiple Python environments (testing)
- [ ] Improve installation guide
- [ ] Integrate Whisper for lyrics
- [ ] Implement a Docker solution

## Author

Hugo How-Choong
