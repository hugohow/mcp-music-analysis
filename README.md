[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/hugohow-mcp-music-analysis-badge.png)](https://mseep.ai/app/hugohow-mcp-music-analysis)

# MCP Music Analysis
[![smithery badge](https://smithery.ai/badge/@hugohow/mcp-music-analysis)](https://smithery.ai/server/@hugohow/mcp-music-analysis)

This repository contains a **Model Context Provider (MCP)** that uses MCP and [librosa](https://librosa.org/) for audio analysis on audio in local, youtube link, or audio link.

## Usage with Claude Desktop

<div style="display: flex; gap: 1rem;">
  <img src="public/screen.png" alt="alt text" width="40%">
  <img src="public/screen1.png" alt="alt text" width="40%">
</div>

## Installation

### Installing via Smithery

To install Music Analysis for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@hugohow/mcp-music-analysis):

```bash
npx -y @smithery/cli install @hugohow/mcp-music-analysis --client claude
```

### Manual Installation

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

#### Locate Configuration File

The configuration file location depends on your operating system:

- **macOS**:
  ```
  ~/Library/Application\ Support/Claude/claude_desktop_config.json
  ```

- **Windows**:
  ```
  %APPDATA%\Claude\claude_desktop_config.json
  ```

- **Linux**:
  ```
  ~/.config/Claude/claude_desktop_config.json
  ```

Add the following to your `claude_desktop_config.json`:

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
