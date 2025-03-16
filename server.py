# server.py

from fastmcp import FastMCP
import librosa

# import numpy as np

# Create an MCP server with a descriptive name and relevant dependencies
mcp = FastMCP(
    "Comprehensive Audio Analysis with librosa",
    dependencies=["librosa"],
)

###############################################################################
# TOOLS
###############################################################################


@mcp.tool()
def beat(file_path: str) -> float:
    """
    Estimates the tempo (in BPM) of the given audio file using librosa.
    """
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo


@mcp.tool()
def duration(file_path: str) -> float:
    """
    Returns the total duration (in seconds) of the given audio file.
    """
    y, sr = librosa.load(file_path)
    return librosa.get_duration(y=y, sr=sr)


@mcp.tool()
def beat_frames(file_path: str) -> list:
    """
    Returns a list of frames where beats are detected in the audio.
    """
    y, sr = librosa.load(file_path)
    _, frames = librosa.beat.beat_track(y=y, sr=sr)
    return frames.tolist()


@mcp.tool()
def beat_times(file_path: str) -> list:
    """
    Returns a list of time positions (in seconds) of the detected beats.
    """
    y, sr = librosa.load(file_path)
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
    times = librosa.frames_to_time(frames, sr=sr)
    return times.tolist()


@mcp.tool()
def spectral_centroid(file_path: str) -> list:
    """
    Computes the spectral centroid for each frame in the audio and
    returns it as a list of floats. The spectral centroid indicates
    the "center of mass" of the spectrum.
    """
    y, sr = librosa.load(file_path)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return centroid.squeeze().tolist()  # Convert from 2D to 1D list


@mcp.tool()
def spectral_bandwidth(file_path: str) -> list:
    """
    Computes the spectral bandwidth for each frame, measuring the
    range of frequencies in the signal.
    """
    y, sr = librosa.load(file_path)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return bandwidth.squeeze().tolist()


@mcp.tool()
def spectral_rolloff(file_path: str) -> list:
    """
    Computes the roll-off frequency for each frame. The roll-off is
    the frequency below which a specified percentage (e.g., 85%) of
    the total spectral energy lies.
    """
    y, sr = librosa.load(file_path)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return rolloff.squeeze().tolist()


@mcp.tool()
def spectral_contrast(file_path: str) -> list:
    """
    Computes the spectral contrast, which returns
    a 2D matrix (frequency subbands x frames). This function
    returns a nested list for convenience.
    """
    y, sr = librosa.load(file_path)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return contrast.tolist()


@mcp.tool()
def zero_crossing_rate(file_path: str) -> list:
    """
    Calculates the zero-crossing rate for each frame in the audio
    and returns it as a list. The zero-crossing rate is the rate at
    which the signal changes from positive to negative or vice versa.
    """
    y, sr = librosa.load(file_path)
    zcr = librosa.feature.zero_crossing_rate(y)
    return zcr.squeeze().tolist()


@mcp.tool()
def rms_energy(file_path: str) -> list:
    """
    Computes the root-mean-square (RMS) energy for each frame
    in the audio and returns it as a list of values.
    """
    y, sr = librosa.load(file_path)
    rms = librosa.feature.rms(y=y)
    return rms.squeeze().tolist()


@mcp.tool()
def mfcc(file_path: str, n_mfcc: int = 13) -> list:
    """
    Computes Mel-frequency cepstral coefficients (MFCCs). By default,
    it returns a 2D array (n_mfcc x frames) as a nested list.
    """
    y, sr = librosa.load(file_path)
    mfcc_values = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc_values.tolist()


@mcp.tool()
def chroma_stft(file_path: str) -> list:
    """
    Computes a chromagram from a waveform or power spectrogram.
    This returns a 2D array (12 x frames) as a nested list,
    representing the intensity of each pitch class.
    """
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma.tolist()


@mcp.tool()
def onset_times(file_path: str) -> list:
    """
    Detects the onset times (in seconds) of events (e.g., notes, beats)
    in the audio.
    """
    y, sr = librosa.load(file_path)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times_sec = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times_sec.tolist()


###############################################################################
# PROMPT
###############################################################################


@mcp.prompt()
def analyze_audio() -> str:
    """
    Creates a prompt for audio analysis. Feel free to customize
    the text below to explain how users can interact with the tools.
    """
    return (
        "Welcome to the Comprehensive Audio Analysis MCP! Please provide "
        "the path to an audio file and call the tools listed below to extract "
        "various audio features.\n\n"
        "Available tools:\n"
        "- beat(file_path) -> BPM\n"
        "- duration(file_path) -> Audio duration in seconds\n"
        "- beat_frames(file_path) -> Frame indices for each detected beat\n"
        "- beat_times(file_path) -> Beat times in seconds\n"
        "- spectral_centroid(file_path) -> List of spectral centroids\n"
        "- spectral_bandwidth(file_path) -> List of spectral bandwidth values\n"
        "- spectral_rolloff(file_path) -> List of roll-off frequencies\n"
        "- spectral_contrast(file_path) -> Spectral contrast (2D array)\n"
        "- zero_crossing_rate(file_path) -> Zero crossing rates per frame\n"
        "- rms_energy(file_path) -> RMS energy per frame\n"
        "- mfcc(file_path) -> MFCC coefficients (2D array)\n"
        "- chroma_stft(file_path) -> Chroma features (2D array)\n"
        "- onset_times(file_path) -> Detected onset times in seconds\n\n"
        "Example usage:\n"
        ">>> beat('my_audio.wav')\n"
        ">>> mfcc('my_audio.wav', n_mfcc=20)\n"
        ">>> spectral_centroid('my_audio.wav')\n"
    )


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
