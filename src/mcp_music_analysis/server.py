# server.py

from fastmcp import FastMCP, Image
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import tempfile
import requests
from pytubefix import YouTube

# Create an MCP server with a descriptive name and relevant dependencies
mcp = FastMCP(
    "Music Analysis with librosa",
    dependencies=["librosa", "matplotlib", "numpy", "requests", "pytube"],
    description="An MCP server for analyzing audio files using librosa.",
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


@mcp.tool()
def separate_harmonic_percussive(file_path: str) -> dict:
    """
    Separates the audio signal into harmonic and percussive components
    and returns the length (in samples) of each part.
    """
    y, sr = librosa.load(file_path)
    y_harm, y_perc = librosa.effects.hpss(y)
    return {"harmonic_length": len(y_harm), "percussive_length": len(y_perc)}


@mcp.tool()
def plot_spectrogram(file_path: str) -> str:
    """
    Generates a spectrogram image from the audio signal and saves it as 'spectrogram.png'.
    It won't display automatically; the user must open the resulting path themselves.
    Returns the path to the saved image.
    """
    y, sr = librosa.load(file_path)
    plt.figure()
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.title("Spectrogram")
    plt.colorbar(format="%+2.f dB")
    output_path = os.path.join(tempfile.gettempdir(), "spectrogram.png")
    plt.savefig(output_path)
    plt.close()
    with open(output_path, "rb") as f:
        data = f.read()
    return Image(data=data, format="png")


@mcp.tool()
def plot_waveform(file_path: str) -> str:
    """
    Generates a waveform image from the audio signal and saves it as 'waveform.png'.
    It won't display automatically; the user must open the resulting path themselves.
    """
    y, sr = librosa.load(file_path)
    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    output_path = os.path.join(tempfile.gettempdir(), "waveform.png")
    plt.savefig(output_path)
    plt.close()
    with open(output_path, "rb") as f:
        data = f.read()
    return Image(data=data, format="png")


@mcp.tool()
def plot_chromagram(file_path: str) -> str:
    """
    Generates a chromagram image from the audio signal and saves it as 'chromagram.png'.
    It won't display automatically; the user must open the resulting path themselves.
    """
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    plt.figure()
    librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma")
    plt.title("Chromagram")
    plt.colorbar()
    output_path = os.path.join(tempfile.gettempdir(), "chromagram.png")
    plt.savefig(output_path)
    plt.close()
    with open(output_path, "rb") as f:
        data = f.read()
    return Image(data=data, format="png")


@mcp.tool()
def download_from_url(url: str) -> str:
    """
    Downloads a file from a given URL and returns the path to the downloaded file.
    """

    # mettre une exception si ce n'est pas un fichier audio !
    if not url.endswith(".mp3") and not url.endswith(".wav"):
        raise ValueError(f"URL: {url} is not a valid audio file")

    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(tempfile.gettempdir(), "downloaded_file")
        with open(file_path, "wb") as file:
            file.write(response.content)
        return file_path
    else:
        raise ValueError(f"Failed to download file from URL: {url}")


@mcp.tool()
def download_from_youtube(youtube_url: str) -> str:
    """
    Downloads a file from a given youtube URL and returns the path to the downloaded file.
    """
    yt = YouTube(youtube_url)
    ys = yt.streams.get_audio_only()
    path = ys.download(filename=yt.video_id + ".mp4", output_path=tempfile.gettempdir())
    return path


# @mcp.tool()
# def download_from_youtube(youtube_url: str) -> str:
#     """
#     Downloads a file from a given youtbe URL and returns the path to the downloaded file.
#     """
#     yt = YouTube(youtube_url)
#     yt.streams.filter(only_audio=True).first().download(tempfile.gettempdir())
#     return os.path.join(tempfile.gettempdir(), yt.title + ".mp4")


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
        "Welcome to the Music Analysis MCP! Please provide "
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
        "- separate_harmonic_percussive(file_path) -> Length of harmonic and percussive components\n"
        "- plot_spectrogram(file_path) -> Path to saved spectrogram image\n\n"
        "- plot_waveform(file_path) -> Path to saved waveform image\n\n"
        "- plot_chromagram(file_path) -> Path to saved chromagram image\n\n"
        "- download_from_url(url) -> Path to downloaded file\n\n"
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


def main():
    # Run the MCP server
    print("Running the MCP server")
    mcp.run()
