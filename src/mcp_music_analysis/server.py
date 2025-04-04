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
import soundfile as sf

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
def load(
    file_path: str,
    offset: float = 0.0,
    duration: float = None,
) -> dict:
    """
    Loads an audio file and returns the path to the audio time series
    Offset and duration are optional, in seconds.
    Be careful, you will never know the name of the song.
    """
    y, sr = librosa.load(path=file_path, offset=offset, duration=duration)

    # stock y inside a csv file
    name = file_path.split("/")[-1].split(".")[0] + "_y"
    y_path = os.path.join(tempfile.gettempdir(), name + ".csv")
    np.savetxt(y_path, y, delimiter=";")

    D = librosa.stft(y)
    harmonics, percussion = librosa.decompose.hpss(D)
    # Save the harmonic and percussive components to separate files
    # name_harmonic = file_path.split("/")[-1].split(".")[0] + "_harmonic"
    # name_percussive = file_path.split("/")[-1].split(".")[0] + "_percussive"
    # harmonic_path = os.path.join(tempfile.gettempdir(), name_harmonic + ".csv")
    # percussive_path = os.path.join(tempfile.gettempdir(), name_percussive + ".csv")
    # np.savetxt(harmonic_path, harmonics, delimiter=";")
    # np.savetxt(percussive_path, percussion, delimiter=";")

    return {
        "y_path": y_path,
        # "y_harmonic_path": harmonic_path,
        # "y_percussive_path": percussive_path,
    }


@mcp.tool()
def get_duration(path_audio_time_series_y: str) -> float:
    """
    Returns the total duration (in seconds) of the given audio time series.
    """
    y = np.loadtxt(path_audio_time_series_y, delimiter=";")
    return librosa.get_duration(y=y)


@mcp.tool()
def tempo(
    path_audio_time_series_y: str,
    hop_length: int = 512,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    ac_size: float = 8.0,
    max_tempo: float = 320.0,
) -> float:
    """
    Estimates the tempo (in BPM) of the given audio time series using librosa.
    Offset and duration are optional, in seconds.
    """
    y = np.loadtxt(path_audio_time_series_y, delimiter=";")
    tempo = librosa.feature.tempo(
        y=y,
        hop_length=hop_length,
        start_bpm=start_bpm,
        std_bpm=std_bpm,
        ac_size=ac_size,
        max_tempo=max_tempo,
    )
    return tempo


@mcp.tool()
def chroma_cqt(
    path_audio_time_series_y: str,
    hop_length: int = 512,
    fmin: float = None,
    n_chroma: int = 12,
    n_octaves: int = 7,
) -> str:
    """
    Computes the chroma CQT of the given audio time series using librosa.
    The chroma CQT is a representation of the audio signal in terms of its
    chromatic content, which is useful for music analysis.
    The chroma CQT is computed using the following parameters:
    - path_audio_time_series_y: The path to the audio time series (CSV file).
        It's sometimes better to take harmonics only
    - hop_length: The number of samples between frames.
    - fmin: The minimum frequency of the chroma feature.
    - n_chroma: The number of chroma bins (default is 12).
    - n_octaves: The number of octaves to include in the chroma feature.
    The chroma CQT is saved to a CSV file with the following columns:
    - note: The note name (C, C#, D, etc.).
    - time: The time position of the note in seconds.
    - amplitude: The amplitude of the note at that time.
    The path to the CSV file is returned.
    """
    y = np.loadtxt(path_audio_time_series_y, delimiter=";")
    chroma_cq = librosa.feature.chroma_cqt(
        y=y,
        hop_length=hop_length,
        fmin=fmin,
        n_chroma=n_chroma,
        n_octaves=n_octaves,
    )
    # Save the chroma_cq to a CSV file
    name = path_audio_time_series_y.split("/")[-1].split(".")[0] + "_chroma_cqt"
    chroma_cq_path = os.path.join(tempfile.gettempdir(), name + ".csv")
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    time_frames = np.arange(chroma_cq.shape[1])
    time_seconds = librosa.frames_to_time(time_frames, hop_length=hop_length)

    with open(chroma_cq_path, "w") as f:
        f.write("note,time,amplitude\n")
        for i, note in enumerate(notes):
            for t_index, amplitude in enumerate(chroma_cq[i]):
                t = time_seconds[t_index]
                f.write(f"{note},{t},{amplitude}\n")
    # Return the path to the CSV file
    return chroma_cq_path


@mcp.tool()
def mfcc(
    path_audio_time_series_y: str,
) -> str:
    """
    Computes the MFCC of the given audio time series using librosa.
    The MFCC is a representation of the audio signal in terms of its
    spectral content, which is useful for music analysis.
    The MFCC is computed using the following parameters:
    - path_audio_time_series_y: The path to the audio time series (CSV file).
        It's sometimes better to take harmonics only
    """
    y = np.loadtxt(path_audio_time_series_y, delimiter=";")
    mfcc = librosa.feature.mfcc(y=y)
    # Save the mfcc to a CSV file
    name = path_audio_time_series_y.split("/")[-1].split(".")[0] + "_mfcc"
    mfcc_path = os.path.join(tempfile.gettempdir(), name + ".csv")
    np.savetxt(mfcc_path, mfcc, delimiter=";")
    # Return the path to the CSV file
    return mfcc_path


@mcp.tool()
def beat_track(
    path_audio_time_series_y: str,
    hop_length: int = 512,
    start_bpm: float = 120,
    tightness: int = 100,
    units: str = "frames",
) -> str:
    """
    Computes the beat track of the given audio time series using librosa.
    The beat track is a representation of the audio signal in terms of its
    rhythmic content, which is useful for music analysis.
    The beat track is computed using the following parameters:
    - hop_length: The number of samples between frames.
    - start_bpm: The initial estimate of the tempo (in BPM).
    - tightness: The tightness of the beat tracking (default is 100).
    - units: The units of the beat track (default is "frames"). It can be frames, samples, time.
    """
    y = np.loadtxt(path_audio_time_series_y, delimiter=";")
    tempo, beats = librosa.beat.beat_track(
        y=y,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=tightness,
        units=units,
    )
    return {
        "tempo": tempo,
        "beats": beats,
    }


@mcp.tool()
def download_from_url(url: str) -> str:
    """
    Downloads a file from a given URL and returns the path to the downloaded file.
    Be careful, you will never know the name of the song.
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
    Be careful, you will never know the name of the song.
    """
    yt = YouTube(youtube_url)
    ys = yt.streams.get_audio_only()
    path = ys.download(filename=yt.video_id + ".mp4", output_path=tempfile.gettempdir())

    return path


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
        "- load(file_path: str, offset: float = 0.0, duration: float = None) -> dict\n"
        "- tempo(path_audio_time_series_y: str, hop_length: int = 512, start_bpm: float = 120, "
        "std_bpm: float = 1.0, ac_size: float = 8.0, max_tempo: float = 320.0) -> float\n"
        "- chroma_cqt(path_audio_time_series_y: str, hop_length: int = 512, fmin: float = None, "
        "n_chroma: int = 12, n_octaves: int = 7) -> str\n"
        "- beat_track(path_audio_time_series_y: str, hop_length: int = 512, start_bpm: float = 120, "
        "tightness: int = 100, units: str = 'frames') -> dict\n"
        "- get_duration(path_audio_time_series_y: str) -> float\n"
        "- download_from_url(url: str) -> str\n"
        "- download_from_youtube(youtube_url: str) -> str\n"
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
