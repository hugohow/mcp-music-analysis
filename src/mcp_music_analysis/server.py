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
def get_tempo(
    file_path: str,
    offset: float = 0.0,
    duration: float = None,
) -> float:
    """
    Estimates the tempo (in BPM) of the given audio file using librosa.
    Offset and duration are optional, in seconds.

    It's better to have only percussive sounds.
    """
    y, sr = librosa.load(path=file_path, offset=offset, duration=duration)
    tempo, _ = librosa.beat.beat_track(y=y)
    return tempo


@mcp.tool()
def get_beats(
    file_path: str,
    offset: float = 0.0,
    duration: float = None,
) -> list:
    """
    Returns a list of time positions (in seconds) of the detected beats.
    Offset and duration are in seconds.

    It's better to have only percussive sounds.
    """
    y, sr = librosa.load(path=file_path, offset=offset, duration=duration)
    tempo, frames = librosa.beat.beat_track(y=y)
    times = librosa.frames_to_time(frames)
    return times.tolist()


@mcp.tool()
def get_chroma(
    file_path: str,
    offset: float = 0.0,
    duration: float = None,
    fmin: float = None,
    n_octaves: int = 7,
    interval: float = 1.0,
) -> list:
    """
    Return the result of the chroma feature of the audio file.
    Offset and duration are in seconds.
    fmin is the minimum frequency of the chroma feature.
    n_octaves is the number of octaves to include in the chroma feature.
    interval is the time interval (in seconds) at which to sample the chroma feature.

    It's better to have only harmonic sounds.
    """
    y, sr = librosa.load(path=file_path, offset=offset, duration=duration)
    # 1 trame 1 second
    hop_length = int(22050 * interval)
    chroma_cq = librosa.feature.chroma_cqt(
        y=y, fmin=fmin, n_octaves=n_octaves, hop_length=hop_length
    )

    time_frames = np.arange(chroma_cq.shape[1])
    time_seconds = librosa.frames_to_time(time_frames, hop_length=hop_length)
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    data_list = []

    # Pour chaque note (ligne)
    for i, note_name in enumerate(notes):
        # Pour chaque frame temporel (colonne)
        for t, amplitude in zip(time_seconds, chroma_cq[i]):
            data_list.append({"note": note_name, "time": t, "amplitude": amplitude})

    return data_list


@mcp.tool()
def show_chroma(
    file_path: str,
    offset: float = 0.0,
    duration: float = None,
    fmin: float = None,
    n_octaves: int = 7,
    on_beats: bool = True,
) -> str:
    """
    Display the chroma feature of the audio file.
    Offset and duration are in seconds.
    fmin is the minimum frequency of the chroma feature.
    n_octaves is the number of octaves to include in the chroma feature.

    It's better to have only harmonic sounds.


    It's just for visualisation, but use get_chroma for analysis.
    """
    y, sr = librosa.load(path=file_path, offset=offset, duration=duration)
    chroma_cq = librosa.feature.chroma_cqt(y=y, fmin=fmin, n_octaves=n_octaves)
    plt.figure(figsize=(10, 4))
    if on_beats:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beats = librosa.util.fix_frames(beats, x_min=0)
        chroma_sync = librosa.util.sync(chroma_cq, beats, aggregate=np.median)
        beat_times = librosa.frames_to_time(beats)
        librosa.display.specshow(
            chroma_sync,
            y_axis="chroma",
            x_axis="time",
            x_coords=beat_times,
            key="C:maj",
        )
    else:
        librosa.display.specshow(chroma_cq, y_axis="chroma", x_axis="tempo")
    plt.colorbar()
    plt.title("Chromagram")
    plt.tight_layout()

    image_path = os.path.join(tempfile.gettempdir(), "chroma.png")
    plt.savefig(image_path)
    plt.close()

    with open(image_path, "rb") as f:
        data = f.read()
    return Image(data=data, format="png")


# @mcp.tool()
# def show_chroma_covariance(
#     file_path: str,
#     offset: float = 0.0,
#     duration: float = None,
#     fmin: float = None,
#     n_octaves: int = 7,
# ) -> str:
#     """
#     Display the chroma covariance feature of the audio file.
#     Offset and duration are in seconds.
#     fmin is the minimum frequency of the chroma feature.
#     n_octaves is the number of octaves to include in the chroma feature.

#     It's better to have only harmonic sounds.
#     """
#     y, sr = librosa.load(path=file_path, offset=offset, duration=duration)
#     chroma_cq = librosa.feature.chroma_cqt(y=y, fmin=fmin, n_octaves=n_octaves)

#     ccov = np.cov(chroma_cq)
#     fig, ax = plt.subplots()
#     img = librosa.display.specshow(
#         ccov, y_axis="chroma", x_axis="chroma", key="C:maj", ax=ax
#     )
#     ax.set(title="Chroma covariance")
#     fig.colorbar(img, ax=ax)
#     image_path = os.path.join(tempfile.gettempdir(), "chroma_covariance.png")
#     plt.savefig(image_path)
#     plt.close()

#     with open(image_path, "rb") as f:
#         data = f.read()
#     return Image(data=data, format="png")


@mcp.tool()
def separate_harmonics_percussion(
    file_path: str,
    offset: float = 0.0,
    duration: float = None,
) -> str:
    """
    Separate the harmonics and percussions of the audio file.
    """
    y, sr = librosa.load(path=file_path, offset=offset, duration=duration)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    name = file_path.split("/")[-1].split(".")[0]
    harmonic_path = os.path.join(tempfile.gettempdir(), name + "_harmonic.wav")
    percussive_path = os.path.join(tempfile.gettempdir(), name + "_percussive.wav")

    sf.write(harmonic_path, y_harmonic, sr)
    sf.write(percussive_path, y_percussive, sr)

    return harmonic_path, percussive_path


# @mcp.tool()
# def onset_detect(file_path: str, offset: float = 0.0, duration: float = None) -> list:
#     """
#     Detects onsets in the audio file. Returns a list of time positions (in seconds) of the detected onsets.
#     Offset and duration are in seconds.
#     """
#     y, sr = librosa.load(path=file_path, offset=offset, duration=duration)
#     onsets = librosa.onset.onset_detect(y=y, units="time")
#     return onsets.tolist()


@mcp.tool()
def get_duration(file_path: str) -> float:
    """
    Returns the total duration (in seconds) of the given audio file.
    """
    y, sr = librosa.load(path=file_path)
    return librosa.get_duration(y=y)


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
    Don't try to identify the song, you cannot do it, it's too hard.
    """
    yt = YouTube(youtube_url)
    ys = yt.streams.get_audio_only()
    path = ys.download(filename=yt.video_id + ".mp4", output_path=tempfile.gettempdir())

    y, sr = librosa.load(path=path)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    name = path.split("/")[-1].split(".")[0]
    harmonic_path = os.path.join(tempfile.gettempdir(), name + "_harmonic.wav")
    percussive_path = os.path.join(tempfile.gettempdir(), name + "_percussive.wav")

    sf.write(harmonic_path, y_harmonic, sr)
    sf.write(percussive_path, y_percussive, sr)

    return {"original": path, "harmonic": harmonic_path, "percussive": percussive_path}


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
        "- get_tempo(file_path: str, offset: float = 0.0, duration: float = None) -> float\n"
        "- get_beats(file_path: str, offset: float = 0.0, duration: float = None) -> list\n"
        "- get_chroma(file_path: str, offset: float = 0.0, duration: float = None, fmin: float = None, n_octaves: int = 7, interval: float = 1.0) -> list\n"
        "- show_chroma(file_path: str, offset: float = 0.0, duration: float = None, fmin: float = None, n_octaves: int = 7) -> str\n"
        "- separate_harmonics_percussion(file_path: str, offset: float = 0.0, duration: float"
        "- get_duration(file_path: str) -> float\n"
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
