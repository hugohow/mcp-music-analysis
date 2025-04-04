# idée -> plus faire des fichiers ou des csv que claude peut comprendre au lieu de faire des chemins
# trop compliqués

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


path = "/tmp/TgntkGc5iBo.mp4"

y, sr = librosa.load(path=path)


hop_length = 512
fmin = librosa.note_to_hz("C2")
n_chroma = 12
n_octaves = 7


chroma_cq = librosa.feature.chroma_cqt(
    y=y,
    hop_length=hop_length,
    fmin=fmin,
    n_chroma=n_chroma,
    n_octaves=n_octaves,
)
# Save the chroma_cq to a CSV file
# name = path_audio_time_series_y.split("/")[-1].split(".")[0] + "_chroma_cqt"
# chroma_cq_path = os.path.join(tempfile.gettempdir(), name + ".csv")
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
time_frames = np.arange(chroma_cq.shape[1])
sr = 22050
time_seconds = librosa.frames_to_time(time_frames, sr=sr, hop_length=hop_length)

with open("chroma.csv", "w") as f:
    f.write("note,time,amplitude\n")
    for i, note in enumerate(notes):
        for t_index, amplitude in enumerate(chroma_cq[i]):
            t = time_seconds[t_index]
            f.write(f"{note},{t},{amplitude}\n")
# (8303584,)


# chroma = librosa.feature.chroma_cqt(y=y, sr=sr)


# Sa = librosa.note_to_hz("F2")


# # Sa = librosa.note_to_hz("F4")
# C = librosa.cqt(y=y, sr=sr)
# fig, ax = plt.subplots()

# C_db = librosa.amplitude_to_db(np.abs(C))
# img = librosa.display.specshow(C_db, y_axis="cqt_svara", x_axis="time", ax=ax, Sa=Sa)
# ax.set(title="Constant-Q power spectrum", ylim=[Sa, 3 * Sa])


# fig.colorbar(img, ax=ax, format="%+2.0f dB")

# plt.show()


# chroma_cq = librosa.feature.chroma_cqt(y=y, hop_length=22050)

# time_frames = np.arange(chroma_cq.shape[1])
# time_seconds = librosa.frames_to_time(time_frames, hop_length=22050)
# notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# data_list = []

# # Pour chaque note (ligne)
# for i, note_name in enumerate(notes):
#     # Pour chaque frame temporel (colonne)
#     for t, amplitude in zip(time_seconds, chroma_cq[i]):
#         data_list.append({"note": note_name, "time": t, "amplitude": amplitude})


# print(data_list)

# chroma_cq = librosa.feature.chroma_cqt(y=y)
# print(chroma_cq.shape)
# librosa.display.specshow(chroma_cq, y_axis="chroma", x_axis="tempo", key="F:dorian")
# plt.colorbar()
# plt.title("Chromagram")
# plt.tight_layout()
# plt.show()
# # (12, 16218)


# D = librosa.stft(y)  # STFT of y
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# fig, ax = plt.subplots()
# img = librosa.display.specshow(S_db, x_axis="time", y_axis="log", ax=ax)
# ax.set(title="Using a logarithmic frequency axis")
# fig.colorbar(img, ax=ax, format="%+2.f dB")
# plt.show()

# comment faire pour afficher juste la chroma de DO, la première

# fig, ax = plt.subplots()

# fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
# # librosa.display.specshow(chroma_cq, y_axis="chroma", x_axis="tempo", ax=ax[0])
# ax.plot(chroma_cq[0])
# ax.yaxis.set_major_formatter(librosa.display.ChromaFormatter())
# plt.colorbar()
# plt.show()

# ax.plot(chroma_cq[0])
# ax.yaxis.set_m

# # (16218,)
# # show the first chroma
# librosa.display.specshow(chroma_cq, y_axis="chroma", x_axis="tempo")
# plt.colorbar()
# plt.title("Chromagram for the first chroma")
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(10, 4))
# librosa.display.specshow(chroma_cq, y_axis="chroma", x_axis="time")
# plt.colorbar()
# plt.title("Chromagram")
# plt.tight_layout()
# plt.show()


# fig, axs = plt.subplots(nrows=12, sharex=True, figsize=(10, 12))
# for i in range(12):
#     img = librosa.display.specshow(chroma_cq[1, :], x_axis="time", ax=axs[i], sr=sr)
#     axs[i].set(title=f"Chroma bin {i}")
#     fig.colorbar(img, ax=axs[i])

# plt.tight_layout()
# plt.show()
# take the first
# chroma_cq
# img1 = librosa.display.specshow(chroma_cq[0], y_axis="chroma", x_axis="time", ax=ax[0])
# fig.colorbar(img1, ax=ax)
# ax[0].set(title="Chromagram")
# plt.show()
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(chroma_cq, y_axis="chroma", x_axis="time")
# plt.colorbar()
# plt.title("Chromagram")
# plt.tight_layout()
# plt.show()
