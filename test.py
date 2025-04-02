import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


path = "/tmp/TgntkGc5iBo.mp4"

y, sr = librosa.load(path=path)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

# beats contains the frame indices of each detected beat
# for synchronization and visualization, we'll need to expand this
# to cover the limits of the data.  This can be done as follows:
beats = librosa.util.fix_frames(beats, x_min=0)

# Now beat-synchronize the chroma features
chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)

# For visualization, we can convert to time (in seconds)
beat_times = librosa.frames_to_time(beats)

# We'll plot the synchronized and unsynchronized features next
# to each other

fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(
    chroma, y_axis="chroma", x_axis="time", ax=ax[0], key="Eb:maj"
)
ax[0].set(title="Uniform time sampling")
ax[0].label_outer()

librosa.display.specshow(
    chroma_sync,
    y_axis="chroma",
    x_axis="time",
    x_coords=beat_times,
    ax=ax[1],
    key="Eb:maj",
)
ax[1].set(title="Beat-synchronous sampling")
fig.colorbar(img, ax=ax)

# For clarity, we'll zoom in on a 15-second patch
# ax[1].set(xlim=[10, 25])
plt.show()

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
