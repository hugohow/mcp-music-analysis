import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


path = "/tmp/TgntkGc5iBo.mp4"

y, sr = librosa.load(path=path)
chroma_cq = librosa.feature.chroma_cqt(y=y)
# print(chroma_cq.shape)

# fig, ax = plt.subplots()

fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
# librosa.display.specshow(chroma_cq, y_axis="chroma", x_axis="tempo", ax=ax[0])
ax.plot(chroma_cq[0])
ax.yaxis.set_major_formatter(librosa.display.ChromaFormatter())
plt.colorbar()
plt.show()

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
