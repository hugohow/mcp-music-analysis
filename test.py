import librosa
import numpy as np


path = "/tmp/YG4eDMUHjCo.mp4"

begin = 1.5
interval = 0.5

y, sr = librosa.load(path=path)
chroma_cq = librosa.feature.chroma_cqt(y=y)

print(chroma_cq)

print(chroma_cq.shape)

time_frames = np.arange(chroma_cq.shape[1])
time_seconds = librosa.frames_to_time(time_frames)
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

data_list = []

for i, note_name in enumerate(notes):
    # Pour chaque frame temporel (colonne)
    for t, amplitude in zip(time_seconds, chroma_cq[i]):
        if t >= begin and abs(t % interval) < 1e-2:
            data_list.append({"note": note_name, "time": t, "amplitude": amplitude})

print(data_list)
