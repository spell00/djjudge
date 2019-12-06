import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


def wav2png(path_to_wav):
    spf = wave.open("test.wav", "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = signal = np.frombuffer(signal, dtype='int16')
    fs = spf.getframerate()

    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)


    Time = np.linspace(0, len(signal) / fs, num=len(signal))

    plt.figure(1)
    plt.title("Signal Wave...")
    plt.plot(Time, signal)
    plt.savefig("output.png")