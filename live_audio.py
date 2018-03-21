import pyaudio
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from matplotlib import style
import scipy.signal as signal
import itertools

#style.use('fivethirtyeight')

CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()
fulldata = np.array([])
dry_data = np.array([])

buffer = np.zeros(1024)
audio_data = list()

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlim(0,1024)
ax1.set_yticks( [-100, -50, 0, 50, 100],minor=False )
ax1.set_title("Raw Audio Signal")

def animate(i):
    global buffer,audio_data
    xs = i * np.arange(1024)
    ys = buffer + audio_data
    if len(buffer) < 10:
        ys = np.arange(1024)
    ax1.clear()
    ax1.plot(xs,ys)
    buffer = buffer + audio_data

def main():
    stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                output=False,
                input=True,
                stream_callback=callback)
    stream.start_stream()
    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()
    while stream.is_active():
        time.sleep(10)
        stream.stop_stream()
    stream.close()

    p.terminate()

def callback(in_data, frame_count, time_info, flag):
    global b,a,fulldata,dry_data,frames,buffer,audio_data 
    audio_data = np.fromstring(in_data, dtype=np.float32)
    dry_data = np.append(dry_data,audio_data)
    #do processing here
    fulldata = np.append(fulldata,audio_data)
    return (audio_data, pyaudio.paContinue)

main()