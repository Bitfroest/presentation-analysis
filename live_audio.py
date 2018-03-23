import pyaudio
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from matplotlib import style
import scipy.signal as signal
import itertools
import sys
import os
import parselmouth
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import butter, lfilter, freqz

#style.use('fivethirtyeight')

CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()
fulldata = np.array([])
dry_data = np.array([])

buffer_data = np.zeros(1024*100)
audio_data = np.zeros(1024)
data = np.random.random((1024,1024))

plt.axis([0, 1024, -1, 1])
plt.ion()
plt.grid(True)
plt.autoscale(enable=True, axis='both', tight=None)
plt.ylim([-1,1])

#fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)
#ax1.set_xlim(0,1024)
#ax1.set_yticks( [-100, -50, 0, 50, 100], minor=False )
#ax1.set_title("Raw Audio Signal")
#im = ax1.imshow(data)

'''
def animate(i):
    global buffer,audio_data,fulldata
    data = audio_data
    xs = len(data)
    ys = data
    if len(data) < 10:
        ys = np.arange(1024)
    ax1.clear()
    #ax1.draw(xs,ys)
    ax1.canvas.draw()
'''

def averageFilter(data,widthSegments):
    length = len(data)
    pitchAverageFilter = []
    for i in range(widthSegments,length-widthSegments):
        sumPitch = 0
        for k in range(i-widthSegments,i+widthSegments):
            sumPitch = sumPitch + data[k]
        sumPitch = sumPitch/(2*widthSegments)
        pitchAverageFilter.append(sumPitch)
    return pitchAverageFilter

def main():
    print(data)
    stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                stream_callback=callback)
    stream.start_stream()
    #ani = animation.FuncAnimation(fig, animate, interval=100)
    #plt.show()

    try:
        while True:
            snd = parselmouth.Sound(buffer_data)
            intensity = snd.to_intensity(time_step=0.01)
            len_intensity = len(intensity)
            len_buffer = len(buffer_data)
            coeff = len_buffer/ len_intensity
            #print(len(intensity.values[0]))
            intensity_average_filtered = averageFilter(intensity.values[0],5)
            x = np.arange(len(intensity_average_filtered))
            for a in range(len(x)):
                x[a] = x[a] * coeff
            min_threshold = np.poly1d([0,0,55]) #66
            len_min_threshold = np.arange(len_buffer)
            min_thresholdZ = min_threshold(len_min_threshold)
            optimal_voice_level = 70
            avg_threshold = np.poly1d([0,0,optimal_voice_level]) # optimal voice loudness
            len_avg_threshold = np.arange(len_buffer)
            avg_thresholdZ = avg_threshold(len_avg_threshold)
            gauss_filtered = gaussian_filter(intensity_average_filtered, sigma=2)
            optimal_voice_count = 0
            for j in range(len(gauss_filtered)):
                if gauss_filtered[j] > optimal_voice_level:
                    optimal_voice_count = optimal_voice_count + 1
            loudness_factor = 100 * (optimal_voice_count/ len_intensity)
            loudness_factor_fct = np.poly1d([0,0,loudness_factor])
            loudnessX = np.arange(len_buffer)
            loudnessY = loudness_factor_fct(loudnessX)
            loudness_threshold = 20
            if loudness_factor > loudness_threshold:
                plt.plot(loudnessX,loudnessY, 'g')
            else:
                plt.plot(loudnessX,loudnessY, 'r')
            plt.plot(len_min_threshold,min_thresholdZ)
            plt.plot(len_avg_threshold,avg_thresholdZ)
            plt.plot(x, intensity_average_filtered)
            plt.plot(x, gauss_filtered)
            plt.plot(np.arange(len(buffer_data)),buffer_data)
            plt.pause(0.01)
            plt.cla()
            #print(sum(audio_data))
    except KeyboardInterrupt:
        stream.stop_stream()
        print('interrupted!')
    #while stream.is_active():

    stream.close()
    p.terminate()

def callback(in_data, frame_count, time_info, flag):
    global b,a,fulldata,dry_data,frames,buffer_data,audio_data
    audio_data = np.fromstring(in_data, dtype=np.float32)
    dry_data = np.append(dry_data,audio_data) #dry_data
    buffer_data = np.append(np.roll(buffer_data, -len(audio_data))[:-1024],audio_data)
    #do processing here
    fulldata = np.append(fulldata,audio_data)
    return (audio_data, pyaudio.paContinue)

main()
