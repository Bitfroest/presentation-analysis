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
from scipy import signal
from numpy import NaN, Inf, arange, isscalar, asarray, array
from flask import Flask, Response, render_template, request, redirect, jsonify
from scipy.io import wavfile
import io
import base64
import serial
import colour as c
import threading, queue
from LedEffects import LedEffects
import struct
from usb.core import find as finddev

CHANNELS = 1
RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paFloat32

p = pyaudio.PyAudio()
fulldata = np.array([])
dry_data = np.array([])

buffer_data = np.zeros(1024*50)
audio_data = np.zeros(1024)
data = np.random.random((1024,1024))
pitch_buffer = np.full(2800,np.nan)
last_led_fac = 0

NUM_LEDS = 23
LIVE = False
optimal_voice_level = 85 #65
silencedb = 67 #53
min_freq = 50
max_freq = 500

##calculate header for Adalight
s = struct.pack('>H', NUM_LEDS)
header = b'Ada' + s
first, second = struct.unpack('>BB', s)

n = first ^ second ^ 85
e = bytes([n])

header += e

leds_byte = ""
led = c.Color("blue")

leds_list =[]
for i in range(NUM_LEDS+1):
    leds_list.append(led)

blue = c.Color("blue")

#### Serial Setup
ser = serial.Serial("/dev/ttyACM0", 115200) #/dev/ttyACM0
LedEff = LedEffects(header,ser,NUM_LEDS)
LedEff.chase(1, colour=blue, offcolour=c.Color("black"))

def callback(in_data, frame_count, time_info, flag):
    global b,a,fulldata,dry_data,frames,buffer_data,audio_data
    audio_data = np.fromstring(in_data, dtype=np.float32)
    dry_data = np.append(dry_data,audio_data) #dry_data
    buffer_data = np.append(np.roll(buffer_data, -len(audio_data))[:-1024],audio_data)
    #do processing here
    if LIVE:
        computeLoudness()
    fulldata = np.append(fulldata,audio_data)
    return (audio_data, pyaudio.paContinue)

def start():
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    mics = []
    #for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
    for i in range (0,numdevices):
        if p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
            mics.append(i)

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=mics[0],
                frames_per_buffer=CHUNK,

                stream_callback=callback)
    stream.start_stream()
    return stream

def stop():
    stream.stop_stream()
    stream.close()

def terminate():
    p.terminate()

stream = start()
stop()

def computeLoudness():
    global optimal_voice_level,last_led_fac,max_freq,min_freq,silencedb,ser
    snd = parselmouth.Sound(buffer_data)
    sample_time = 0.5
    #mindip = 'minimum_dip_between_peaks'
    #showtext = 'keep_Soundfiles_and_Textgrids'
    #minpause = 'minimum_pause_duration'

    #use intensity to get threshold
    intensity = snd.to_intensity(time_step=sample_time)
    pitch = snd.to_pitch(time_step=sample_time)

    # estimate noise floor
    minint = np.amin(intensity.values[0])
    # estimate noise max
    maxint = np.amax(intensity.values[0])
    #get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max99int = np.percentile(intensity.values[0], 99)

    # estimate Intensity threshold
    threshold = max99int + silencedb
    threshold2 = maxint - max99int
    threshold3 = silencedb - threshold2
    if threshold < minint:
        threshold = minint

    duration = snd.duration
    minpause = 0.1

    speakingParts = []

    data = gaussian_filter(intensity.values[0],sigma=1)

    pitchFilteredX = []
    pitchFilteredY = []
    for k in range(len(pitch.selected_array['frequency'])):
        if pitch.selected_array['frequency'][k] > min_freq and pitch.selected_array['frequency'][k] < max_freq:
            pitchFilteredY.append(pitch.selected_array['frequency'][k])
            pitchFilteredX.append(k)

    len_speakingParts = 0
    optimal_voice_count = 0
    for i in range(0, len(data)):
        if data[i] > threshold3 and i in pitchFilteredX:
            speakingParts.append((i,data[i]))
            len_speakingParts = len_speakingParts + 1
            if data[i] > optimal_voice_level:
                optimal_voice_count = optimal_voice_count + 1
        else:
            speakingParts.append((i,np.nan))

    #print(speakingParts)
    x = [i[0] for i in speakingParts]
    y = [i[1] for i in speakingParts]

    leds_list = []
    factor = last_led_fac

    if len_speakingParts:
        factor = optimal_voice_count * len_speakingParts/ len(intensity.values[0])
        #last_led_fac = np.append(np.roll(last_led_fac, -1)[:-(len(last_led_fac)-1)],factor)
        #factor = np.average(last_led_fac)
        if factor >= last_led_fac:
            factor = last_led_fac + 0.05
        elif factor < last_led_fac:
            factor = last_led_fac - 0.05
        else:
            factor = last_led_fac
        last_led_fac = factor
    else:
        if factor > 0.4:
            factor = factor - 0.05
        else:
            factor = factor + 0.05
        last_led_fac = factor

    if factor > 1:
        factor = 1
    if factor < 0:
        factor = 0
    for i in range(NUM_LEDS+1):
        r = (1-factor)
        g = factor
        leds_list.append(c.Color(rgb=(r,g, 0)))
    try:
    	ser.write(LedEff.leds_list_to_byte(leds_list))
    except serial.SerialException as e:
        #There is no new data from serial port
        found = False
        ser.close()
        while found:
            if finddev(idVendor=0x2341, idProduct=0x8036):
                found = True
                dev.reset()
                ser = serial.Serial("/dev/ttyACM0", 115200)
        return None
    except TypeError as e:
        #Disconnect of USB->UART occured
        self.port.close()
        return None
    else:
        #Some data was received
        return None
    #print((optimal_voice_count/ len_intensity))
    #if loudness_factor > loudness_threshold:
    #    plt.plot(loudnessX,loudnessY, 'g')
    #else:
    #    plt.plot(loudnessX,loudnessY, 'r')
    #plt.plot(len_min_threshold,min_thresholdZ)
    #plt.plot(len_avg_threshold,avg_thresholdZ)
    #plt.plot(x, intensity_average_filtered)
    #plt.plot(x, gauss_filtered)
    #plt.plot(np.arange(len(buffer_data)),buffer_data)

    #plt.ylim(0,120)

    #plt.pause(0.01)
    #plt.cla()


def peakdet(v, delta, x = None):
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

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
            computeLoudness()
            #print(sum(audio_data))
    except KeyboardInterrupt:
        stream.stop_stream()
        print('interrupted!')
    #while stream.is_active():

    stream.close()
    p.terminate()
