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

app = Flask(__name__)
app.config.from_object(__name__)

email_addresses = []
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paFloat32

p = pyaudio.PyAudio()
fulldata = np.array([])
dry_data = np.array([])

buffer_data = np.zeros(1024*100)
audio_data = np.zeros(1024)
data = np.random.random((1024,1024))
pitch_buffer = np.full(2800,np.nan)
last_led_fac = np.zeros(100)

NUM_LEDS = 21
LIVE = False
optimal_voice_level = 76

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
ser = serial.Serial("/dev/ttyACM0", 115200)
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

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)

@app.route('/')
def index():
    content = get_file('index.html')
    return Response(content, mimetype="text/html")

@app.route('/signup', methods = ['POST'])
def signup():
    email = request.form['email']
    email_addresses.append(email)
    print(email_addresses)
    return redirect('/')

@app.route('/record', methods = ['GET'])
def record():
    global stream,p
    status = request.args.get('status')
    fileName = request.args.get('file')
    if status == 'start':
        stream = start()
    else:
        stop()
        save(fileName)
    return render_template('status.html', status=status, fileName=fileName)

@app.route('/emails.html')
def emails():
    return render_template('emails.html', email_addresses=email_addresses)

plt.axis([0, 1024, -1, 1])
plt.ion()
plt.grid(True)
plt.autoscale(enable=True, axis='both', tight=None)
plt.ylim([-1,1])

@app.route('/loudness', methods = ['GET'])
def generateLoudnessGraph():
    snd = parselmouth.Sound(fulldata)
    intensity = snd.to_intensity(time_step=0.01)
    len_intensity = len(intensity)
    len_buffer = len(buffer_data)
    coeff = len_buffer/ len_intensity
    plt.figure(figsize=(16, 4), dpi = 300)
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
    #plt.plot(x, intensity_average_filtered)
    plt.plot(x, gauss_filtered)
    #plt.plot(np.arange(len(buffer_data)),buffer_data)

    plt.ylim(40,120)

    f = io.BytesIO()
    plt.savefig(f, format="png", facecolor=(0.95, 0.95, 0.95))
    encoded_img = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
    f.close()
    # And here with the JsonResponse you catch in the ajax function in your html triggered by the click of a button
    #return jsonify('<img src="data:image/png;base64,'+encoded_img+'" />')
    return render_template('loudness.html', loudness=encoded_img)

def computeLoudness():
    global optimal_voice_level,last_led_fac
    snd = parselmouth.Sound(buffer_data)
    silencedb = 60
    sample_time = 0.1
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
        if pitch.selected_array['frequency'][k] > 50 and pitch.selected_array['frequency'][k] < 300:
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

    if len_speakingParts:
        factor = optimal_voice_count / (len(intensity.values[0])/len_speakingParts)
        last_led_fac = np.append(np.roll(last_led_fac, -1)[:-(len(last_led_fac)-1)],factor)
        factor = np.average(last_led_fac)
    else:
        factor = 0.5

    for i in range(NUM_LEDS+1):
        r = 1 - min(factor,1)
        g = min(factor,1)
        leds_list.append(c.Color(rgb=(r,g, 0)))

    ser.write(LedEff.leds_list_to_byte(leds_list))
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

@app.route('/live', methods = ['GET'])
def live():
    global stream,LIVE
    stream = start()
    param = request.args.get('param')
    if param == 'start':
        LIVE = True
    else:
        LIVE = False
    return redirect('/')



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

def save(fileName):
    filename = './audio/' + fileName + '.wav'
    wavfile.write(filename, RATE, fulldata)

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

def computeSpeed():
    silencedb = 67
    sample_time = 0.01
    min_freq = 50
    max_freq = 500
    #mindip = 'minimum_dip_between_peaks'
    #showtext = 'keep_Soundfiles_and_Textgrids'
    #minpause = 'minimum_pause_duration'

    #use intensity to get threshold
    snd = parselmouth.Sound(buffer_data)
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
    #plot stuff
    #plt.figure(figsize=(12,4))
    '''
    plt.plot(intensity.values[0] , 'c')
    p = np.poly1d([0,0,minint])
    x = np.arange(len(intensity.values[0]))
    z = p(x)
    plt.plot(x,z,'y')
    p = np.poly1d([0,0,maxint])
    x = np.arange(len(intensity.values[0]))
    z = p(x)
    plt.plot(x,z,'g')
    p = np.poly1d([0,0,max99int])
    x = np.arange(len(intensity.values[0]))
    z = p(x)
    plt.plot(x,z,'b')
    p = np.poly1d([0,0,threshold3])
    x = np.arange(len(intensity.values[0]))
    z = p(x)
    plt.plot(x,z,'g')
    '''

    duration = snd.duration
    minpause = 0.1

    speakingParts = []

    data = gaussian_filter(intensity.values[0],sigma=1)

    pitchFilteredX = []
    for k in range(len(pitch.selected_array['frequency'])):
        if pitch.selected_array['frequency'][k] > min_freq and pitch.selected_array['frequency'][k] < max_freq:
            pitchFilteredX.append(k)


    for i in range(0, len(data)):
        if data[i] > threshold3 and i in pitchFilteredX:
            speakingParts.append((i,data[i]))
        else:
            speakingParts.append((i,np.nan))

    #print(speakingParts)
    x = [i[0] for i in speakingParts]
    y = [i[1] for i in speakingParts]
    plt.plot(x,y,'r')

    '''
    lastVoicePart = 0
    splittedSpeakingParts = []
    tmp = []
    for i in range(0,len(speakingParts)):
        if speakingParts[i][0] > speakingParts[lastVoicePart][0] + 1:
            splittedSpeakingParts.append(tmp)
            tmp = []
        else:
            tmp.append(speakingParts[i][0])
        lastVoicePart = i

    print(splittedSpeakingParts)
    '''

    lastVoicePart = 0
    splittedSpeakingParts = []
    tmp = []

    #print(speakingParts)

    for i in range(0,len(speakingParts) - 1):
        if np.isnan(speakingParts[i+1][1]):
            splittedSpeakingParts.append(tmp)
            tmp = []
        else:
            tmp.append(speakingParts[i+1][1])

    splittedSpeakingParts[:] = [item for item in splittedSpeakingParts if item != []]
    #print(splittedSpeakingParts)

    # remove all parts with len < 1 of splittedSpeakingParts

    numwords = 0
    for i in range(len(splittedSpeakingParts)):
        maxtab, mintab = peakdet(splittedSpeakingParts[i],0.5)
        #print(splittedSpeakingParts[i])
        if len(maxtab) > 0:
            numwords = numwords + len(maxtab)
            #plt.figure(figsize=(12,4))
            #plt.plot(splittedSpeakingParts[i] , 'g')
            #plt.scatter(np.array(maxtab)[:,0], np.array(maxtab)[:,1], color='blue')
            #print(array(maxtab))

    # shorten duration to start from first word and end with last word
    duration_shorten = 0
    for i in range(len(y)):
        if np.isnan(y[i]):
            duration_shorten = duration_shorten + 1

    duration_onlySpeaking = ((duration/sample_time) - duration_shorten) * sample_time

    #print(duration/60)
    #print(numwords/duration*60)
    #print(numwords)
    speed = numwords/duration*60
    speed_onlySpeaking = numwords/duration_onlySpeaking*60

    p = np.poly1d([0,0,speed_onlySpeaking])
    x = np.arange(len(intensity.values[0]))
    z = p(x)
    plt.plot(x,z,'y')

    plt.ylim(0,1000)

    plt.pause(0.1)
    plt.cla()

def computeMonotony():
    global pitch_buffer
    min_freq = 50
    max_freq = 500
    sample_time = 0.01
    snd = parselmouth.Sound(buffer_data)
    pitch = snd.to_pitch(time_step=sample_time)

    pitchFilteredX = []
    pitchFilteredY = []
    for k in range(len(pitch.selected_array['frequency'])):
        if pitch.selected_array['frequency'][k] > min_freq and pitch.selected_array['frequency'][k] < max_freq:
            pitchFilteredY.append(pitch.selected_array['frequency'][k])
            pitchFilteredX.append(k)
        else:
            pitchFilteredX.append(k)
            pitchFilteredY.append(np.nan)

    pitch_buffer = np.append(np.roll(pitch_buffer, -len(pitchFilteredY))[:-len(pitchFilteredY)],pitchFilteredY)
    pitch_std = np.nanstd(pitch_buffer)
    pitch_avg = np.nanmean(pitch_buffer)
    #pitch_avg = sum(pitch.selected_array['frequency'])/len(pitch.selected_array['frequency'])
    pdq = pitch_std/pitch_avg
    p = np.poly1d([0,0,pdq*max_freq])
    x = np.arange(len(pitchFilteredX))
    z = p(x)
    plt.plot(x,z)

    #plt.plot(pitch.selected_array['frequency'])
    #plt.plot(gaussian_filter(pitch.selected_array['frequency'], sigma=3), 'y')

    #plt.plot(butter_lowpass_filter(pitch.selected_array['frequency'], cutoff, fs, order ), 'g')
    # Make a new figure

    plt.plot(pitchFilteredX, pitchFilteredY)
    plt.plot(pitchFilteredX, gaussian_filter(pitchFilteredY, sigma=3), 'y')
    plt.ylim(0,500)

    plt.pause(0.01)
    plt.cla()

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
            computeSpeed()
            #print(sum(audio_data))
    except KeyboardInterrupt:
        stream.stop_stream()
        print('interrupted!')
    #while stream.is_active():

    stream.close()
    p.terminate()

if __name__ == '__main__':
    app.run()
