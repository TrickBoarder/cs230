import numpy as np     
import sys
from pylab import *
import wave
from os import listdir
from os.path import isfile,join
import sounddevice as sd
import soundfile as sf

#Directory Path to test out LA-file
dir_path = 'C:\\Users\\15126\\Downloads\ASVspoof\\LA\\LA\\ASVspoof2019_LA_train\\flac'

#Put all files into list for easy cycle
list_of_files = [f for f in listdir(dir_path) if isfile(join(dir_path,f))]

#Test First file
data, samplerate = sf.read(dir_path+"\\"+list_of_files[0],dtype='float32') 
data, fs = sf.read(filename, dtype='float32')  

# Extract data and sampling rate from file
sd.play(data, samplerate)
status = sd.wait()  # Wait until file is done playing

#Example Image of Audio File (For Jupyter Notebook)
subplot(211)
plot(data)
subplot(212)
spectrogram = specgram(data, Fs = samplerate, scale_by_freq=True,sides='default')

def save_spectogram(file_name,label):
    """ Save Spectrogram to Local """
    clip, sample_rate = sf.read(dir_path+"/"+file_name)
    file_name = file_name.split('/')[1]
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = "path/to/audio/spectrograms/"+label+"/"+file_name.replace('.flac','.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close('all')
