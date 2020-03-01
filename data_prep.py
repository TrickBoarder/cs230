import numpy as np     
import sys
import soundfile as sf  
from pylab import *
import wave
from os import listdir
from os.path import isfile,join
import librosa
import librosa.display

#Get Target Information
dev_path = 'C:\\Users\\15126\\Downloads\ASVspoof\\LA\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.dev.trl.txt'
train_path = 'C:\\Users\\15126\\Downloads\ASVspoof\\LA\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt'
eval_path = 'C:\\Users\\15126\\Downloads\ASVspoof\\LA\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.eval.trl.txt'

def get_asvspoof_file_names(target_path):
    target_dict = dict()
    with open(target_path,'r') as f:
        for lines in f:
            line = lines.strip("\n")
            line_split=line.split(" ")
            identifier = line_split[0]
            file_path = line_split[1]
            if line_split[4] == 'bonafide':
                target = 0
            else:
                target = 1
            target_dict[file_path] = [identifier,target]
    return target_dict

train_dict = get_asvspoof_file_names(train_path)
dev_dict = get_asvspoof_file_names(dev_path)
eval_dict = get_asvspoof_file_names(eval_path)

# file_name = 'C:\\Users\\15126\\Downloads\ASVspoof\\LA\\LA\\ASVspoof2019_LA_dev\\flac\\LA_D_1448869.flac'    
eval_dir_path = 'C:\\Users\\15126\\Downloads\ASVspoof\\LA\\LA\\ASVspoof2019_LA_eval\\flac'
train_dir_path = 'C:\\Users\\15126\\Downloads\ASVspoof\\LA\\LA\\ASVspoof2019_LA_train\\flac'
dev_dir_path = 'C:\\Users\\15126\\Downloads\ASVspoof\\LA\\LA\\ASVspoof2019_LA_dev\\flac'


def generate_mel(file_name
                ,n_fft = 1024
                ,hop_length = 128
                ,n_mels = 128
                ,f_min = 20
                ,f_max = 8000
                ,sample_rate = 16000):
    clip, sample_rate = sf.read(file_name)
    mel_spec = librosa.feature.melspectrogram(clip
                                          , n_fft=n_fft
                                          , hop_length=hop_length
                                          , n_mels=n_mels
                                          , sr=sample_rate
                                          , power=1.0
                                          , fmin=f_min
                                          , fmax=f_max)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

    return mel_spec_db

def create_partition(file_names
                     ,dir_path
                     ,output_path
                     ,by=5000
                     ):
    import pickle
    cnt=0
    if cnt % by == 0:
        with open(output_path+"_"+str(cnt) + ".pkl",'wb') as f:
            pickle.dump(mel_dict, f)
            mel_dict=dict()
    mel_dict={}
    for key in file_names:
        mel = generate_mel(file_name = dir_path + '\\'+key+'.flac')
        mel_dict[key] = dict()
        mel_dict[key]['data'] = mel
        mel_dict[key]['target'] = file_names[key]
    with open(output_path,'wb') as f:
        pickle.dump(mel_dict, f)
        
# create_partition(file_names = train_dict,dir_path =train_dir_path,output_path = 'C:\\Users\\15126\\Desktop\\stanford\\train')   
# create_partition(file_names = dev_dict,dir_path = dev_dir_path,output_path = 'C:\\Users\\15126\\Desktop\\stanford\\dev')
# create_partition(file_names = eval_dict,dir_path = eval_dir_path,output_path = 'C:\\Users\\15126\\Desktop\\stanford\\eval') 

keys = train_dict.keys()
mel = generate_mel(file_name = train_dir_path + "\\"+'LA_T_1138215.flac')
mel.shape


def create_dummy_dataset(file_names
                     ,dir_path
                     ,output_path
                     ,by=150
                     ):
    import pickle
    import random
    mel_dict={}
    cnt=1

    key_list = list(file_names.keys())
    random.shuffle(key_list)
    for key in key_list:
        if cnt % by == 0:
            with open(output_path+"_"+str(cnt) + ".pkl",'wb') as f:
                pickle.dump(mel_dict, f)
            
            return mel_dict
        
        cnt+=1
        mel = generate_mel(file_name = dir_path + '\\'+key+'.flac')
        mel_dict[key] = dict()
        mel_dict[key]['data'] = mel
        mel_dict[key]['target'] = file_names[key]
    with open(output_path,'wb') as f:
        pickle.dump(mel_dict, f)
        
    
test_dummy = create_dummy_dataset(file_names = dev_dict
                 ,dir_path = dev_dir_path
                 ,output_path = 'C:\\Users\\15126\\Desktop\\stanford\\dev_dummy')

train_dummy = create_dummy_dataset(file_names = train_dict
                 ,dir_path = train_dir_path
                 ,output_path = 'C:\\Users\\15126\\Desktop\\stanford\\train_dummy') 

dummy1 = train_dummy['LA_T_3630688']['data']


def generate_time_step_samples(np_array
                               ,ts_length = 16
                               ,offset = 16):
    """Outputs dimensions (num_samples, )"""
    
    list_of_arrays = []
    col_dim = np_array.shape[0]
    time_len = np_array.shape[1]
    print(col_dim,time_len)
    
    start = 0
    end = ts_length
    
    while(end < time_len):
        trn_data = np_array[:,start:end]
        trn_data = np.reshape(trn_data.T,(ts_length,col_dim,1),order='C')
        list_of_arrays.append(trn_data)
        
        start+=offset
        end+=offset
        
    new_array = np.stack(list_of_arrays)     
    return new_array        
        
new_array = generate_time_step_samples(dummy1,ts_length=16,offset=1)   
new_array.shape
