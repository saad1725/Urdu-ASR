from __future__ import division


from transformers import AutoProcessor
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from functools import partial
import pandas as pd
import numpy as np
from datasets import (
    load_dataset, 
    load_from_disk,
    load_metric,)
# from datasets.filesystems import S3FileSystem
from transformers import (
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from transformers import Wav2Vec2ProcessorWithLM
import torchaudio
import re
import json
from datasets import ClassLabel


from typing_extensions import Self

from transformers import MarianTokenizer, TFMarianMTModel, pipeline
import tensorflow as tf
import torch
torch.cuda.empty_cache()
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import os
from fairseq import utils
import sys
import librosa
import soundfile as sf
import wave
import webrtcvad 
import numpy as np
import pyaudio
import re
import sys
import threading


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from six.moves import queue


# torchaudio.set_audio_backend("sox_io")
import time



from pydub import AudioSegment
from scipy.signal import butter, lfilter
# %matplotlib inline
from time import sleep, perf_counter
from threading import Thread
from glob import glob
import playsound
import librosa
import io
import wave
import subprocess
import shlex
import pyaudio





from pyctcdecode import build_ctcdecoder
import kenlm

#Loading wav2vec Model and processor

processor = AutoProcessor.from_pretrained("data/wav2vec2-large-xlsr-53-ur/checkpoint-11000")    
model = Wav2Vec2ForCTC.from_pretrained("data/wav2vec2-large-xlsr-53-ur/checkpoint-11000")

#fetching vocabulary
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}


#loading language model
ken= "data/model_with_lm1/language_model/urdu.bin"

#building decoder for inference
decoder = build_ctcdecoder(
     labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path=ken  # tuned on a val set
)

processor= Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)


#convert audio file into array for model processing

def speech_file_to_array_fn(audio,resampling_to=16000):

    speech_array, sampling_rate = torchaudio.load(audio)
    resampler=torchaudio.transforms.Resample(sampling_rate, resampling_to)
    speech = resampler(speech_array).numpy()
    return speech

# function that runs the ASR model
def asr_model(audio):
    speech=speech_file_to_array_fn(audio)
    inputs = processor(speech.squeeze(), sampling_rate=16_000, return_tensors="pt", padding=True)
    # print(inputs.input_values.shape)
    with torch.no_grad():
        logits = model(inputs.input_values,).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    if processor.batch_decode(logits.numpy()).text[0] != '':
        print("Prediction:", processor.batch_decode(logits.numpy()).text)



#setting global values

RATE = 16000  #sample rate
CHUNK = 160    #chunk size of data that will be aquired from microphone
TH_pool = []   #array of threads
path_r="recording/"   #path for recordings
#path_s="synthesized/"


#class for inference that will be used by threads

class ModelInfer(threading.Thread):
    def __init__(self, filepath,file_bytes):
        super(ModelInfer,self).__init__()
        self.filepath = filepath
        self.file_byte=file_bytes
    def infer(self):
        filenamed=self.filepath
        bytes_file=self.file_byte
        if os.path.exists(filenamed):
            try:
                asr_model(filenamed)      #asr model function being called
            except Exception as e:
                print(str(e))
    def run(self):
        self.infer()


#create file from bytes received from the microphone 
def makefile(data,sample_format,filename):
        audio_bytes = np.frombuffer(b''.join(data), np.int16)
        audio=librosa.util.buf_to_float(b''.join(data))
        clip = librosa.effects.trim(audio, top_db= 5)  #if audio contains audio of level less that 5db its stored as empty
        sf.write(filename, clip[0], 16000)
        return filename,audio_bytes

# main Recording Function
def recording():
    p=pyaudio.PyAudio()
    vad=webrtcvad.Vad()    #voice Activity detection
    sample_format=pyaudio.paInt16
    vad.set_mode(3)         # 3 mode is the most aggressive
    i=0
    j=0
    filenamed_list = []
    file_bytes=[]
    
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)  #initializing microphone stream
    buff=[]
    count_sil=0
    
    print("Speak")


    # main continous recording loop
    while(True):       
        data=stream.read(CHUNK,exception_on_overflow=False)    #read chunk of data
        if vad.is_speech(data,RATE):                           #detect voice activity
            buff.append(data)
            # print("VAD")
        if len(buff)>0:
            if (vad.is_speech(buf=data,sample_rate=RATE))==False:
                count_sil=count_sil+1
                if count_sil>150:                               #wait for End of sentence silence

                    filenamed,filename_byte=makefile(buff,p.get_sample_size(sample_format),path_r+str(i)+'.wav')
                    
                    filenamed_list.append(filenamed)  #creating file list
                    file_bytes.append(filename_byte)
                    buff=[]
                    i=i+1
                    count_sil=0
            if len(filenamed_list)!=0:
                if (len(filenamed_list)>j):
                    flag, th = getTHread(filenamed_list[j],file_bytes[j]) 
                    if flag:
                        
                        th.start() #creating processing thread
                        th.join()
                        TH_pool.append(th)
                        j += 1            

#function maintains an array of threads with max value being 3. If thread is to be created this functions the availbility of space in array and
# if space is available new thread is created other wise the processing will have to wait.
def getTHread(fileName,bytesname):
    if len(TH_pool) < 3: 
        return True, ModelInfer(fileName,bytesname)
    else:
        for idx, thread in enumerate(TH_pool):
            if not thread.is_alive():
                TH_pool.pop(idx)
                return True, ModelInfer(fileName,bytesname)
    return False, None




# main Function 
def main():
    t1=Thread(target=recording)
    t1.start()



                    
if __name__=='__main__':
    main()



