from os import listdir
import essentia
from os.path import isfile,join
import essentia.standard
import essentia.streaming
import numpy as np
from essentia.standard import *
w = Windowing(type = 'hann')
#################PATHS#############################
mypath='/home/sandepp/Final_Projct/wav'
#################PATHS#############################
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC()
loader = essentia.standard.MonoLoader()
#frame = audio[1*16000 : 4*16000 + 1024]
name_of_files=[]
feature_matrix=np.zeros(70)

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print onlyfiles  # Get names of all the files
emotion_class_map = {'W' : 0, 'L' : 1, 'E' : 2, 'A' : 3, 'F' : 4, 'T' : 5, 'N' : 6}

for wav_files in onlyfiles:
	
	loader.configure(filename=join(mypath,wav_files))
	audio=loader()
	
	frame = audio[1*16000 : 4*16000 + 1024]
	mfccs=[]
	if frame.shape[0]%2==0:
		frameSize = 1024
		hopSize = 512
		for fstart in range(0, len(audio)-frameSize, hopSize):
			
			frame = audio[fstart:fstart+frameSize]
			mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
			mfccs.append(mfcc_coeffs)
			#print np.array(mfccs).shape
	x= np.array(mfccs)
	ti=x.shape
	if ti[0]<70:
		print '***************************',ti,wav_files
	else:
		print ti,wav_files


	

#loader = essentia.standard.MonoLoader(filename = filepath)

