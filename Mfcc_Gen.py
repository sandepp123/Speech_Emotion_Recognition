from os import listdir
import essentia
from os.path import isfile,join
import essentia.standard
import essentia.streaming
import numpy as np
from essentia.standard import *
import pandas as pd
w = Windowing(type = 'hann')
#################PATHS#############################
mypath='/home/sandepp/Final_Projct/wav'
#################PATHS#############################
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC()
loader = essentia.standard.MonoLoader()
energy_func = essentia.standard.Energy()
zcr=ZeroCrossingRate()
pitch=PitchYinFFT()
#frame = audio[1*16000 : 4*16000 + 1024]
name_of_files=[]
feature_matrix=np.zeros(75)

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#print onlyfiles  # Get names of all the files
emotion_class_map = {'W' : 0, 'L' : 1, 'E' : 2, 'A' : 3, 'F' : 4, 'T' : 5, 'N' : 6}

for wav_files in onlyfiles:

	loader.configure(filename=join(mypath,wav_files))
	audio=loader()
	frame_energy = energy_func(audio)
	frame_zcr = zcr(audio)
	frame = audio[1*16000 : 4*16000 + 1024]
	spec1=spectrum(w(frame))
	frame_pitch,t=pitch(spec1)
	emotion=wav_files[-6]

	mfccs=[]
	if frame.shape[0]%2==0:
		frameSize = 1024
		hopSize = 512
		for fstart in range(0, len(audio)-frameSize, hopSize):

			frame = audio[fstart:fstart+frameSize]
			mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
			mfccs.append(mfcc_coeffs)
			#print np.array(mfccs).shape
	elif frame.shape[0]!=0:
		print wav_files
		frame = audio[1*16000 : 4*16000 + 1023]
		frameSize = 1024
		hopSize = 512
		for fstart in range(0, len(audio)-frameSize, hopSize):

			frame = audio[fstart:fstart+frameSize]
			mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame[:-1])))
			mfccs.append(mfcc_coeffs)

	x= np.array(mfccs)
	t=[]
	for i in range(70):

		t.append(x[i][1:].mean())
		if i==69:
			y=t
		#	y=np.array(t)
		#	z=np.reshape(y,-1,'C')
		#	print y.shape
			y.append(frame_energy)
			y.append(frame_zcr)
			y.append(frame_pitch)
			y.append(emotion)
			y.append(wav_files)
			#print z.shape
			feature_matrix=np.vstack((feature_matrix,y))
			#print y


#print "HA"'''
print feature_matrix
print feature_matrix.shape
#np.savetxt(MFcc,feature_matrix,delimiter=(','))
f_Matrix = pd.DataFrame(feature_matrix)
header=[]
for i in range(70):
	header.append("Mfcc"+str(i))
header.append('frame_energy')
header.append('frame_zcr')
header.append('frame_pitch')
header.append('emotion')
header.append('wav_files')
print len(header)
f_Matrix.to_csv('final.csv',sep=';',header=header,index=False)




#loader = essentia.standard.MonoLoader(filename = filepath)
