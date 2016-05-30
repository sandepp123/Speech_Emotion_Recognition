from os import listdir
import essentia
from os.path import isfile,join
import essentia.standard
import essentia.streaming
import numpy as np
from essentia.standard import *
import pandas as pd
low_bound=20
upper_bound=20000
w = Windowing(type = 'hann')
def Mfcc_(mypath):
	spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
	mfcc = MFCC(lowFrequencyBound=20,highFrequencyBound=20000,numberCoefficients=1)
	loader = essentia.standard.MonoLoader()
	energy_func = essentia.standard.Energy()
	zcr=ZeroCrossingRate()
	pitch=PitchYinFFT(maxFrequency=20000,minFrequency=20)

	name_of_files=[]
	feature_matrix=np.zeros(108)

	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	for wav_files in onlyfiles:

		loader.configure(filename=join(mypath,wav_files))
		audio=loader()
		frame_energy = energy_func(audio)
		frame_zcr = zcr(audio)
		frame = audio[1*44100 : 7*44100 + 1024]
		print "size of frame",frame
		#frame = audio[:]
		print len(frame)
		print wav_files
		if  len(frame)%2!=0:
			print "llana"
			spec1=spectrum(w(frame[0:-1]))
			frame_pitch,t=pitch(spec1)
		else:
			print "else"
			spec1=spectrum(w(frame[0:-2]))
			print spec1.shape[0]
			print "el2"
			frame_pitch,t=pitch(spec1[0:-1])
		emotion=wav_files[2:5]
		print "emotion",emotion
		gender=wav_files[0]
		print "gender",gender
		speaker=wav_files[1]
		print "speaker",speaker
		sentence=wav_files[-5]
		print "sentence",sentence

		mfccs=[]
		if frame.shape[0]%2==0:
			frameSize = 1024
			hopSize = 512

			for fstart in range(0, len(audio)-frameSize, hopSize):
				spectrums=spectrum(w(frame))
				frame = audio[fstart:fstart+frameSize]

				if spectrums.shape[0]%2==0:
					mfcc_bands, mfcc_coeffs = mfcc(spectrums)
				else:
					mfcc_bands,mfcc_coeffs=mfcc(spectrums[0:-1])
				mfccs.append(mfcc_coeffs)
				#print np.array(mfccs).shape
		elif frame.shape[0]%2!=0:
			print wav_files
			frame = audio[:-1]
			frameSize = 1024
			hopSize = 512
			print str(frame.shape) + "edode"

			for fstart in range(0, len(audio)-frameSize, hopSize):
				spectrums=spectrum(w(frame))
				if spectrums.shape[0]%2==0:
					mfcc_bands, mfcc_coeffs = mfcc(spectrums)
				else:
					mfcc_bands,mfcc_coeffs=mfcc(spectrums[0:-1])
				mfccs.append(mfcc_coeffs)
				mfccs.append(mfcc_coeffs)

		x= np.array(mfccs)
		t=[]
		for i in range(100):

			#t.append(x[i][1:].mean())
			t.append(x[i])
			if i==100:
				y=t
			#	y=np.array(t)
			#	z=np.reshape(y,-1,'C')
			#	print y.shape
				y.append(frame_energy)
				y.append(frame_zcr)
				y.append(frame_pitch)
				y.append(emotion)
				y.append(gender)
				y.append(speaker)
				y.append(sentence)
				y.append(wav_files)
				#print z.shape
				feature_matrix=np.vstack((feature_matrix,y))
	return feature_matrix

emotion=['anger','boredom','fear','joy','neutral','neutral','sadness','test']
for i in emotion:
	path="/home/sandepp/Final_Projct/emotions/speech"
	path=path+"/"+i
	x=Mfcc_(path)
	x=pd.DataFrame(x)
	name='polish'+'_'+i
	x.to_csv(name,index=False)
