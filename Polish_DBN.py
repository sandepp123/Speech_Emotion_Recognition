import Model
from nolearn.dbn import DBN
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

t=open('Tuning4.txt','w')
train=Model.file_open('Train',',')
test2=Model.file_open('polish_test',',')
test=Model.file_open('polish_test',',')
#print test['emotion']==test2['emotion']
column_names=list(train.columns)
#print column_names
dependent  =['emotion']

for i in range(122):
	independent=[]
	print independent
	#independent=column_names[0:10]
	independent.extend(column_names[20:60])
	#independent.extend(column_names[90:100])
	independent.append('energy')
	independent.append('zcr')
	independent.append('pitch')
	print independent
	#independent.append('gender')
	#print independent
	#print train[independent].shape

	Xtrain=train[independent]#/train[independent].max()#training matrix
	Xtrain=pd.concat([Xtrain,train['emotion']],axis=1)#labeling of training features
	Xtest=test[independent]#/train[independent].max()#testing matrix
	Xtest=pd.concat([Xtest[independent],test['emotion']],axis=1)#testing feature labeling
	#Xtrain=Model.labeling(Xtrain,['emotion'])
	#train_train,test_train=Model.split_dataset(Xtrain)
#	print Xtrain['emotion'].value_counts()
#	print Xtest['emotion'].value_counts()
	X=np.array(train[independent])
	y=np.array(train[dependent])
	X2=np.array(test[independent])
#	print train[train['emotion']=='ang'].shape
	#print X[independent].shape[0]
	#print y.shape
	#test=test[independent]/train[independent].max()
	#x_test=np.array(test[independent])
#	print test2['emotion']

	#y_test=np.array(test2[dependent])
	#clf=DecisionTreeClassifier()
	#[-1,300,-1],learn_rates=.001,epochs=500,verbose=False)#DBN([783, 300, 10], learn_rates=0.1, learn_rates_pretrain=0.005)
	#clf=DBN()
	#clf.fit(X,y)
	pred=Model.classification(Xtrain,Xtest,independent,dependent)
	pred=pd.DataFrame(pred)
	#print pred
	score=pred[0]==test2['emotion']
	print score.value_counts()
	print pd.concat([pred,test2['emotion'],test['wav_file']],axis=1)
	final=float(score[score==True].sum())/score.shape[0]
	t.write("for mfcc"+' '+str(i)+' accuracy = '+str(final)+'\n')
	print "for mfcc",i,final
t.close()
