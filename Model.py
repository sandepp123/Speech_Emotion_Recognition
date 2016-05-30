import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from nolearn.dbn import DBN
import warnings
import numpy as np
def file_open(x,y):
	return pd.read_csv(x,delimiter=y)                       # returns open file connection

def data_clean(x,list_missing_value_col):
	for  i in list_missing_value_col:
		x[i].fillna(mode(x[i])[0][0],inplace=True)

	return x

def converting_to_object(x,list_of_col_to_convert):        # some columns should work better as object than numerical value
	for i in list_of_col_to_convert:
		x[i]=x[i].apply(lambda x: str(x))
	return x

def labeling(x,col_as_objects):
	le=LabelEncoder()
	for i in col_as_objects:
		#warnings.filterwarnings("ignore",category=DataConversionWarning)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			x[i]=le.fit_transform(x[i].ravel())
		#y[i]=le.fit_transform(y[i])

	return x

def split_dataset(x):
	x_train,y_train=cross_validation.train_test_split(x,test_size=0.3,random_state=0)
	return x_train,y_train

def classification(x,y,features_independent,dependent):
	dt=DBN(verbose=True)
	dt.fit(np.array(x[features_independent]),np.array(x[dependent]))
	predict=dt.predict(np.array(y[features_independent]))
	return predict

def Normalizing(train,categorical,thresvalue):
	for columns in categorical:
		print "entering",columns
		frq=train[columns].value_counts()
		categories_to_combine=frq.loc[frq.values<thresvalue[categorical.index(columns)]].index
		for cat in categories_to_combine:
			train[columns].replace({cat:'Others'},inplace=True)
		print 'exiting',columns
	return train
