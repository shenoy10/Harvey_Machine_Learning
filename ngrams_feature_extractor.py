import tensorflow as tf
print(tf.__version__)
import numpy as np
#from ApplicationEntry import ApplicationEntry
#from TensorflowApplicationEntry import TensorflowApplicationEntry
import traceback
import os, shutil
import csv
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

csv.field_size_limit(sys.maxsize)
class NGramsFeatureExtractor():
	def removeURLs(self, text):
		text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE)
		return text
	def removeNonAscii(self, text):
		text.replace('"', "")

		text = ''.join([i if ord(i) < 128 else '' for i in text])
		text.replace('"', '');
		text.replace('"', '');
		text.replace('\"', '');
		text.replace('\"', '');
		text = ''.join([i if i is not "\"" else '' for i in text])

		return text

	def removePunctuation(self, text):
		finalText = re.sub('\<.+\>', '', text)
		finalText.replace("\n", "")
		finalText.replace("\r", "")
		finalText.replace("\t", "")
		finalText = finalText.lower()
		return finalText


	def __init__(self, csvFileName):
		self.inputFileName = csvFileName;
		#self.outputFileName = csvFileName2;

	def extractData(self, labelCol, csvFileName = None, outputFileName = None):
		labels = list()
		features = list()
		testFeatures = list()

		trainRawTexts = list()
		testRawTexts = list()
		
		trainPrintingRawTexts = list()
		testPrintingRawTexts = list()


		with open(self.inputFileName) as csvfile:
			reader = csv.reader(csvfile)
			first = True
			amit = 0
			for row in reader:
				amit += 1
				
				print (row)
				#below is if there is a header column
				'''if(first):
					first = False
					continue
				if len(row) <= 1:
					continue'''

				#starting from 1 because of id column (may not apply here)

				if len(row) < 3:
					continue
				newFeature = row[1:labelCol]
				
				text = row[labelCol]
				
				text = self.removeURLs(text)

				text = self.removeNonAscii(text)


				trainPrintingRawTexts.append(text)

				
				textNoPunctuation = self.removePunctuation(text)
				if textNoPunctuation is not None:
					trainRawTexts.append(textNoPunctuation)
				features.append(row)
		

		

		'''with open(self.csvFileName2) as csvfile:
			reader = csv.reader(csvfile)
			first = True
			for row in reader:

				#below is if there is a header column
				#if(first):
				#	first = False
				#	continue
				#if len(row) <= 1:
				#	continue
				
				#the one gets rid of ID column
				newFeature = row[1:labelCol-1]
				
				
				text = row[labelCol-1]
				
				text = removeNonAscii(text)

				testPrintingRawTexts.append(text)

				textNoPunctuation = removePunctuation(text)

				testRawTexts.append(textNoPunctuation) #happens to be the raw text column

				testFeatures.append(newFeature)'''	


		tfidf = TfidfVectorizer(min_df = 0.01, max_df = 0.99, ngram_range = (1, 4), sublinear_tf = False)

		extraFeatures = tfidf.fit_transform(trainRawTexts)

		

		ngramFeatureNames = list()
		newFeatureNames = tfidf.get_feature_names()
		print (newFeatureNames)
		ngramsextracted = open("ngramsextracted.txt", 'w');
		for i in range(len(newFeatureNames)):
			ngramFeatureNames.append("frequency: " + str(newFeatureNames[i]))
			ngramsextracted.write(str(newFeatureNames[i]) + "\n")
			#print (str(i) + ": " + ngramFeatureNames[i])
		ngramsextracted.close()
		'''densed = extraFeatures.todense()

		counter = 0
		for newFeature in densed:
			toAdd = np.array(newFeature)[0].tolist()
			print ("adding to train: " + str(counter))
			features[counter].extend(toAdd)
			counter += 1





		labels = np.asarray(labels)

		features = np.asarray(features)

		testFeatures = np.asarray(testFeatures)
		print ("--------------------Features retrieved-----------------------")
		print(features)
		print ("--------------------Features Printed-----------------------")

		origFeaturesHeader = "ID,Text,"
		totalFeaturesHeader = origFeaturesHeader
		for i in range(len(ngramFeatureNames)):
			if i < len(ngramFeatureNames) - 1:
				totalFeaturesHeader += ngramFeatureNames[i] + ","
			else:
				totalFeaturesHeader += ngramFeatureNames[i] + ","

		f = open(outputFileName, 'w')
		f.write(totalFeaturesHeader + ",text\n")
		for i in range(len(features)):
			f.write(str(i) + ",")
			for j in range(0, len(features[i])):
				f.write(str(features[i][j]) + ",")
			f.write("\"" + trainPrintingRawTexts[i] + "\"" + ",")
			f.write("\n")
'''


extraFeatureExtractor = NGramsFeatureExtractor("labeled_prelim.csv");
extraFeatureExtractor.extractData(labelCol = 1, outputFileName = "labeled_prelim_with_ngrams.csv")




