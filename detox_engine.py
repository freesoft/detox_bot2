'''
Author : Wonhee Jung (wonheej2@illinois.edu, wonhee.jung@gmail.com), Cindy Tseng (cindyst2@illinois.edu )
Since : Feb, 2019

UIUC MCS-DS CS498 CCA Spring 2019 Project.
'''

import gc
import os
import os.path
import sys
import time
import numpy as np
import atexit
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, NGram, Word2Vec, CountVectorizer
from pyspark.sql.types import StringType
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, GBTClassifier, LinearSVC, LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.sql.functions import when
from pyspark.ml import Pipeline, PipelineModel
import string

import constant

os.environ['OMP_NUM_THREADS'] = '2'

# The class will create two joblib files to store the trained classifier and vectorizer so that it can be reused quickly
# instead of training it from the beginning.
# 
# !!!! IMPORTANT !!!
#
# Which means if you ever chagne any code in __init__ that could affect the way how classifier/vectorizer is going to work,
# then you need to delete two files so that program recrete them. Those two files you need to delete are "classifier.joblib", and "vectorizer.joblib"
# defined in constant.py.
class ToxicityClassifier():

    def __init__(self):

        start_time = time.time()
        
        # start spark session
        self.spark = SparkSession.builder.appName("DetoxBot").getOrCreate()
        self.sc = self.spark.sparkContext
        self.sqlContext = SQLContext(self.sc)

        # if and only if model doesn't exist in the file, execute this block, means you need to delete the existing model file to re-run this.
        if os.path.exists(constant.MODEL_DIR) == False:
            print("Can't find existing classifier stored in the file. Creating one...")

            # read data
            spDF = self.sqlContext.read.csv(constant.TRAINING_DATA_PATH, 
                                       header="true", 
                                       multiLine=True, 
                                       inferSchema=True,
                                       escape="\"")
            # generate label column
            spDF = spDF.withColumn('label', 
                spDF['toxic'] + \
                spDF['severe_toxic'] +  \
                spDF['obscene'] + \
                spDF['threat'] + \
                spDF['insult'] + \
                spDF['identity_hate'] )
            spDF = spDF.withColumn('label', when(spDF['label'] > 0, 1).otherwise(0) )

            # remove redudant columns
            drop_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            spDF = spDF.drop(*drop_list)
            
            # generate features column
            # tokenize
            tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
            wordsData = tokenizer.transform(spDF)
            
            # stop words remover
            stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filteredWords")
            filteredWordsData = stopwords_remover.transform(wordsData)
            
            # remove punctuations
            list_punct = list(string.punctuation)
            punc_remover = StopWordsRemover(inputCol="filteredWords", outputCol="filteredPunc", stopWords=list_punct)
            filteredPuncData = punc_remover.transform(filteredWordsData)
            
            hashingTF = HashingTF(inputCol="filteredPunc", outputCol="rawFeatures")
            featurizedData = hashingTF.transform(filteredPuncData)
            
            # inverse document frequency transformations
            idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=1)
            idfModel = idf.fit(featurizedData)
            rescaledData = idfModel.transform(featurizedData)

            # split data into train and test
            training, test = spDF.randomSplit([0.8, 0.2], seed = 0)
            
            # naive bais classification
            nb = NaiveBayes(smoothing=1.5, labelCol='label', featuresCol='features')

            # configure ML pipeline
            pipeline = Pipeline(stages=[tokenizer, stopwords_remover, punc_remover, hashingTF, idf, nb])
            
            # fit the pipeline to training documents
            print("Initiating training...")
            self.model = pipeline.fit(training)
            del training
            gc.collect()
            print("Completed training. Generating classification result...")
            
            predictions = self.model.transform(test)

            # evaluate prediction
            self.eval_test(predictions)
          
            del test
            gc.collect()

            print("Storing model info into disk...")
            self.model.save(constant.MODEL_DIR)

        else:
            print("Found existing model stored in the file. Loading...")
            self.model = PipelineModel.load(constant.MODEL_DIR)

        # to measure programing execution time
        print("--- time spent for initializing the classifier : %s seconds ---" % (time.time() - start_time))

    def eval_test(self, predictions):
        TP = predictions.select("label", "prediction").filter((predictions.label == 1) & (predictions.prediction == 1)).count()
        TN = predictions.select("label", "prediction").filter((predictions.label == 0) & (predictions.prediction == 0)).count()
        FP = predictions.select("label", "prediction").filter((predictions.label == 0) & (predictions.prediction == 1)).count()
        FN = predictions.select("label", "prediction").filter((predictions.label == 1) & (predictions.prediction == 0)).count()
        total = predictions.select("label").count()
        l1_num = predictions.select('label').filter(predictions.label == 1).count()
        l0_num = predictions.select('label').filter(predictions.label == 0).count()
        
        accuracy	= (TP + TN) / total
        l1_precision   = TP / (TP + FP)
        l1_recall      = TP / (TP + FN)
        l1_F1		= 2/(1/l1_precision + 1/l1_precision)
        
        l0_precision   = TN / (TN + FN)
        l0_recall      = TN / (TN + FP)
        l0_F1		= 2/(1/l0_precision + 1/l0_precision)

        weighted_avg_precision = (l1_precision * l1_num + l0_precision * l0_num)/(l1_num + l0_num)
        weighted_avg_recall = (l1_recall * l1_num + l0_recall * l0_num)/(l1_num + l0_num)
        weighted_avg_F1 = (l1_F1 * l1_num + l0_F1 * l0_num)/(l1_num + l0_num)

        avg_F1 = (l0_F1 + l1_F1)/2
        print('accuracy:', accuracy)
        print('label 1 precision:', l1_precision)
        print('label 1 recall:', l1_recall)
        print('label 1 F1:', l1_F1)
        
        print('label 0 precision:', l0_precision)
        print('label 0 recall:', l0_recall)
        print('label 0 F1:', l0_F1)
        
        print('num label 0', l0_num)
        print('num label 1', l1_num)
        print('weighted average precision', weighted_avg_precision)
        print('weighted average recall', weighted_avg_recall)
        print('weighted average f1', weighted_avg_F1)
        
    def stopClassifier(self):
        # close spark session
        self.sc.stop()

    # tokenizer only works off of dataframe, need to convert text to dataframe
    # before we can pass it to the stored ML pipeline
    def txtIsToxic(self, txt):
        df = self.sqlContext.createDataFrame([txt], StringType())
        txtDF = df.withColumnRenamed('value', 'comment_text')
   
        return self.isToxic(txtDF)
    
    # with given parameter s, it returns whether s is toxic or not
    # it is not expecting any arrays, it should be just single string value
    def isToxic(self, s):

        pred_df = self.model.transform(s)
        pred_list = pred_df.select('prediction').collect()
        return np.any(pred_list)
        

# main function if you need to run it separated, not through chatbot.py.
# The function will load local test CSV file and execute the prediction, instead of getting messages from TwitchTV channel it has deployed.
# With "test/test.csv", the code is going to take anothe 300MB +- to load all the text data and process it.
def main():
    print("Initiating...")

    # below file has smaller set of test data. Enable it instead if you want quicker testing
    test_data_path = "data/test_sample.csv"
    # below file has full set of test data. Try with it if you see more dresult. Beware : it will take some time.

    toxicClassifier = ToxicityClassifier()

    # loading chat logs from csv file
    df = toxicClassifier.sqlContext.read.csv(test_data_path, 
                                       header="true", 
                                       multiLine=True, 
                                       inferSchema=True,
                                       escape="\"")

    # transform test data's chat log to the existing vecorizer so it can be used for prediction        
    preds_list = toxicClassifier.model.transform(df).select('prediction').collect()
    preds = [row.prediction for row in preds_list]
    text = df.select('comment_text').collect()
    #print('preds', preds, 'text', text)
    for i, p in enumerate(preds):
        if p == 1:
            print("TOXIC>>> " + text[i].comment_text)            
            
    atexit.register(toxicClassifier.stopClassifier)
    
# just in case if need to run the test locally without TwitchBox working together
if __name__ == "__main__":

    if not sys.version_info[:1] == (3,):
        print(sys.version_info[:1] )
        sys.stderr.write("Python version 3 is required.\n")
        exit(1)

    main()
