'''
Author : Wonhee Jung ( wonheej2@illinois.edu, wonhee.jung@gmail.com )
Since : Nov, 2018

UIUC MCS-DS CS410 Fall 2018 Project.
'''
import csv
import gc
import os
import os.path
import sys
import time
from builtins import range

#import nltk
import numpy as np
import pandas as pd
from joblib import dump, load

from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.sql.types import DoubleType
#from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import col, expr, when
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
# includes all the defined constant variables.
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
#from dbmlModelExport import ModelExport

import constant

# code to make Windows happy with spark
#spark_path = os.path.join(*['C:','\\Progra~2', 'spark-2.3.3-bin-hadoop2.7'])
##spark_path = r"C:\\Progra~2\\spark-2.3.3-bin-hadoop2.7\\spark-2.3.3-bin-hadoop2.7" # spark installed folder
#os.environ['SPARK_HOME'] = spark_path
#sys.path.insert(0, os.path.join(*[spark_path,"bin"]))
#sys.path.insert(0, os.path.join(*[spark_path, "python', 'pyspark"]))
#sys.path.insert(0, os.path.join(*[spark_path, "python', 'lib', 'pyspark.zip"]))
#sys.path.insert(0, os.path.join(*[spark_path, "python','lib', 'py4j-0.10.7-src.zip"]))


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
        spark = SparkSession.builder.appName("DetoxBot").getOrCreate()
        self.sc = spark.sparkContext
        self.sqlContext = SQLContext(self.sc)
        self.stopwords = list(set(w.rstrip() for w in open('stopwords.txt')))
#        sc = spark.sparkContext
#        sqlContext = SQLContext(sc)
#        stopwords = list(set(w.rstrip() for w in open('stopwords.txt')))

        # if and only if model doesn't exist in the file, execute this block, means you need to delete the existing model file to re-run this.
        if os.path.exists(constant.MODEL_DIR) == False:
            print("Can't find existing classifier stored in the file. Creating one...")
            
            # read data
            spDF = self.sqlContext.read.csv(constant.TRAINING_DATA_PATH, 
            #spDF = sqlContext.read.csv(constant.TRAINING_DATA_PATH, 
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
                
            # generate features column
            # tokenize
            tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
            wordsData = tokenizer.transform(spDF)
            
            # stop words remover
            remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=self.stopwords)
            #remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=stopwords)
            filteredData = remover.transform(wordsData)
            
            # term frequency transformation
            hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures")
            featurizedData = hashingTF.transform(filteredData)
            
            # inverse document frequency transformations
            idf = IDF(inputCol="rawFeatures", outputCol="features")
            idfModel = idf.fit(featurizedData)
            rescaledData = idfModel.transform(featurizedData)

            # split data into train and test
            training, test = spDF.randomSplit([0.8, 0.2], seed = 0)
            
            #rescaledData.select("features").show()
            # naive bais classification
            nb = NaiveBayes(smoothing=1.0, labelCol='label', featuresCol='features')
            
            # configure ML pipeline
            pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, nb])
            
            # fit the pipeline to training documents
            print("Initiating training...")
            self.model = pipeline.fit(training)
            del training
            gc.collect()
            print("Completed training. Generating classification result...")
            
            # make predictions
            predictions = self.model.transform(test)

            # evaluate prediction
            TP = predictions.select("label", "prediction").filter((predictions.label == 1) & (predictions.prediction == 1)).count()
            TN = predictions.select("label", "prediction").filter((predictions.label == 0) & (predictions.prediction == 0)).count()
            FP = predictions.select("label", "prediction").filter((predictions.label == 0) & (predictions.prediction == 1)).count()
            FN = predictions.select("label", "prediction").filter((predictions.label == 1) & (predictions.prediction == 0)).count()
            total = predictions.select("label").count()

            accuracy	= (TP + TN) / total
            precision   = TP / (TP + FP)
            recall      = TP / (TP + FN)
            F1		= 2/(1/precision + 1/recall)

            print('accuracy:', accuracy)
            print('precision:', precision)
            print('recall:', recall)
            print('F1:', F1)
#            
#            evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction')
#            accuracy = evaluator.evaluate(predictions)
#            print("Test set binarcy evaluator ara under curve accuracy = " + str(accuracy))
#            
#            mcevaluator = MulticlassClassificationEvaluator(predictionCol='prediction', metricName='weightedPrecision')
#            mcaccuracy = mcevaluator.evaluate(predictions)
#            print("Test set multiclass weighted precision accuracy = " + str(mcaccuracy))
#            
#            mcevaluator = MulticlassClassificationEvaluator(predictionCol='prediction', metricName='weightedRecall')
#            mcaccuracy = mcevaluator.evaluate(predictions)
#            print("Test set multiclass weighted recall  = " + str(mcaccuracy))
#            
#            mcevaluator = MulticlassClassificationEvaluator(predictionCol='prediction', metricName='accuracy')
#            mcaccuracy = mcevaluator.evaluate(predictions)
#            print("Test set multiclass accuracy = " + str(mcaccuracy))
#            
#            mcevaluator = MulticlassClassificationEvaluator(predictionCol='prediction', metricName='f1')
#            mcaccuracy = mcevaluator.evaluate(predictions)
#            print("Test set multiclass f1 = " + str(mcaccuracy))
#            
            del test
            gc.collect()

            print("Storing model info into disk...")
            self.model.save(constant.MODEL_DIR)

        else:
            print("Found existing model stored in the file. Loading...")
            self.model = PipelineModel.load(constant.MODEL_DIR)

        # to measure programing execution time
        print("--- time spent for initializing the classifier : %s seconds ---" % (time.time() - start_time))

        
    def stopClassifier(self):
        # close spark session
        self.sc.stop()

    # with given parameter s, it returns whether s is toxic or not
    # it is not expecting any arrays, it should be just single string value
    def isToxic(self, s):

        pred_df = self.model.transform(s)
        pred_list = pred_df.select('prediction').collect()
        pred = [bool(row.prediction) for row in pred_list]
        
        return pred
        

# main function if you need to run it separated, not through chatbot.py.
# The function will load local test CSV file and execute the prediction, instead of getting messages from TwitchTV channel it has deployed.
# With "test/test.csv", the code is going to take anothe 300MB +- to load all the text data and process it.
def main():
    print("Initiating...")

    # below file has smaller set of test data. Enable it instead if you want quicker testing
    test_data_path = "data/test_sample.csv"
    # below file has full set of test data. Try with it if you see more dresult. Beware : it will take some time.
    #test_data_path = "data/test.csv"

    toxicClassifier = ToxicityClassifier()

    # loading chat logs from csv file
    df = toxicClassifier.sqlContext.read.csv(test_data_path, 
                                       header="true", 
                                       multiLine=True, 
                                       inferSchema=True,
                                       escape="\"")

    # transform test data's chat log to the existing vecorizer so it can be used for prediction        
    preds = toxicClassifier.isToxic(df)
    
    # print(pd.DataFrame(preds, columns=toxicClassifier.classifier.classes_))
    text = df.select('comment_text').collect()
    for i, p in enumerate(preds):
        #if p == 1:
        if p == True:
            print("TOXIC>>> " + text[i].comment_text)            
            
    toxicClassifier.stopClassifier()
# just in case if need to run the test locally without TwitchBox working together
if __name__ == "__main__":

    if not sys.version_info[:1] == (3,):
        print(sys.version_info[:1] )
        sys.stderr.write("Python version 3 is required.\n")
        exit(1)

    main()
