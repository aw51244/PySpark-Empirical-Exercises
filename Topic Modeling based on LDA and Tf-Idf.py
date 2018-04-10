# Load data

# path = "~reviews.csv" # (this one is not working)
path = "hdfs:///home/reviews.csv" # load data from s3 
rawdata = spark.read.load(path, format="com.databricks.spark.csv", header = True, inferschema = True)
rawdata.show(5)

# tokenize text data
from pyspark.sql.functions import udf
from pyspark.ml.feature import Tokenizer, RegexTokenizer
def extract_tokens(df):
	reTokenizer = RegexTokenizer(inputCol="Text", outputCol="clean_text",toLowercase=True, minTokenLength=3)
	newdf=reTokenizer.transform(df)
	return newdf

df_tokens=extract_tokens(rawdata)


# remove stop words
from pyspark.ml.feature import StopWordsRemover
remover = StopWordsRemover(inputCol="clean_text", outputCol="filtered")
df_remove = remover.transform(df_tokens)
df_remove.show(5)

# Tf-Idf
from pyspark.ml.feature import HashingTF, IDF
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(df_remove)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData.cache()

# Trains a LDA model.
from pyspark.ml.clustering import LDA
lda = LDA(k=10, maxIter=100)
model = lda.fit(rescaledData)

# Show top words
topics = model.describeTopics(10)
topics.show(truncate=False)

