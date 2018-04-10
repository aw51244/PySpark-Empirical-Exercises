# read the data from the local folder. 
# infer the schema (e.g., columns) from the csv file. 

df = spark.read.load("~/titanic_train.csv", format='com.databricks.spark.csv',
  header='true', inferSchema='true')
df.take(2)
df.printSchema()
df.describe().show()

train = train.select(col("Survived"),col("Sex"),col("Embarked"),col("Pclass").cast("float"),col("Age").cast("float"),col("SibSp").cast("float"),col("Fare").cast("float"))

df.select("Survived").groupBy("Survived").count().orderBy("count", ascending = False).show()
df.select("Pclass").groupBy("Pclass").count().orderBy("count", ascending = False).show(10)
df.select("Sex").groupBy("Sex").count().orderBy("count", ascending = False).show()
df.select("Cabin").groupBy("Cabin").count().orderBy("count", ascending = False).show(10)
df.select("Embarked").groupBy("Embarked").count().orderBy("count", ascending = False).show(10)

df = df.select(col("Survived").cast("float"),
	col("Pclass").cast("float"),
	col("Sex"),
	col("Age").cast("float"),
	col("SibSp").cast("float"),
	col("Parch").cast("float"),
	col("Fare").cast("float"),
	col("Embarked"))

df = df.withColumn("AgeNA", col("Age").isNull())
df.select(mean(df.Age)).show()
df.show(10)

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, IndexToString

genderIndexer = StringIndexer(inputCol="Sex", outputCol="indexedSex")
embarkIndexer = StringIndexer(inputCol="Embarked", outputCol="indexedEmbarked")
 
# surviveIndexer = StringIndexer(inputCol="Survived", outputCol="indexedSurvived")
 
# One Hot Encoder on indexed features
genderEncoder = OneHotEncoder(inputCol="indexedSex", outputCol="sexVec")
embarkEncoder = OneHotEncoder(inputCol="indexedEmbarked", outputCol="embarkedVec")
 
# Create the vector structured data (label,features(vector))
assembler = VectorAssembler(inputCols=["Pclass","sexVec","Age","SibSp","Parch","Fare","embarkedVec"],outputCol="features")

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="Survived", featuresCol="features", maxIter=10)

# set up pipeline
pipeline = Pipeline(stages=[genderIndexer, embarkIndexer, genderEncoder,embarkEncoder, assembler, lr]) 

# split the data
from pyspark.ml.tuning import TrainValidationSplit
train, test = df.randomSplit([0.7, 0.3], seed=41)

# fit the model
model = pipeline.fit(train)

# make prediction
predictions = model.transform(test)

lrmodel = model.stages[-1]
print("Coefficients: " + str(lrmodel.coefficientMatrix))

# evaluate performance
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="Survived")
evaluator.evaluate(predictions)
