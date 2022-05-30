package edu.ucr.cs.cs167

import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.{col, concat_ws}

object App {
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Twitter Analysis")
    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    val spark: SparkSession.Builder = SparkSession.builder().config(conf)
    val sparkSession: SparkSession = spark.getOrCreate()

    val operation = args(0)
    try{
      // switch case on argument (add more for your parts)
      operation match{
        case "Topic-Prediction" =>
          val inputFile = args(1)
          var tweetsDF = sparkSession.read.json(inputFile)

          tweetsDF = tweetsDF.withColumn("all_text",concat_ws(" ",col("text"),col("user_description")))
          val Array(training, test) = tweetsDF.randomSplit(Array(0.7, 0.3))
          tweetsDF.show()

          // change inputCol to topic for actual data
          val indexer = new StringIndexer()
            .setInputCol("hashtag")
            .setOutputCol("label")
            .fit(tweetsDF)
          val indexed = indexer.transform(tweetsDF)

          // tokenize all text
          val tokenizer = new Tokenizer()
            .setInputCol("all_text")
            .setOutputCol("words")

          // converts tokens to numeric features
          val hashingTF = new HashingTF()
            .setNumFeatures(20)
            .setInputCol(tokenizer.getOutputCol)
            .setOutputCol("features")

          // classifier - regParam is helps prevent overfitting?
          val lr = new LogisticRegression()
            .setMaxIter(10)
          //.setRegParam(0.001)

          // pipeline
          val pipeline = new Pipeline()
            .setStages(Array(indexer,tokenizer,hashingTF,lr))

          // build model with training split
          val model = pipeline.fit(training)

          // use model on test data
          val results = model.transform(test)
            .select("id", "text", "hashtag", "user_description", "label", "prediction")

          // get and print out metrics
          val predictionAndLabels = results.select("prediction", "label")
            .rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)
            }
          val metrics = new MulticlassMetrics(predictionAndLabels)
          println("Summary Statistics")
          println(s"Accuracy = ${metrics.accuracy}")
          println(s"Weighted precision: ${metrics.weightedPrecision}")
          println(s"Weighted recall: ${metrics.weightedRecall}")
      }
    }
    finally {
      sparkSession.stop()
    }
  }

}
