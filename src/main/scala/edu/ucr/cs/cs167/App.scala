package edu.ucr.cs.cs167

import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.{array_intersect, col, concat_ws, lit, length}

object App {
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Twitter Analysis")
    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    val spark: SparkSession.Builder = SparkSession.builder().config(conf)
    val sparkSession: SparkSession = spark.getOrCreate()
    import sparkSession.implicits._


    val operation: String = args(0)
    val inputFile: String = args(1)
    var list = new Array[String](20);
    try{
      // switch case on argument (add more for your parts)
      operation match{
        case "Clean-Data" =>
          // reads in the json file
          val tweetsDF = sparkSession.read.format("json").load(inputFile)

          // selects desired columns and cleans selected file
          val cleanTweetsDF = tweetsDF.selectExpr("id", "text", "entities.hashtags.text AS hashtags",
            "user.description AS user_description", "retweet_count", "reply_count", "quoted_status_id")
          cleanTweetsDF.write.format("json").save("tweets_clean.json")

          // prints schema of cleaned data
          cleanTweetsDF.printSchema()

          // creates a temp view to view filtered data
          cleanTweetsDF.createOrReplaceTempView("tweets")

          // sql to select top 20 hashtags
          var list = new Array[String](20);
          list = sparkSession.sql(
            s"""
                SELECT explode(hashtags) as hashtags, count(*) AS count
                FROM tweets
                GROUP BY hashtags
                ORDER BY count DESC
                LIMIT 20
                """).map(f=>f.getString(0)).collect()

          // print it as a comma separated list
          println(list.mkString(","))

        case "Data-Preparation" =>
          //output from Task 1
          val top_topics = Array("ALDUBxEBLoveis","no309","FurkanPalalı","LalOn","sbhawks","DoktorlarDenklikistiyor",
            "Benimisteğim","احتاج_بالوقت_هذا","happy","السعودية","nowplaying","CNIextravaganza2017",
            "love","beautiful","art","türkiye","vegalta","KittyLive","tossademar","鯛")

          val inputFile = args(1)
          var tweetsDF = sparkSession.read.json(inputFile)
          tweetsDF = tweetsDF.withColumn("top_topics", lit(top_topics))

          //array_intersect only keeps first element
          var intersectDF = tweetsDF.withColumn("Intersect", array_intersect(col("top_topics"),col("hashtags"))(0))

          //filters out arrays with no topics
          intersectDF = intersectDF.filter(length(col("Intersect")) > 0)

          //drops hashtags col, makes topics the new hashtags
          intersectDF = intersectDF.drop("hashtags").drop("top_topics")
          intersectDF = intersectDF.toDF("id", "quoted_status_id", "reply_count", "retweet_count", "text", "user_description", "topic")

          //rearranges schema
          val finalDF = intersectDF.select("id", "text", "topic", "user_description", "retweet_count", "reply_count", "quoted_status_id")

          finalDF.write.json("tweets_topic")

        case "Topic-Prediction" =>
          println("Beginning topic prediction model building")
          val inputFile = args(1)
          var tweetsDF = sparkSession.read.json(inputFile)

          // combine text columns
          tweetsDF = tweetsDF.withColumn("all_text",concat_ws(" ",col("text"),col("user_description")))

          val indexer = new StringIndexer()
            .setInputCol("topic")
            .setOutputCol("label")
            .fit(tweetsDF)

          val tokenizer = new Tokenizer() // tokenize all text
            .setInputCol("all_text")
            .setOutputCol("words")

          val word2vec = new Word2Vec() // word2vec instead of TF.IDF - can use seed for init
            .setInputCol(tokenizer.getOutputCol)
            .setOutputCol("features")
            .setVectorSize(300)

          val lr = new LogisticRegression() // Using logistic reg as classifier
            .setMaxIter(10)

          val pipeline = new Pipeline() // pipeline
            .setStages(Array(indexer,tokenizer,word2vec,lr))

          val paramGrid = new ParamGridBuilder() // parameter grid for cross validation
            .addGrid(word2vec.minCount, Array(0, 1))
            .addGrid(lr.elasticNetParam, Array(0,0.01,0.1,0.3,0.8))
            .addGrid(lr.regParam, Array(0,0.01,0.1,0.3,0.8))
            .build()

          val cv = new CrossValidator() // Running cross validator to produce best model given parameter grid
            .setEstimator(pipeline)
            .setEvaluator(new MulticlassClassificationEvaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(3)  // Use 3+ in practice
            .setParallelism(3)  // Evaluate up to 3 parameter settings in parallel docs say ~10

          val Array(training, test) = tweetsDF.randomSplit(Array(0.7, 0.3)) // splits data into training and test - can use seed
          val model = cv.fit(training) // build model with training split
          val minCount = model.bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[Word2VecModel].getMinCount
          val elasticRegParam = model.bestModel.asInstanceOf[PipelineModel].stages(3).asInstanceOf[LogisticRegressionModel].getElasticNetParam
          val RegParam = model.bestModel.asInstanceOf[PipelineModel].stages(3).asInstanceOf[LogisticRegressionModel].getRegParam

          model.write.overwrite().save("models/logistic-regression-model") // save model

          val results = model.transform(test) // use model on test data
            .select("id", "text", "topic", "user_description", "label", "prediction")

          // get and print out metrics
          val multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
          val metrics = multiclassClassificationEvaluator.getMetrics(results)
          println("Summary Statistics")
          println(s"Accuracy = ${metrics.accuracy}")
          println(s"Weighted precision: ${metrics.weightedPrecision}")
          println(s"Weighted recall: ${metrics.weightedRecall}")
          println("Best model parameters")
          println(s"Best minCount: $minCount")
          println(s"Best elasticRegParam: $elasticRegParam")
          println(s"Best RegParam: $RegParam")
      }
    }
    finally {
      sparkSession.stop()
    }
  }

}
