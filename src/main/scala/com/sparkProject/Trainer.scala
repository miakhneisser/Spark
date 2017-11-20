package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.regression.LabeledPoint

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/

   val parquetFileDF = spark.read.parquet("/home/mia/Documents/spark_tp/TP_ParisTech_2017_2018_starter/data/prepared_trainingset")
    print("parquet file")
    print("*****************************************************************************************************************************")

    //parquetFileDF.show()

    /** TF-IDF **/
    //2-a)

    val tokenizer​= new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")


    //val regexTokenized= tokenizer​.transform(parquetFileDF)
    //regexTokenized.select("tokens").show()


    //2-b)
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    print("*****************************************************************************************************************************")
    print("remover")
    print("*****************************************************************************************************************************")
    //val removed= remover.transform(regexTokenized)
    //removed.select("filtered").show()

    //2-c)
    print("*****************************************************************************************************************************")
    print("TF-Count")
    print("*****************************************************************************************************************************")

    val cvModel: CountVectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("tf-count")

    //val idf=cvModel.transform(removed)
    //idf.select("tf-count").show()

    //2-d)
    print("*****************************************************************************************************************************")
    print("IDF")
    print("*****************************************************************************************************************************")

    val idf2 = new IDF().setInputCol("tf-count").setOutputCol("tfidf")
    //val idfModel2 = idf2.fit(idf)

    //val rescaledData = idfModel2.transform(idf)
    //rescaledData.select("tfidf").show()

    print("ALLLLLLLLLLLLLLLLLLLLLLLLLLL")
    //rescaledData.show()

    //3-e)

    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    print("*****************************************************************************************************************************")
    print("Country Indexer")
    print("*****************************************************************************************************************************")

    //val indexed = indexer_country.fit(rescaledData).transform(rescaledData)
    //indexed.select("country_indexed").show()

    //3-f)
    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    print("*****************************************************************************************************************************")
    print("Currency Indexer")
    print("*****************************************************************************************************************************")

    //val indexed2 = indexer_currency.fit(indexed).transform(indexed)
    //indexed2.show()

    //4-g)


    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    //val output = assembler.transform(indexed2)
    //output.show()

    //4-h)
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)


    //4-i)
    val pipeline= new Pipeline().setStages(Array(tokenizer​, remover, cvModel, idf2, indexer_country, indexer_currency, assembler, lr))

    //5-j)
    val Array(training, test) = parquetFileDF.randomSplit(Array(0.9, 0.1), seed = 12345)
    //added just to try
    //val model = pipeline.fit(training)


    //5-k)
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(cvModel.minDF, Array(55.0 , 75.0, 95.0))
      .build()

    print("---------------------------paramGrid-------------------------")


    val f1Score= new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")



    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
        .setEvaluator(f1Score)
        .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)


    val model = trainValidationSplit.fit(training)
    val df_WithPredictions=model.transform(test)

    print("f1 evaluate : ", f1Score.evaluate(df_WithPredictions))

    df_WithPredictions.groupBy("final_status", "predictions").count.show()



    /** VECTOR ASSEMBLER **/


    /** MODEL **/


    /** PIPELINE **/


    /** TRAINING AND GRID-SEARCH **/

  }
}
