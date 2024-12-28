using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Spark.Sql;

class Program { 
    public static void Main(string[] args)
    {
        SparkSession spark = SparkSession
         .Builder()
         .AppName("Review Analysis")
         .GetOrCreate();

        DataFrame df = spark
         .Read()
         .Option("header", true)
         .Option("inferSchema", true)
         .Csv(args[1]);
        df.Show();

        spark.Udf().Register<string, bool>(
         "MLudf",
         (text) => Sentiment(text, args[2]));

        df.CreateOrReplaceTempView("Reviews");
        DataFrame sqlDf = spark.Sql("SELECT ReviewText, MLudf(ReviewText) FROM Reviews");
        sqlDf.Show();
        spark.Stop();
    }

    public static bool Sentiment(string text, string modelPath)
    {
        var mlContext = new MLContext();
        ITransformer mlModel = mlContext
        .Model
        .Load(modelPath, out DataViewSchema _);
        PredictionEngine<Review, ReviewPrediction> predEngine =
       mlContext
        .Model
        .CreatePredictionEngine<Review,
       ReviewPrediction>(mlModel);
        ReviewPrediction result = predEngine.Predict(
        new Review { ReviewText = text });
        // Returneaza true pentru review pozitiv, false pentru negativ
        return result.Prediction;
    }

    public class Review
    {

        [LoadColumn(0)]
        public string ReviewText;
    }

    public class ReviewPrediction : Review
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }

}





//spark-submit --class org.apache.spark.deploy.dotnet.DotnetRunner --master local C:\Users\Bacan\MLSparkModel\MLModelSentiment_FirstApp\bin\Debug\net8.0\microsoft-spark-3-0_2.12-2.1.1.jar C:\Users\Bacan\MLSparkModel\MLModelSentiment_FirstApp\bin\Debug\net8.0\MLModelSentiment_FirstApp.exe C:\Users\Bacan\MLSparkModel\MLModelSentiment_FirstApp\bin\Debug\net8.0\MLModelSEntiment_FirstApp.Program C:\Users\Bacan\MLSparkModel\MLModelSentiment_FirstApp\reviews.csv C:\Users\Bacan\MLSparkModel\MLModelSentiment_FirstApp\MLModelSentiment.zip