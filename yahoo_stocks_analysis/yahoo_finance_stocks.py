import yfinance as yf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sys import argv
from datetime import datetime, timedelta

today = datetime.today().date().strftime('%Y-%m-%d')
def fetch_and_save_data(ticker: str, start_date: str, end_date: str, output_path: str):

    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.reset_index()
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data.to_parquet(output_path, index=False)

def read_data_into_spark(input_path: str):

    spark = SparkSession.builder.appName("YahooFinanceReader").getOrCreate()
    df = spark.read.parquet(input_path)
    return df

def get_stats(data, timewindow):
    windowSpec = Window.partitionBy("Date").orderBy("Date").rowsBetween(-timewindow, timewindow)  # 11-day window
    data = data.withColumn("MovingAvg", F.avg("Close").over(windowSpec))
    windowSpecVolume = Window.orderBy("Date").rowsBetween(-timewindow, timewindow)
    # Average Volume
    data = data.withColumn("AvgVolume", F.avg("Volume").over(windowSpecVolume)).withColumn("HighVolume", F.when(
        F.col("Volume") > 1.5 * F.col("AvgVolume"), "High").otherwise("Normal"))
    # Calculate moving average
    print("Moving Average")
    data = data.withColumn("MovingAvg", F.avg("Close").over(windowSpec))
    data = data.withColumn("PriceChange", F.col("Close") - F.col("Open")).withColumn("Volatility",
                                                                                 F.stddev("PriceChange").over(
                                                                                     Window.rowsBetween(-5, 5)))
    data = data.withColumn("DailyRange", F.col("High") - F.col("Low")).orderBy(F.col("DailyRange").desc())
    return data
if __name__ == "__main__":
    # Define parameters
    # ticker = "AAPL"
    # start_date = "2020-01-01"
    # end_date = "2022-01-01"
    # output_path = "AAPL_data.parquet"
    ticker = argv[1] # Ticker is the name of the stock like "AAPL"
    time_window = int(argv[2])
    start_date = "1700-01-01"
    end_date = today
    output_path = ticker+'.'+'parquet'

    # Fetch data and save it
    fetch_and_save_data(ticker, start_date, end_date, output_path)

    # Read data into PySpark
    df = read_data_into_spark(output_path)
    df.repartition("Date")
    print("Original Data")
    df.show()
    # Generate Descriptive Statistics
    df.describe().show()
    data_df = get_stats(df, time_window)
    print("Correlation between Open and Close:", data_df.stat.corr('Open', 'Close'))
    print("Correlation between High and Low:", data_df.stat.corr('High', 'Low'))
    data_df.show()
