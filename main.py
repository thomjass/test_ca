from ar_coefficient_module import ar_coefficient_spark
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from datetime import datetime
import itertools


def generate_time_series(n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   vague 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + vague 2
    series += 0.1 * (np.random.rand(n_steps) - 0.5)   # + bruit
    ts_df = pd.DataFrame(series.astype(np.float32), columns=['value'])
    ts_df['time'] = pd.date_range(end=datetime.today(), periods=n_steps)
    return ts_df


def main():
    spark = SparkSession.builder.appName('testCA').getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    n_steps = 50
    df_pd = generate_time_series(n_steps)

    df_ts = spark.createDataFrame(df_pd).cache()

    param = [
        {'k': i, 'coeff': j} for i, j in itertools.product(range(1, 10), range(1, 10)) if j <= i
    ]

    number_of_cores = spark.sparkContext.defaultParallelism
    spark.conf.set("spark.sql.shuffle.partitions", 9)

    print(ar_coefficient_spark.ar_coefficient_spark(spark, df_ts, param))


if __name__ == "__main__":
    main()

