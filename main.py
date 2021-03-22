import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from pyspark.sql import SparkSession
from datetime import datetime
import itertools
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pyspark.sql.functions as F
from statsmodels.tsa.ar_model import AR
from pyspark.sql.types import *
from tsfresh.feature_extraction.feature_calculators import ar_coefficient

schema = StructType([
    StructField("k", IntegerType()),
    StructField("coeff", IntegerType()),
    StructField("value_ar", DoubleType()),
])


def computeARk_generator(dict_params):
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def computeARk(pdf):
        # order od pdf is not guaranteed by the groupby
        pdf_ordered = pdf.sort_values('time')
        k = pdf.iloc[0]['k']
        try:
            calculated_AR = AR(pdf_ordered['value'].to_list())
            calculated_ar_params = calculated_AR.fit(maxlag=k, solver="mle").params
        except (LinAlgError, ValueError):
            calculated_ar_params = [np.NaN] * k
        p_to_get = dict_params[k]
        dict_res = {'k': [], 'coeff': [], 'value_ar': []}
        for p in p_to_get:
            dict_res['k'].append(k)
            dict_res['coeff'].append(p)
            try:
                dict_res['value_ar'].append(calculated_ar_params[p])
            except IndexError:
                dict_res['value_ar'].append(0)
        return pd.DataFrame(dict_res)
    return computeARk


def ar_coefficient_spark(spark, df, param):
    # We convert the params into a pandas dataframe to apply a groupBy
    # and to convert a part of the dataframe to a pyspark dataframe
    df_param = pd.DataFrame(param)
    dict_params = df_param.groupby('k')['coeff'].apply(list).to_dict()
    df_k = spark.createDataFrame(df_param[['k']].drop_duplicates())
    # We will apply a pandas udf to each partition which is composed of the full time series
    # multiplied k times because for each k we will compute AR(k) in parallel in the cluster
    df_ts_k = df.crossJoin(F.broadcast(df_k))
    df_value_ar = df_ts_k.groupBy('k').apply(computeARk_generator(dict_params))
    return df_value_ar.rdd.map(lambda x : ("coeff_{}__k_{}".format(x.coeff, x.k), x.value_ar)).collect()


# Function to create a random time series with a size on n_steps timestamps
# Two sinus waves combined with noise
# Return a pandas Dataframe with two columns: time and value
def generate_time_series(n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # vague 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + vague 2
    series += 0.1 * (np.random.rand(n_steps) - 0.5)   # + bruit
    ts_df = pd.DataFrame(series.astype(np.float32), columns=['value'])
    ts_df['time'] = pd.date_range(end=datetime.today(), periods=n_steps)
    return ts_df


def main():
    spark = SparkSession.builder.appName('testCA').getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.shuffle.partitions", 9)

    n_steps = 100
    df_pd = generate_time_series(n_steps)

    df_ts = spark.createDataFrame(df_pd).cache()

    param = [
        {'k': i, 'coeff': j} for i, j in itertools.product(range(1, 9), range(1, 9)) if j <= i
    ]
    res_spark = ar_coefficient_spark(spark, df_ts, param)
    res_attendu = ar_coefficient(df_pd.value, param)
    print("Res SPARK:")
    print(sorted(res_spark, key=lambda tup: tup[0]))
    print("Res Attendu:")
    print(sorted(res_attendu, key=lambda tup: tup[0]))


if __name__ == "__main__":
    main()

