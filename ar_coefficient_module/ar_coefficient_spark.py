import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from statsmodels.tsa.ar_model import AR
from numpy.linalg import LinAlgError


def computeARk_generator(dict_params):
    @pandas_udf("k long, coeff long, value_ar double", PandasUDFType.GROUPED_MAP)
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
    df_param = pd.DataFrame(param)
    dict_params = df_param.groupby('k')['coeff'].apply(list).to_dict()
    df_k = spark.createDataFrame(df_param[['k']].drop_duplicates())
    df_ts_k = df.crossJoin(F.broadcast(df_k))
    df_value_ar = df_ts_k.groupBy('k').apply(computeARk_generator(dict_params))
    return df_value_ar.rdd.map(lambda x: ("coeff_" + str(x.coeff) + "__k_" + str(x.k), x.value_ar)).collect()
