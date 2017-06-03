from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
# https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html
# https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS

sc 		  = SparkContext("local", "Recomendations")
movies  = sc.textFile("/home/mati/Documentos/2017-1/mineria de datos/Tarea3-MD/ml-10M100K/movies.dat")
ratings = sc.textFile("/home/mati/Documentos/2017-1/mineria de datos/Tarea3-MD/ml-10M100K/ratings.dat")
print movies.count()
print ratings.count()

sc.stop()