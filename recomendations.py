from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
# https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html
# https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS

# movies.dat  --> MovieID::Title::Genres.
# ratings.dat --> UserID::MovieID::Rating::Timestamp

sc 		  = SparkContext("local", "Recomendations")
movies  = sc.textFile("/home/mati/Documentos/2017-1/DM/Tarea3-MD/ml-10M100K/movies.dat")
ratings = sc.textFile("/home/mati/Documentos/2017-1/DM/Tarea3-MD/ml-10M100K/ratings.dat")

movies 	= movies.map(lambda l: l.split('::'))
ratings = ratings.map(lambda l: l.split('::'))\
					.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# classmethod train(ratings, rank, iterations=5, lambda_=0.01, blocks=-1, nonnegative=False, seed=None)
# Parameters:	ratings – RDD of Rating or (userID, productID, rating) tuple.

sc.stop()