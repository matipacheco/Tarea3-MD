import sys, time, os, tempfile
from math import sqrt, ceil
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
# https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html
# https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS
# https://spark.apache.org/docs/1.1.1/api/python/pyspark.rdd.RDD-class.html

start_time = time.time()

# parameters = [[1, 0.01], [5, 0.01], [10, 0.01],
#               [1, 0.02], [5, 0.02], [10, 0.02]]

rank    = int(sys.argv[1])
lambda_ = float(sys.argv[2])

sc      = SparkContext("local", "Recomendations")
movies  = sc.textFile("/path/to/movies.dat")
ratings = sc.textFile("/path/to/ratings.dat")

# movies.dat  --> MovieID::Title::Genres.
# ratings.dat --> UserID::MovieID::Rating::Timestamp

movies 	= movies.map(lambda l: l.split('::'))
ratings = ratings.map(lambda l: l.split('::'))\
					.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# classmethod train(ratings, rank, iterations=5, lambda_=0.01, blocks=-1, nonnegative=False, seed=None)
# Parameters: ratings - RDD of Rating or (userID, productID, rating) tuple.
#             rank    - Rank of the feature matrices computed (number of features).
#             lambda  - Regularization parameter. (default: 0.01)

model = ALS.train(ratings, rank = rank, lambda_ = lambda_)

testdata      = ratings.map(lambda l: (l[0], l[1]))
predictions   = model.predictAll(testdata).map(lambda l: ((l[0], l[1]), l[2]))
ratesAndPreds = ratings.map(lambda l: ((l[0], l[1]), l[2])).join(predictions)

MSE  = ratesAndPreds.map(lambda l: (l[1][0] - l[1][1])**2).mean()
RMSE = sqrt(MSE)

model.save(sc, "/path/to/save/model")

file = open('results.txt', 'a')
file.write('Rank: ' + str(rank) + ', Lambda: ' + str(lambda_) + ', RMSE: ' + str(RMSE) + ', Tiempo: ' + str(ceil(time.time() - start_time)) + '[s]\n')
file.close()

sc.stop()