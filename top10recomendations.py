import sys, time, os, tempfile
from math import sqrt, ceil
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating, MatrixFactorizationModel 
# https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html
# https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS
# https://spark.apache.org/docs/1.1.1/api/python/pyspark.rdd.RDD-class.html

start_time = time.time()

# parameters = [[1, 0.01], [5, 0.01], [10, 0.01],
#               [1, 0.02], [5, 0.02], [10, 0.02]]

sc      = SparkContext("local", "Recomendations")
movies  = sc.textFile("/usr/local/spark/spark-2.1.1-bin-hadoop2.7/ml-10M100K/movies.dat")
ratings = sc.textFile("/usr/local/spark/spark-2.1.1-bin-hadoop2.7/ml-10M100K/ratings.dat")

# movies.dat  --> MovieID::Title::Genres.
# ratings.dat --> UserID::MovieID::Rating::Timestamp

movies 	= movies.map(lambda l: l.split('::'))
ratings = ratings.map(lambda l: l.split('::'))\
					.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

model = MatrixFactorizationModel.load(sc, "/usr/local/spark/spark-2.1.1-bin-hadoop2.7/peorModelo/")

# The format of each line is (userID, movieID, rating)
new_user_ratings1 = [
    (1,122,5),
	(1,185,5),
	(1,231,5),
	(1,292,5),
	(1,316,5),
	(1,329,5),
	(1,355,5),
	(1,356,5),
	(1,362,5),
	(1,364,5),
	(1,370,5),
	(1,377,5),
	(1,420,5),
	(1,466,5),
	(1,480,5),
	(1,520,5),
	(1,539,5),
	(1,586,5),
	(1,588,5),
	(1,589,5),
	(1,594,5),
	(1,616,5)]

new_user_ratings_RDD = sc.parallelize(new_user_ratings1)
#print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

new_user_ratings_ids = map(lambda x: x[1], new_user_ratings1) # get just movie IDs
# keep just those not on the ID list (thanks Lei Li for spotting the error!)
new_user_unrated_movies_RDD = (movies.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (1, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = model.predictAll(new_user_unrated_movies_RDD)

top_recomendations = new_user_recommendations_RDD.filter(lambda r: r[2]>=0).takeOrdered(10, key=lambda x: -x[2])
print 'Top 10 recomendations userID 1: %s' % top_recomendations

new_user_ratings2 = [
	(2,110,5),
	(2,151,3),
	(2,260,5),
	(2,376,3),
	(2,539,3),
	(2,590,5),
	(2,648,2),
	(2,719,3),
	(2,733,3),
	(2,736,3),
	(2,780,3),
	(2,786,3),
	(2,802,2),
	(2,858,2),
	(2,1049,3),
	(2,1073,3),
	(2,1210,4),
	(2,1356,3),
	(2,1391,3),
	(2,1544,3)]

new_user_ratings_RDD = sc.parallelize(new_user_ratings2)
#print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

new_user_ratings_ids = map(lambda x: x[1], new_user_ratings2) # get just movie IDs
# keep just those not on the ID list (thanks Lei Li for spotting the error!)
new_user_unrated_movies_RDD = (movies.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (2, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = model.predictAll(new_user_unrated_movies_RDD)

top_recomendations = new_user_recommendations_RDD.filter(lambda r: r[2]>=0).takeOrdered(10, key=lambda x: -x[2])
print 'Top 10 recomendations userID 2: %s' % top_recomendations

new_user_ratings3 = [
	(3,110,4.5),
	(3,151,4.5),
	(3,213,5),
	(3,590,3.5),
	(3,1148,4),
	(3,1246,4),
	(3,1252,4),
	(3,1276,3.5),
	(3,1288,3),
	(3,1408,3.5),
	(3,1552,2),
	(3,1564,4.5),
	(3,1597,4.5),
	(3,1674,4.5),
	(3,3408,4),
	(3,3684,4.5),
	(3,4535,4),
	(3,4677,4),
	(3,4995,4.5),
	(3,5299,3),
	(3,5505,2),
	(3,5527,4.5),
	(3,5952,3.5),
	(3,6287,3),
	(3,6377,4),
	(3,6539,5),
	(3,7153,4),
	(3,7155,3.5),
	(3,8529,4),
	(3,8533,4.5),
	(3,8783,5),
	(3,27821,4.5),
	(3,33750,3.5)]

new_user_ratings_RDD = sc.parallelize(new_user_ratings3)
#print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

new_user_ratings_ids = map(lambda x: x[1], new_user_ratings3) # get just movie IDs
# keep just those not on the ID list (thanks Lei Li for spotting the error!)
new_user_unrated_movies_RDD = (movies.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (3, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = model.predictAll(new_user_unrated_movies_RDD)

top_recomendations = new_user_recommendations_RDD.filter(lambda r: r[2]>=0).takeOrdered(10, key=lambda x: -x[2])
print 'Top 10 recomendations userID 3: %s' % top_recomendations

new_user_ratings4 = [
	(4,21,3),
	(4,34,5),
	(4,39,3),
	(4,110,5),
	(4,150,5),
	(4,153,5),
	(4,161,5),
	(4,165,5),
	(4,208,3),
	(4,231,1),
	(4,253,3),
	(4,266,5),
	(4,292,3),
	(4,316,5),
	(4,317,5),
	(4,329,5),
	(4,344,2),
	(4,349,3),
	(4,364,5),
	(4,367,3),
	(4,377,3),
	(4,380,3),
	(4,410,5),
	(4,420,3),
	(4,432,3),
	(4,434,3),
	(4,435,3),
	(4,440,3),
	(4,480,5),
	(4,500,5),
	(4,586,5),
	(4,587,5),
	(4,588,5),
	(4,589,5),
	(4,590,5),
	(4,592,5),
	(4,595,5),
	(4,597,3)]

new_user_ratings_RDD = sc.parallelize(new_user_ratings4)
#print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

new_user_ratings_ids = map(lambda x: x[1], new_user_ratings4) # get just movie IDs
# keep just those not on the ID list (thanks Lei Li for spotting the error!)
new_user_unrated_movies_RDD = (movies.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (4, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = model.predictAll(new_user_unrated_movies_RDD)

top_recomendations = new_user_recommendations_RDD.filter(lambda r: r[2]>=0).takeOrdered(10, key=lambda x: -x[2])
print 'Top 10 recomendations userID 4: %s' % top_recomendations

new_user_ratings5 = [
	(5,1,1),
	(5,7,3),
	(5,25,3),
	(5,28,3),
	(5,30,5),
	(5,32,5),
	(5,47,5),
	(5,52,4),
	(5,57,3),
	(5,58,3),
	(5,85,3),
	(5,111,4),
	(5,141,3),
	(5,171,3),
	(5,194,3),
	(5,230,4),
	(5,232,3),
	(5,235,4),
	(5,242,3),
	(5,249,4),
	(5,253,3),
	(5,299,3),
	(5,306,3),
	(5,307,4),
	(5,308,3),
	(5,321,3),
	(5,326,5),
	(5,334,5),
	(5,345,3),
	(5,348,4),
	(5,412,5),
	(5,446,5),
	(5,475,3),
	(5,477,3),
	(5,495,5),
	(5,508,3),
	(5,509,4),
	(5,515,3),
	(5,527,5),
	(5,532,5),
	(5,535,3),
	(5,538,5),
	(5,541,5),
	(5,562,5),
	(5,592,3),
	(5,593,4),
	(5,608,5),
	(5,648,3),
	(5,708,1),
	(5,736,1),
	(5,778,4),
	(5,780,1),
	(5,818,3),
	(5,858,4),
	(5,903,4),
	(5,912,4),
	(5,919,5),
	(5,920,5),
	(5,923,5),
	(5,926,5),
	(5,969,4),
	(5,1041,4),
	(5,1046,5),
	(5,1073,4),
	(5,1080,4),
	(5,1094,4),
	(5,1096,5),
	(5,1097,3),
	(5,1103,4),
	(5,1104,5),
	(5,1172,4),
	(5,1183,5),
	(5,1199,5),
	(5,1206,4),
	(5,1207,4),
	(5,1219,5),
	(5,1221,4),
	(5,1225,5),
	(5,1230,5),
	(5,1235,5),
	(5,1244,4),
	(5,1247,4),
	(5,1258,4),
	(5,1280,4),
	(5,1295,5),
	(5,1300,4),
	(5,1391,1)]

new_user_ratings_RDD = sc.parallelize(new_user_ratings5)
#print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

new_user_ratings_ids = map(lambda x: x[1], new_user_ratings5) # get just movie IDs
# keep just those not on the ID list (thanks Lei Li for spotting the error!)
new_user_unrated_movies_RDD = (movies.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (5, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = model.predictAll(new_user_unrated_movies_RDD)

top_recomendations = new_user_recommendations_RDD.filter(lambda r: r[2]>=0).takeOrdered(10, key=lambda x: -x[2])
print 'Top 10 recomendations userID 5: %s' % top_recomendations

sc.stop()