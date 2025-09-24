# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()

# Load datasets
listening_logs_df = spark.read.csv("input/listening_logs.csv", header=True, inferSchema=True)
song_metadata_df = spark.read.csv("input/songs_metadata.csv", header=True, inferSchema=True)

joined_df = listening_logs_df.join(song_metadata_df, on="song_id", how="inner")

# Task 1: User Favorite Genres
user_genre_count = joined_df.groupBy("user_id", "genre").count()
w = Window.partitionBy("user_id").orderBy(col("count").desc())
favorite_genres = (
    user_genre_count.withColumn("rank", row_number().over(w))
    .filter(col("rank") == 1)
    .select("user_id", "genre", "count")
)

favorite_genres.write.format("csv").option("header", True).save("output/user_favorite_genres")

# Task 2: Average Listen Time
avg_listen_time = joined_df.groupBy("song_id", "title").agg(
    avg("duration_sec").alias("avg_duration_sec")
)

avg_listen_time.write.format("csv").option("header", True).save("output/avg_listen_time_per_song")


# Task 3: Genre Loyalty Scores
user_total = joined_df.groupBy("user_id").count().withColumnRenamed("count", "total_plays")
user_fav_genre = favorite_genres.withColumnRenamed("count", "fav_genre_plays")

loyalty_scores = user_fav_genre.join(user_total, on="user_id").withColumn(
    "loyalty_score", col("fav_genre_plays") / col("total_plays")
)

loyalty_scores.show()
loyal_users = loyalty_scores.filter(col("loyalty_score") > 0.8)

loyal_users.write.format("csv").option("header", True).save("output/genre_loyalty_scores")


# Task 4: Identify users who listen between 12 AM and 5 AM
logs_with_hour = joined_df.withColumn("hour", hour(to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss")))

night_listeners = logs_with_hour.filter((col("hour") >= 0) & (col("hour") < 5)).select("user_id").distinct()

night_listeners.write.format("csv").option("header", True).save("output/night_owl_users")



spark.stop()