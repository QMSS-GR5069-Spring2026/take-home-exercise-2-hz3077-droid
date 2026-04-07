# Databricks notebook source
f_drivers = spark.read.csv(
    "/Volumes/gr5069/raw/f1_data/drivers.csv",
    header=True,
    inferSchema=True
)

# COMMAND ----------

df_pit = spark.read.csv(
    "/Volumes/gr5069/raw/f1_data/pit_stops.csv",
    header=True,
    inferSchema=True
)

# COMMAND ----------

display(df_pit)

# COMMAND ----------

# MAGIC %md
# MAGIC Q1

# COMMAND ----------

# MAGIC %md
# MAGIC For this question, I first inspect the pit stop dataset to confirm which columns identify the race, the driver, and the pit stop time. I use the milliseconds column because it is numeric and allows me to calculate averages, minimums, and maximums directly. Since the question asks for each driver in each race, the final grouping will need to happen at the raceId and driverId level.

# COMMAND ----------

from pyspark.sql import functions as F

df_pit.printSchema()
display(df_pit)

# COMMAND ----------

df_pit.select("raceId", "driverId", "milliseconds", "duration").show(10, truncate=False)

# COMMAND ----------

df_pit_stats = df_pit.groupBy("raceId", "driverId").agg(
    F.avg("milliseconds").alias("avg_pit_time"),
    F.min("milliseconds").alias("fastest_pit"),
    F.max("milliseconds").alias("slowest_pit")
)

display(df_pit_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC To answer this question, I focused on the pit stop dataset and used the milliseconds column as the measure of pit stop time since it is numeric and easier to aggregate than the string based duration column. Because the question asks for statistics for each driver within each race, I grouped the data by both raceId and driverId. Within each group, I calculated the average pit stop time, as well as the fastest and slowest pit stops. This allows me to capture both typical performance and variation in pit stop times for each driver in a given race.
# MAGIC
# MAGIC I used the groupBy("raceId", "driverId") function to organize the data so that each group represents a single driver within a specific race. Then, I applied aggregation functions using agg(). Specifically, F.avg("milliseconds") computes the average pit stop time, F.min("milliseconds") identifies the fastest pit stop, and F.max("milliseconds") identifies the slowest one. I used the .alias() function to rename each resulting column for clarity. The result is a new dataframe where each row corresponds to one driver in one race, along with summary statistics describing their pit stop performance.

# COMMAND ----------

# MAGIC %md
# MAGIC Q2

# COMMAND ----------

# MAGIC %md
# MAGIC To answer this question, I combine pit stop performance with race outcomes. I start from the aggregated pit stop dataset, which contains the average pit stop time for each driver in each race. Since finishing position is not included in this dataset, I join it with the results dataset using raceId and driverId. I use the positionOrder column instead of position because it is numeric and consistently defined. After joining, I rank drivers within each race based on their finishing position, allowing me to compare pit stop performance relative to race results.

# COMMAND ----------

df_results = spark.read.csv(
    "/Volumes/gr5069/raw/f1_data/results.csv",
    header=True,
    inferSchema=True
)

df_results.printSchema()
display(df_results)


# COMMAND ----------

df_results.select("raceId", "driverId", "position", "positionOrder").show(20, truncate=False)

# COMMAND ----------

df_joined = df_pit_stats.join(
    df_results.select("raceId", "driverId", "positionOrder"),
    on=["raceId", "driverId"],
    how="inner"
)

display(df_joined)

# COMMAND ----------

df_ranked = df_joined.orderBy("raceId", "positionOrder")

display(df_ranked)

# COMMAND ----------

# MAGIC %md
# MAGIC The join operation combines pit stop statistics with race results by matching on raceId and driverId, ensuring that each driver’s performance is aligned with their finishing position. The positionOrder column is used because it provides a clean numeric ranking of finishing positions. I then define a window using Window.partitionBy("raceId").orderBy("positionOrder"), which groups drivers within each race and orders them by their finishing position. The rank() function assigns a rank to each driver within the race, making the ordering explicit and allowing for direct comparison of pit stop performance across finishing positions.

# COMMAND ----------

# MAGIC %md
# MAGIC Q2 extra credit (cleaned version)

# COMMAND ----------

df_results = spark.read.csv(
    "/Volumes/gr5069/raw/f1_data/results.csv",
    header=True,
    inferSchema=True
)

df_results.select("raceId", "driverId", "position", "positionOrder").show(20, truncate=False)

df_joined = df_pit_stats.join(
    df_results.select("raceId", "driverId", "positionOrder"),
    on=["raceId", "driverId"],
    how="inner"
)

df_ranked = df_joined.orderBy("raceId", "positionOrder")

display(df_ranked)

# COMMAND ----------

# MAGIC %md
# MAGIC Q3

# COMMAND ----------

# MAGIC %md
# MAGIC To answer this question, I first inspect the drivers dataset to identify which rows have missing driver codes. Before filling anything in, I need to understand how many codes are missing and what other identifying information is available for those drivers, such as surname, driver reference, or full name. This helps me determine a reasonable rule for inserting the missing values.

# COMMAND ----------

f_drivers.select("driverId", "driverRef", "number", "code", "forename", "surname") \
    .filter(F.col("code").isNull() | (F.col("code") == "\\N")) \
    .show(truncate=False)

# COMMAND ----------

f_drivers.select("driverId", "driverRef", "code", "forename", "surname").show(20, truncate=False)

# COMMAND ----------

df_drivers_clean = f_drivers.withColumn(
    "code_filled",
    F.when(
        (F.col("code") == "\\N") | F.col("code").isNull(),
        F.upper(F.substring(F.col("surname"), 1, 3))
    ).otherwise(F.col("code"))
)

display(df_drivers_clean.select("driverRef", "code", "code_filled"))

# COMMAND ----------

# MAGIC %md
# MAGIC I used the withColumn() function to create a new column called code_filled. Within this column, I applied a conditional transformation using F.when() to identify rows where the code value is missing, either as \N or null. For these cases, I generated a new code by extracting the first three characters of the surname using F.substring() and converting them to uppercase with F.upper(). For all other rows, the original code is retained using otherwise(). This ensures that only missing values are replaced while preserving existing data.

# COMMAND ----------

# MAGIC %md
# MAGIC Q4

# COMMAND ----------

# MAGIC %md
# MAGIC To determine the youngest and oldest driver in each race, I need both the drivers’ birth dates and the date on which each race took place. Age should be defined relative to the race date rather than the current date, since drivers compete at different points in time. I therefore load the races dataset and inspect its date column so I can later compare each driver’s date of birth to the corresponding race date.
# MAGIC

# COMMAND ----------

df_races = spark.read.csv(
    "/Volumes/gr5069/raw/f1_data/races.csv",
    header=True,
    inferSchema=True
)

df_races.select("raceId", "year", "name", "date").show(20, truncate=False)

# COMMAND ----------

df_age_base = df_results.select("raceId", "driverId") \
    .join(df_races.select("raceId", "date"), on="raceId", how="inner") \
    .join(f_drivers.select("driverId", "dob"), on="driverId", how="inner")

display(df_age_base)

# COMMAND ----------

df_age = df_age_base.withColumn(
    "Age",
    F.floor(F.datediff(F.col("date"), F.col("dob")) / 365)
)

display(df_age)

# COMMAND ----------

df_age_summary = df_age.groupBy("raceId").agg(
    F.min("Age").alias("youngest_age"),
    F.max("Age").alias("oldest_age")
)

display(df_age_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC I first joined the results, races, and drivers datasets to create a table containing each driver’s participation in a race along with the race date and their date of birth. I then created an Age column using datediff() to compute the difference in days between the race date and date of birth, dividing by 365 and applying floor() to convert it into whole years. Finally, I grouped the data by raceId and used min() and max() to identify the youngest and oldest drivers in each race.

# COMMAND ----------

# MAGIC %md
# MAGIC Q5

# COMMAND ----------

# MAGIC %md
# MAGIC To answer this question, I need to track each driver’s podium history across races in chronological order. I start from the race results dataset and keep each driver’s finishing position for each race. Then I join it with the races dataset so that I can order races over time. This is necessary because the question asks for the number of podium finishes a driver has “at any given race,” which implies a cumulative count up to that point in the season and across prior races.

# COMMAND ----------

df_podium_base = df_results.select("raceId", "driverId", "positionOrder") \
    .join(df_races.select("raceId", "date", "year", "round"), on="raceId", how="inner")

display(df_podium_base)

# COMMAND ----------

df_podium_flags = df_podium_base \
    .withColumn("win", F.when(F.col("positionOrder") == 1, 1).otherwise(0)) \
    .withColumn("second", F.when(F.col("positionOrder") == 2, 1).otherwise(0)) \
    .withColumn("third", F.when(F.col("positionOrder") == 3, 1).otherwise(0))

display(df_podium_flags)

# COMMAND ----------

from pyspark.sql.window import Window

window_spec = Window.partitionBy("driverId") \
    .orderBy("date", "raceId") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df_podium_counts = df_podium_flags \
    .withColumn("wins_so_far", F.sum("win").over(window_spec)) \
    .withColumn("seconds_so_far", F.sum("second").over(window_spec)) \
    .withColumn("thirds_so_far", F.sum("third").over(window_spec))

display(df_podium_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC I joined the results and races datasets so that each row contains a driver, a race, their finishing position, and the race date. I then created three binary columns using withColumn() and F.when(): win, second, and third, which take the value 1 when a driver finishes in that podium position and 0 otherwise. To make these counts cumulative, I defined a window with Window.partitionBy("driverId").orderBy("date", "raceId"), which groups rows by driver and orders their races over time. Using F.sum(...).over(window_spec), I calculated running totals for wins, second places, and third places up to each race.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Q6

# COMMAND ----------

# MAGIC %md
# MAGIC Question: Which drivers have the highest total number of podium finishes?

# COMMAND ----------

# MAGIC %md
# MAGIC For this exploration, I examine which drivers have achieved the highest total number of podium finishes across all races. I build on the cumulative podium counts calculated previously, which track each driver’s wins, second places, and third places over time. By summing these values and identifying the final total for each driver, I can compare overall podium performance across the dataset.

# COMMAND ----------

df_total_podiums = df_podium_counts.withColumn(
    "total_podiums",
    F.col("wins_so_far") + F.col("seconds_so_far") + F.col("thirds_so_far")
)

# COMMAND ----------

df_driver_podiums = df_total_podiums.groupBy("driverId").agg(
    F.max("total_podiums").alias("total_podiums")
)

df_top_drivers = df_driver_podiums.orderBy(F.col("total_podiums").desc())

display(df_top_drivers)

# COMMAND ----------

# MAGIC %md
# MAGIC I created a new column called total_podiums by summing the cumulative counts of wins, second places, and third places for each driver. Since the cumulative values increase over time, I used groupBy("driverId") and max() to extract each driver’s final total number of podium finishes. I then sorted the results in descending order using orderBy() to identify the drivers with the highest number of podiums.