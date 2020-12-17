from pyspark import SparkContext, SparkConf
import os
import sys
import math

# Mikko Saukkoriipi 
# 013877851
# mikko.p.saukkoriipi@helsinki.fi

# Datasets path on shared group directory on Ukko2.
dataset = "/wrk/group/grp-ddi-2020/datasets/data-1-sample.txt"
#dataset = "/wrk/group/grp-ddi-2020/datasets/data-1.txt"

conf = (SparkConf()
        .setAppName("mpsaukko")   ##change app name to your username
        .setMaster("spark://10.251.52.13:7077")
        .set("spark.cores.max", "10")  ##dont be too greedy ;)
        .set("spark.rdd.compress", "true")
        .set("spark.executor.memory", "32G")
        .set("spark.broadcast.compress", "true"))
sc = SparkContext(conf=conf)

# Read the data
data = sc.textFile(dataset)
data = data.map(lambda s: float(s))
data.cache()

# Define functio find_median
def find_median(data):
  count = data.count()
  middle = round(count / 2) + 1

  # If number of numbers in list is odd
  if (count % 2 != 0):
    return quickselect(data, middle)

  # If number of numbers in file is even, then calculate average of two middle numbers
  else:
    middle1 = quickselect(data, middle)
    middle2 = quickselect(data, middle - 1)
    return (middle1 + middle2) / 2

def quickselect(array, k):

  # Select pivot number
  pivot = array.takeSample(False, 1, seed=0)
  pivot = pivot[0]

  # Create two arrays. One with values more than pivot and one with values greater than pivot.
  numbers_less = array.filter(lambda x: x < pivot)
  numbers_greater = array.filter(lambda x: x > pivot)

  # If middle value found, then return pivot
  if k == (numbers_less.count() + 1):
    return pivot

  # If k<= number_less, then run quickselect again.
  if k <= (numbers_less.count()):
    return quickselect(numbers_less, k)

  # If k > number less, then run quickselect again.
  if k > (numbers_less.count() + 1):
    return quickselect(numbers_greater, k - (numbers_less.count() + 1))

# Find and print median
median = find_median(data)
print(" ")
print("--------- Median is: " + str(median) + " ''''''''''''")
print(" ")