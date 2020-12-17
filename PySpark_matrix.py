from pyspark import SparkContext, SparkConf, mllib
import os
import sys
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, CoordinateMatrix, MatrixEntry, RowMatrix

# Mikko Saukkoriipi 
# 013877851
# mikko.p.saukkoriipi@helsinki.fi

# Select the dataset
#dataset = "/wrk/group/grp-ddi-2020/datasets/data-2-sample.txt"
dataset = "/wrk/group/grp-ddi-2020/datasets/data-2.txt"

conf = (SparkConf()
        .setAppName("mpsaukko")
        .setMaster("spark://10.251.52.13:7077")
        .set("spark.cores.max", "8")  ##dont be too greedy ;)
        .set("spark.rdd.compress", "true")
        .set("spark.executor.memory", "64G")
        .set("spark.broadcast.compress", "true"))

sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()


# Read the data
A = sc.textFile(dataset)
A = A.map(lambda s : [float(x) for x in s.split()])

# Zip index values with cell values
A = A.zipWithIndex().map(lambda x: (x[1], x[0]))

# Print step 1 ready. With full set 1min.
print(" ")
print("Step 1 ready")
print(" ")

# Conver A to IndexedRowMatrix
A = IndexedRowMatrix(A)

# Convert A to blockmatrices and set block size.
#A = A.toBlockMatrix(1000, 1000) # Works with sample set. Data to 1 block.
A = A.toBlockMatrix(100, 1000) # Testing with full dataset

# Cache A, because it is used multiple times
A.cache()

# Print step 2 ready. With full set 3mins.
print(" ")
print("Step 2 ready")
print(" ")

# Next multiplications. We need to calculate A*AT*A.
# Size of the A is 1000000 x 1000 so size of the A*AT would be 100000*100000.
# and AT*A would be 1000*1000. For better performance use matrix multiplication 
# rule (A*AT)*A = A*(AT*A).

# Calculate A transpose
AT = A.transpose()

# Make firts multiplication AT*A
ATA = AT.multiply(A)

# Print step 3 ready. With full set 9mins.
print(" ")
print("Step 3 ready")
print("ATA rows and cols:")
print("Rows:", ATA.numRows() , "Cols:", ATA.numCols())
print(" ")

# Make second multiplication A*ATA
AATA = A.multiply(ATA)

# Print step 4 ready
print(" ")
print("Step 4 ready")
print(" ")

# Convert AATA to indexRowMatrix
AATA = AATA.toIndexedRowMatrix()

# Get first row from AATA
first_row = AATA.rows.filter(lambda x: x.index == 0).first().vector.toArray()

# Print first row
print(first_row)