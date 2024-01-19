# Przykładowa implementacja
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

dataFrame = spark.read.json("s3://projekt-landing-zone/*.json")

dynamicFrame = DynamicFrame.fromDF(dataFrame, glueContext, "")
glueContext.write_dynamic_frame.from_options(
    frame=dynamicFrame,
    connection_type='s3',
    connection_options={
        'path': "s3://projekt-formatted-data", "partitionKeys": ["review_date"]
    },
    format='parquet',
)

job.commit()