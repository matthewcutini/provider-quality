# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Preparing Data for Retrieval-Augmented Generation (RAG)
# MAGIC
# MAGIC The objective of this lab is to demonstrate the process of ingesting and processing documents for a Retrieval-Augmented Generation (RAG) application. This involves extracting text from PDF documents, computing embeddings using a foundation model, and storing the embeddings in a Delta table.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks:
# MAGIC
# MAGIC * **Task 1 :** Read the PDF files and load them into a DataFrame.
# MAGIC
# MAGIC * **Task 2 :** Extract the text content from the PDFs and split it into manageable chunks.
# MAGIC
# MAGIC * **Task 3 :** Compute embeddings for each text chunk using a foundation model endpoint.
# MAGIC
# MAGIC * **Task 4 :** Create a Delta table to store the computed embeddings.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**

# COMMAND ----------

# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.0.319 llama-index==0.9.3 pydantic==1.10.9 mlflow==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog_name = 'mcutini'
schema_name = 'synthea'
articles_path = '/Volumes/mcutini/synthea/landing/notes/'
table_name = 'clinical_notes'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Read the PDF files and load them into a DataFrame.
# MAGIC
# MAGIC To start, you need to load the PDF files into a DataFrame.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC 1. Use Spark to load the binary PDFs into a DataFrame.
# MAGIC
# MAGIC 2. Ensure that each PDF file is represented as a separate record in the DataFrame.

# COMMAND ----------

# Run this cell to import the required libraries
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index import Document, set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator
from pyspark.sql.functions import col, udf, length, pandas_udf, explode
import os
import pandas as pd 
from unstructured.partition.auto import partition
import io

# COMMAND ----------

# # Use Spark to load the txt files into a DataFrame

# Define the path to the notes folder
notes_path = "/Volumes/mcutini/synthea/synthetic_files_raw/output/notes/"

# Get all files in the notes folder
note_files = [f for f in os.listdir(notes_path) if os.path.isfile(os.path.join(notes_path, f))]

# Create a list to hold the file data
data = []

# Iterate over each file and read its content
for note_file in note_files:
    file_path = os.path.join(notes_path, note_file)
    with open(file_path, 'r') as file:
        file_text = file.read()
        data.append((note_file, file_text))

# Create a DataFrame from the data
df = spark.createDataFrame(data, ["file_name", "file_text"])

# Write the DataFrame to a Delta table
df.write.format("delta").mode("overwrite").saveAsTable("mcutini.synthea.notes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Extract the text content from the PDFs and split it into manageable chunks
# MAGIC
# MAGIC Next, extract the text content from the PDFs and split it into manageable chunks.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC 1. Define a function to split the text content into chunks.
# MAGIC
# MAGIC     * Split the text content into manageable chunks.
# MAGIC
# MAGIC     * Ensure each chunk contains a reasonable amount of text for processing.
# MAGIC
# MAGIC 2. Apply the function to the DataFrame to create a new DataFrame with the text chunks.
# MAGIC

# COMMAND ----------

import time
from unstructured.partition.auto import partition
import re
import io

def extract_doc_text(x):
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document
  # concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections]) 

# COMMAND ----------

from pyspark.sql.functions import explode
from pyspark.sql.types import ArrayType, StringType

# Define a function to split the text content into chunks
def read_as_chunk(file_text):
    from transformers import AutoTokenizer
    from llama_index import Document
    from unstructured.partition.auto import extract_text

    # Set llama2 as tokenizer
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    
    # Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    
    def extract_and_split(b):
        # Ensure the input is in bytes
        if isinstance(b, str):
            b = b.encode('utf-8')
        txt = extract_doc_text(b)
        nodes = splitter.get_nodes_from_documents([Document(text=txt)])
        return [n.text for n in nodes]

    return extract_and_split(file_text)

# Apply the function to the DataFrame
def process_dataframe(df):
    # Collect the DataFrame to the driver
    data = df.collect()
    
    # Process each row
    processed_data = []
    for row in data:
        file_name = row['file_name']
        file_text = row['file_text']
        chunks = read_as_chunk(file_text)
        for chunk in chunks:
            processed_data.append((file_name, chunk))
    
    # Create a new DataFrame from the processed data
    return spark.createDataFrame(processed_data, ["file_name", "content"])

# Process the DataFrame
df_chunks = process_dataframe(df)
display(df_chunks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Compute embeddings for each text chunk using a foundation model endpoint
# MAGIC Now, compute embeddings for each text chunk using a foundation model endpoint.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC 1. Define a function to compute embeddings for text chunks.
# MAGIC     + Use a foundation model endpoint to compute embeddings for each text chunk.
# MAGIC     + Ensure that the embeddings are computed efficiently, considering the limitations of the model.  
# MAGIC
# MAGIC 2. Apply the function to the DataFrame containing the text chunks to compute embeddings for each chunk.
# MAGIC

# COMMAND ----------

# Define a function to compute embeddings for text chunks
@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    # define deployment client
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        # calculate embeddings using the deployment client's predict function 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input":batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.to_list())

    return pd.Series(all_embeddings)
    
df_chunk_emd = (df_chunks
                .withColumn("embedding", get_embedding("content"))
                .selectExpr('pdf_name', 'content', 'embedding')
               )
display(df_chunk_emd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Create a Delta table to store the computed embeddings
# MAGIC
# MAGIC Finally, create a Delta table to store the computed embeddings.
# MAGIC
# MAGIC Steps:
# MAGIC
# MAGIC   1. Define the schema for the Delta table.
# MAGIC
# MAGIC   2. Save the DataFrame containing the computed embeddings as a Delta table.
# MAGIC
# MAGIC
# MAGIC **Note:** Ensure that the Delta table is properly structured to facilitate efficient querying and retrieval of the embeddings.
# MAGIC
# MAGIC **📌 Instructions:** 
# MAGIC
# MAGIC Please execute the following SQL code block to create the Delta table. This table will store the computed embeddings along with other relevant information. 
# MAGIC
# MAGIC **Importance:** Storing the computed embeddings in a structured format like a Delta table ensures efficient querying and retrieval of the embeddings when needed for various downstream tasks such as retrieval-augmented generation. Additionally, setting the `delta.enableChangeDataFeed` property to true enables Change Data Feed (CDC), which is required for VectorSearch to efficiently process changes in the Delta table.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS pdf_text_embeddings (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   pdf_name STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC   -- Note: the table has to be CDC because VectorSearch is using DLT that is requiring CDC state
# MAGIC   ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# Define the schema for the Delta table
#Create Delta table
embedding_table_name = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_embeddings"
# Save the DataFrame as a Delta table
df_chunk_emd.write.mode("append").saveAsTable(embedding_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from pdf_text_embeddings limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC **🚨 Warning:** Please refrain from deleting the catalog and tables created in this lab, as they are required for upcoming labs. To clean up the classroom assets, execute the classroom clean-up script provided in the final lab.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you learned how to prepare data for Retrieval-Augmented Generation (RAG) applications. By extracting text from PDF documents, computing embeddings, and storing them in a Delta table, you can enhance the capabilities of language models to generate more accurate and relevant responses.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
