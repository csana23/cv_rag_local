#!/bin/bash

# Run any setup steps or pre-processing tasks here
echo "Running ETL to move CV data from pdf to chromadb server running in docker container..."

# Run the ETL script
python write_data_to_db.py