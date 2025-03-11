#!/bin/bash

# Check if the user provided the folders file
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <folders_list.txt> <base_path> <main_code> <params>"
    exit 1
fi

FOLDER_LIST=$1
BASE_PATH=$2
MAIN_PY_CODE=$3
PARAMS_FILE=$4

# Check if the folder list file exists
if [ ! -f "$FOLDER_LIST" ]; then
    echo "Error: File $FOLDER_LIST not found."
    exit 1
fi

# Loop over each folder in the list
while IFS= read -r folder; do
    FULL_PATH="$BASE_PATH/$folder"
    if [ -d "$FULL_PATH" ]; then
        echo "Processing folder: $FULL_PATH"
        (cd "$FULL_PATH" && python3 ${MAIN_PY_CODE} ${PARAMS_FILE})
    else
        echo "Warning: Folder $FULL_PATH does not exist, skipping."
    fi
done < "$FOLDER_LIST"

