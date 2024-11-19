#!/bin/bash

prefix="unet_checkpoint-"
repo="hf_username/repo_name"

# *********************************

find_last_checkpoint() {
    local max_steps=0
    local last_checkpoint=""

    for dir in ${prefix}*; do
        if [ -d "$dir" ]; then
            num=$(echo "$dir" | sed -E "s/${prefix}([0-9]+)$/\1/")
            
            if [[ "$num" =~ ^[0-9]+$ ]] && (( num > max_steps )); then
                max_steps=$num
                last_checkpoint=$dir
            fi
        fi
    done

    echo "$last_checkpoint"
}

last_checkpoint=$(find_last_checkpoint)

#echo "Enter the local path to upload:"
#read -e upload_path
#echo $last_checkpoint

huggingface-cli upload --repo-type=model $repo $last_checkpoint $last_checkpoint