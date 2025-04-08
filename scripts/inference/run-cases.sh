#!/bin/bash

# This script runs the inference cases for the given model and dataset.

model_size=("n" "s" "m")
precision=("FP16")
hardware=("GPU" "DLA0")
mode=("MAXN")
parallel=("threads" "mp" "mp_shared_memory" "mp_hardware")
max_fps=30

# Loop through all combinations
for size in "${model_size[@]}"; do
    for prec in "${precision[@]}"; do
        for hw in "${hardware[@]}"; do
            for m in "${mode[@]}"; do
                for par in "${parallel[@]}"; do
                    echo "-------------------------------------------------------------------------------------------------------"
                    echo "Running case: Model=${size}, Precision=${prec}, Hardware=${hw}, Mode=${m}, Parallel=${par}"
                    python3 inference.py \
                        --model_size $size \
                        --precision $prec \
                        --hardware $hw \
                        --mode $m \
                        --parallel $par \
                        --max_fps $max_fps

                    # Check if the command was successful
                    if [ $? -ne 0 ]; then
                        echo "Error running case: Model=${size}, Precision=${prec}, Hardware=${hw}, Mode=${m}, Parallel=${par}"
                        exit 1
                    fi
                    echo
                    echo
                    echo
                done
            done
        done
    done
done
