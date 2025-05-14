#!/bin/bash

# This script runs the inference cases for the given model and dataset.

model_size=("n")
precision=("FP16")
hardware=("GPU")
mode=("MAXN")
parallel=("mp_shared_memory")
max_fps=("30" "infinite")
num_objects=("40" "88" "variable")

# Loop through all combinations
for size in "${model_size[@]}"; do
    for prec in "${precision[@]}"; do
        for hw in "${hardware[@]}"; do
            for m in "${mode[@]}"; do
                for par in "${parallel[@]}"; do
                    for fps in "${max_fps[@]}"; do
                        for num_obj in "${num_objects[@]}"; do
                            echo "-------------------------------------------------------------------------------------------------------"
                            echo "Running case: Model=${size}, Precision=${prec}, Hardware=${hw}, Mode=${m}, Parallel=${par}, Max FPS=${fps}, Num Objects=${num_obj}"
                            command="python3 inference.py \
                                --model_size $size \
                                --precision $prec \
                                --hardware $hw \
                                --mode $m \
                                --parallel $par \
                                --num_objects $num_obj"

                            # Add max_fps only if it's not "infinite"
                            if [ "$fps" != "infinite" ]; then
                                command="$command --max_fps $fps"
                            fi

                            eval $command

                            # Check if the command was successful
                            if [ $? -ne 0 ]; then
                                echo "Error running case: Model=${size}, Precision=${prec}, Hardware=${hw}, Mode=${m}, Parallel=${par}, Max FPS=${fps}, Num Objects=${num_obj}"
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
    done
done