#!/bin/bash

# This script runs the inference cases for the given model and dataset.
models=("yolo11n")
precision=("FP16")
hardware=("GPU")
mode=("30W")
parallel=("mp_shared_memory")
max_fps=("infinite" "30")
num_objects=("88")

# Filter combinations based on allowed cases
declare -A allowed_cases
allowed_cases["FP32,GPU"]=1
allowed_cases["FP32,CPU"]=1
allowed_cases["FP16,GPU"]=1
allowed_cases["FP16,DLA0"]=1

# Loop through all combinations
for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        for hw in "${hardware[@]}"; do
            for m in "${mode[@]}"; do
                for par in "${parallel[@]}"; do
                    for fps in "${max_fps[@]}"; do
                        for num_obj in "${num_objects[@]}"; do
                            # Check if the case is allowed
                            case_key="${prec},${hw}"
                            if [[ -z "${allowed_cases[$case_key]}" ]]; then
                                echo "Skipping case: Precision=${prec}, Hardware=${hw} (not allowed)"
                                continue
                            fi
                            echo "-------------------------------------------------------------------------------------------------------"
                            echo "Running case: Model=${model}, Precision=${prec}, Hardware=${hw}, Mode=${m}, Parallel=${par}, Max FPS=${fps}, Num Objects=${num_obj}"
                            command="python3 inference.py \
                                --model $model \
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
                                echo "Error running case: Model=${model}, Precision=${prec}, Hardware=${hw}, Mode=${m}, Parallel=${par}, Max FPS=${fps}, Num Objects=${num_obj}. Continuing to the next case."
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
