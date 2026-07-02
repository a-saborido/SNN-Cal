#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.36.04/x86_64-almalinux9.6-gcc115-opt/bin/thisroot.sh

# Default values
start=1
end=""
input_dir="/lustre/ific.uv.es/ml/uovi123/snncalo/optCalData"
output_dir=""
genflags=""

usage() {
    echo "Usage: $0 [-s <start>] -e <end> -i <input_dir> -o <output_dir> [-g \"<extra_genPhotons_flags>\"]"
    exit 1
}

# Parse command-line flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--start)
            start="$2"
            shift 2
            ;;
        -e|--end)
            end="$2"
            shift 2
            ;;
        -i|--input)
            input_dir="$2"
            shift 2
            ;;
        -o|--output)
            output_dir="$2"
            shift 2
            ;;
        -g|--genflags)
            genflags="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required fields
if [[ -z "$end" || -z "$input_dir" || -z "$output_dir" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Validate numeric parameters
if ! [[ $start =~ ^[0-9]+$ ]]; then
    echo "Error: Start value must be a positive integer."
    exit 1
fi

if ! [[ $end =~ ^[0-9]+$ ]]; then
    echo "Error: End value must be a positive integer."
    exit 1
fi

if (( start > end )); then
    echo "Error: Start must be <= end."
    exit 1
fi

# Particle types
particles=("proton" "pion")

# Main loop
for particle in "${particles[@]}"; do
    for ((i=start; i<=end; i++)); do
        input_file="${input_dir}/${particle}/${particle}_${i}.root"

        if [[ -f "$input_file" ]]; then
            echo "Processing file: $input_file"
            ./gp_opt -f "$input_file" -o "${output_dir}/${particle}" -po -r 4 $genflags
        else
            echo "Warning: File not found - $input_file"
        fi
    done
done
