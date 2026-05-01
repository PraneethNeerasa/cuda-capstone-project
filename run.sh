#!/bin/bash
mkdir -p output
./cuda_image_processor --input ./images --output ./output
echo "Done! Check the output/ folder for results."