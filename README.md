# GPU-Accelerated Batch Image Processor

A CUDA-based image processing pipeline that applies Gaussian blur to multiple images
in parallel using NVIDIA's NPP (NVIDIA Performance Primitives) library.

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x
- NPP libraries (included with CUDA Toolkit)

## Build
```bash
nvcc main.cu -o image_processor \
    -std=c++17 \
    -I/usr/local/cuda-12.8/include \
    -L/usr/local/cuda-12.8/targets/x86_64-linux/lib \
    -lnppc -lnppif -lnppig -lnppim -lnppist -lnppitc -lnppisu -lnppial -lnppicc -lnppidei \
    -Wno-deprecated-gpu-targets
```

## Run
```bash
./image_processor --input images_png --output output
```

### Arguments
| Argument | Description | Default |
|---|---|---|
| `--input` | Directory containing input PNG/JPG images | `./images_png` |
| `--output` | Directory to save processed images | `./output` |

## Example
```bash
./image_processor --input images_png --output output
# Input : images_png
# Output: output
# ---
# Processing: girl.png ... OK
# Processing: mandrill.png ... OK
# Processing: couple.png ... OK
# ---
# Done! Processed: 3 | Failed: 0
```

## How it works
1. Reads all PNG/JPG images from the input directory
2. Uploads each image to GPU memory using `cudaMemcpy2D`
3. Applies a 5x5 Gaussian blur kernel using `nppiFilterGauss_8u_C1R` (NPP)
4. Downloads the result back to CPU memory
5. Saves the blurred image to the output directory

## Results
See the `output/` folder for processed images and `output/execution_log.txt` for the run log.
See `output/comparison.png` for a side-by-side before/after comparison.

## Code Style
This project follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
