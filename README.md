# CUDA Image Processing — Gaussian Blur with NPP

## Description
This project applies a **Gaussian Blur** filter to a large batch of grayscale
images using the **CUDA NPP (NVIDIA Performance Primitives)** library.
It processes all images in a given input directory and saves the blurred
results to an output directory.

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit (11.x or later)
- OpenCV 4 (with TIFF support)
- CUDA NPP library (included with CUDA Toolkit)

## Build
```bash
make
```

## Run
```bash
./cuda_image_processor --input ./images --output ./output
```

## Arguments
| Argument   | Description                        | Default    |
|------------|------------------------------------|------------|
| `--input`  | Directory containing input images  | `./images` |
| `--output` | Directory to save processed images | `./output` |

## How It Works
1. Reads each `.tiff`, `.png`, or `.jpg` image from the input directory
2. Uploads image data to GPU memory using NPP malloc
3. Applies 5x5 Gaussian Blur using `nppiFilterGauss_8u_C1R`
4. Copies blurred result back to CPU
5. Saves output as PNG in the output directory

## Dataset
Images sourced from the USC SIPI Image Database (39 grayscale TIFF images).