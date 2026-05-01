#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <npp.h>
#include <nppi.h>

namespace fs = std::filesystem;

bool processImage(const std::string& inputPath, const std::string& outputPath) {
    // Read image as grayscale
    cv::Mat img = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to read: " << inputPath << std::endl;
        return false;
    }

    int width = img.cols;
    int height = img.rows;

    // Allocate GPU memory using NPP
    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

    if (!d_src || !d_dst) {
        std::cerr << "NPP malloc failed for: " << inputPath << std::endl;
        return false;
    }

    // Copy image data to GPU
    cudaMemcpy2D(d_src, srcStep, img.data, img.step,
                 width, height, cudaMemcpyHostToDevice);

    // Apply Gaussian Blur on GPU using NPP
    NppiSize roi = {width, height};
    NppStatus status = nppiFilterGauss_8u_C1R(
        d_src, srcStep, d_dst, dstStep, roi, NPP_MASK_SIZE_5_X_5);

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Gaussian filter failed, status: " << status << std::endl;
        nppiFree(d_src);
        nppiFree(d_dst);
        return false;
    }

    // Copy result back from GPU to CPU
    cv::Mat result(height, width, CV_8UC1);
    cudaMemcpy2D(result.data, result.step, d_dst, dstStep,
                 width, height, cudaMemcpyDeviceToHost);

    // Save output image
    cv::imwrite(outputPath, result);

    nppiFree(d_src);
    nppiFree(d_dst);
    return true;
}

int main(int argc, char* argv[]) {
    std::string inputDir  = "./images";
    std::string outputDir = "./output";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input"  && i + 1 < argc) inputDir  = argv[++i];
        if (arg == "--output" && i + 1 < argc) outputDir = argv[++i];
    }

    fs::create_directories(outputDir);

    std::cout << "Input  dir : " << inputDir  << std::endl;
    std::cout << "Output dir : " << outputDir << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    int processed = 0, failed = 0;

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        std::string ext = entry.path().extension().string();
        if (ext == ".tiff" || ext == ".tif" ||
            ext == ".png"  || ext == ".jpg") {

            std::string inputPath  = entry.path().string();
            std::string filename   = entry.path().stem().string();
            std::string outputPath = outputDir + "/" + filename + "_blurred.png";

            std::cout << "Processing: " << filename << " ... ";

            if (processImage(inputPath, outputPath)) {
                std::cout << "OK" << std::endl;
                processed++;
            } else {
                std::cout << "FAILED" << std::endl;
                failed++;
            }
        }
    }

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Done! Processed: " << processed
              << " | Failed: " << failed << std::endl;
    return 0;
}