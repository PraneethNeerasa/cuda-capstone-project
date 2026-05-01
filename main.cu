#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <npp.h>
#include <nppi.h>
namespace fs = std::filesystem;
bool ProcessImage(const std::string& input_path, const std::string& output_path) {
  int width, height, channels;
  unsigned char* h_src = stbi_load(input_path.c_str(), &width, &height, &channels, 1);
  if (!h_src) { std::cerr << "Failed to read: " << input_path << std::endl; return false; }
  Npp8u* d_src = nullptr; Npp8u* d_dst = nullptr;
  int src_step, dst_step;
  d_src = nppiMalloc_8u_C1(width, height, &src_step);
  d_dst = nppiMalloc_8u_C1(width, height, &dst_step);
  cudaMemcpy2D(d_src, src_step, h_src, width, width, height, cudaMemcpyHostToDevice);
  NppiSize roi = {width, height};
  nppiFilterGauss_8u_C1R(d_src, src_step, d_dst, dst_step, roi, NPP_MASK_SIZE_5_X_5);
  std::vector<unsigned char> h_dst(width * height);
  cudaMemcpy2D(h_dst.data(), width, d_dst, dst_step, width, height, cudaMemcpyDeviceToHost);
  stbi_write_png(output_path.c_str(), width, height, 1, h_dst.data(), width);
  nppiFree(d_src); nppiFree(d_dst); stbi_image_free(h_src);
  return true;
}
int main(int argc, char* argv[]) {
  std::string input_dir = "./images_png", output_dir = "./output";
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--input" && i+1 < argc) input_dir = argv[++i];
    if (arg == "--output" && i+1 < argc) output_dir = argv[++i];
  }
  fs::create_directories(output_dir);
  std::cout << "Input : " << input_dir << "\nOutput: " << output_dir << "\n---\n";
  int processed = 0, failed = 0;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    std::string ext = entry.path().extension().string();
    if (ext==".png"||ext==".jpg") {
      std::string out = output_dir+"/"+entry.path().stem().string()+"_blurred.png";
      std::cout << "Processing: " << entry.path().filename().string() << " ... ";
      if (ProcessImage(entry.path().string(), out)) { std::cout<<"OK\n"; processed++; }
      else { std::cout<<"FAILED\n"; failed++; }
    }
  }
  std::cout << "---\nDone! Processed: " << processed << " | Failed: " << failed << "\n";
  return 0;
}
