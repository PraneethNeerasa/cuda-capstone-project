NVCC = nvcc
NPP_PATH = /usr/local/cuda-12.8/targets/x86_64-linux/lib
TARGET = cuda_image_processor
all: $(TARGET)
$(TARGET): main.cu stb_image.h stb_image_write.h
	$(NVCC) -std=c++17 -o $(TARGET) main.cu -L$(NPP_PATH) -lnppig -lnppif -lnppitc -lnppicc -lnppisu -lnppc
clean:
	rm -f $(TARGET)
