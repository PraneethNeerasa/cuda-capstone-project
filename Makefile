NVCC     = nvcc
CXX_STD  = -std=c++17
OPENCV   = $(shell pkg-config --cflags --libs opencv4)
NPP_LIBS = -lnppi -lnppc
TARGET   = cuda_image_processor

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(CXX_STD) -o $(TARGET) main.cu $(OPENCV) $(NPP_LIBS)

clean:
	rm -f $(TARGET)