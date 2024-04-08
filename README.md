# Tensors-FromScratch-Cpp
A C++ library for tensor operations. A personal project undertaken to apply OOPS concepts, design patterns and obtain deep insights into memory management and tensor computations using Parallel Programming with OpenCL, OpenMP.

## Setting up OpenCL: 
**System details:** OS - Windows 11; CPU - 12th Gen Intel(R) Core(TM) i7-12700H, GPU (integrated) - Intel(R) Iris(R) Xe Graphics
* Download Intel oneAPI Base Toolkit, available free, from Intel Developer Zone: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
* Make sure that Intel oneAPI DPC++ Library is installed. There are also other libraries offered in the toolkit.
* I did custom installation where I avoided the libraries - Math Kernel, Data Analytics, Deep Neural Network to save space.
* Make note of the installation location, as you need to include the paths in your CMake file. <br>
Example: (CMakeLists.txt)
``` cmake
set(OpenCL_INCLUDE_DIR "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/include/sycl")
set(OpenCL_LIBRARY "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib/OpenCL.lib")
find_package(OpenCL REQUIRED)

target_include_directories(TensorFramework PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(TensorFramework PRIVATE ${OpenCL_LIBRARIES})
```

Refer Intel® oneAPI Programming Guide for details:
https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2024-1/overview.html 

## Project File Organization

├── src/ <br>
│ ├── main.cpp - Entry point of the project. <br>
│ ├── opencl_setup.cpp - Select device, create context, execute OpenCL kernels. <br>
│ ├── Operations.cpp - Operations on Tensors defined for CPU and GPU classes separately. <br>
│ └── Tensor.cpp -Tensor class definitions and functionalities. <br>
│  <br>
├── include/ <br>
│ ├── Globals.hpp - Global variables, settings. <br>
│ ├── opencl_kernels.h - Kernel implementations declared as C strings. <br>
│ ├── opencl_setup.h - Setup functions declared. <br>
│ ├── Operations.hpp - CPU and GPU classes declared. <br>
│ ├── Tensor.hpp - Tensor and its proxy class declared. <br>
│ └── Testing.hpp - Full testing routines being written here. <br>
│ <br>
├── CMakeLists.txt - CMake configuration. <br>
└── README.md - This file. 

## More details will be updated soon...
