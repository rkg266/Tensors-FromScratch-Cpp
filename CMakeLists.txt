cmake_minimum_required(VERSION 3.10)  # Minimum version of CMake required

project(TensorFramework VERSION 1.0)  # Your project name and version

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OpenCL_INCLUDE_DIR "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/include/sycl")
set(OpenCL_LIBRARY "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib/OpenCL.lib")
find_package(OpenCL REQUIRED)

# Include directories for header files
include_directories(include)

# Add executable to the project using the specified source files
add_executable(TensorFramework "src/main.cpp" 
								"src/Tensor.cpp" 
								"src/Operations.cpp"  "include/opencl_setup.h" "src/opencl_setup.cpp")

target_include_directories(TensorFramework PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(TensorFramework PRIVATE ${OpenCL_LIBRARIES})

# target_compile_features(TensorFramework PUBLIC cxx_std_17)
# Enable debug symbols for gdb
set(CMAKE_BUILD_TYPE Debug)