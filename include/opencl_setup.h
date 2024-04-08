#ifndef OPENCL_SETUP_H
#define OPENCL_SETUP_H

#include <CL/cl.h> // OpenCL for parallel programming from Intel oneAPI
#include "Globals.hpp"
#include <vector>


// Function prototypes
void SelectTargetDevice(cl_platform_id* selectedPlatform, cl_device_id* selectedDevice);
cl_context CreateOpenCLContext(cl_platform_id* selectedPlatform, cl_device_id* selectedDevice);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id* device);

// Kernel based operations
void MatrixMultiplyKernelBased(std::vector<dataType>& input1, std::vector<int>& shape1,
    std::vector<dataType>& input2, std::vector<int>& shape2,
    std::vector<dataType>& output, const char** KernelSource);

void PrintKernelBuildLog(cl_program program, cl_device_id device);

#endif // OPENCL_SETUP_H
