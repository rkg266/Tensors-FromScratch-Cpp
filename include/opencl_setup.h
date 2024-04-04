#ifndef OPENCL_SETUP_H
#define OPENCL_SETUP_H

#include <CL/cl.h> // OpenCL for parallel programming from Intel oneAPI

// Function prototypes
void SelectTargetDevice(cl_platform_id* selectedPlatform, cl_device_id* selectedDevice);
cl_context CreateOpenCLContext(cl_platform_id* selectedPlatform, cl_device_id* selectedDevice);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id* device);

#endif // OPENCL_SETUP_H
