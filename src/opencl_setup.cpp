#include "opencl_setup.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Available platforms and devices on my PC
*******************************************
* Platform 0: Intel(R) OpenCL Graphics
  Device 0: Intel(R) Iris(R) Xe Graphics
    OpenCL Version: OpenCL 3.0 NEO
* Platform 1: Intel(R) OpenCL
  Device 0: 12th Gen Intel(R) Core(TM) i7-12700H
    OpenCL Version: OpenCL 3.0 (Build 0)
* Platform 2: Intel(R) FPGA Emulation Platform for OpenCL(TM)
  Device 0: Intel(R) FPGA Emulation Device
    OpenCL Version: OpenCL 1.2
* Platform 3: Intel(R) FPGA SDK for OpenCL(TM)
*/

void SelectTargetDevice(cl_platform_id* selectedPlatform, cl_device_id* selectedDevice) {
    cl_int clStatus;
    cl_uint numPlatforms;
    cl_platform_id* platforms = NULL;
    const char* targetPlatformName = "Intel(R) OpenCL Graphics";

    // Get the number of platforms
    clStatus = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (clStatus != CL_SUCCESS || numPlatforms == 0) {
        printf("Failed to find any OpenCL platforms.\n");
        exit(EXIT_FAILURE);
    }

    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    clStatus = clGetPlatformIDs(numPlatforms, platforms, NULL);

    for (cl_uint i = 0; i < numPlatforms; ++i) {
        size_t nameSize;
        clStatus = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &nameSize);
        char* platformName = (char*)malloc(nameSize);
        clStatus = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, nameSize, platformName, NULL);

        if (strcmp(platformName, targetPlatformName) == 0) {
            printf("Selected platform: %s\n", platformName);
            *selectedPlatform = platforms[i];
            free(platformName);
            break;
        }
        free(platformName);
    }

    if (*selectedPlatform == NULL) {
        printf("Target platform '%s' not found.\n", targetPlatformName);
        free(platforms);
        exit(EXIT_FAILURE);
    }

    // Select the first device of the chosen platform
    cl_uint numDevices;
    clStatus = clGetDeviceIDs(*selectedPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    if (clStatus != CL_SUCCESS || numDevices == 0) {
        printf("Failed to find any devices on the selected platform.\n");
        free(platforms);
        exit(EXIT_FAILURE);
    }

    cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
    clStatus = clGetDeviceIDs(*selectedPlatform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    *selectedDevice = devices[0]; // Select the first device

    free(devices);
    free(platforms);
}


cl_context CreateOpenCLContext(cl_platform_id* selectedPlatform, cl_device_id* selectedDevice) {
    cl_int clStatus;
    cl_context context = NULL;

    // Ensure the selectedPlatform and selectedDevice are valid pointers
    if (!selectedPlatform || !selectedDevice) {
        printf("Invalid platform or device pointer.\n");
        exit(EXIT_FAILURE);
    }

    // Context properties list, including the selected platform
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)*selectedPlatform,
        0 // Terminate the list with 0
    };

    // Create a context for the selected device
    context = clCreateContext(contextProperties, 1, selectedDevice, NULL, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("Failed to create an OpenCL context. Error %d\n", clStatus);
        exit(EXIT_FAILURE);
    }

    return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id* device) {
    cl_int clStatus;
    cl_command_queue queue = NULL;

    // Ensure the device pointer is valid
    if (!device) {
        printf("Invalid device pointer.\n");
        exit(EXIT_FAILURE);
    }

    // Create the command queue
#ifdef CL_VERSION_2_0
// For OpenCL 2.0 and above, use clCreateCommandQueueWithProperties
    const cl_command_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0 // Terminate the list
    };
    queue = clCreateCommandQueueWithProperties(context, *device, properties, &clStatus);
#else
// For older versions, use clCreateCommandQueue
    queue = clCreateCommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE, &clStatus);
#endif

    if (clStatus != CL_SUCCESS) {
        printf("Failed to create a command queue. Error %d\n", clStatus);
        exit(EXIT_FAILURE);
    }

    return queue;
}


/**************** Kernel based operations ******************/

void MatrixMultiplyKernelBased(std::vector<dataType>& input1, std::vector<int>& shape1,
    std::vector<dataType>& input2, std::vector<int>& shape2,
    std::vector<dataType>& output, const char** KernelSource) {
    // Implementation of naive matrix multiplication using OpenCL
    // Use the OpenCL setup functions defined in opencl_setup.c

    // Set up OpenCL environment: context, command queue, and device are assumed to be initialized already.
    cl_platform_id platform;
    cl_device_id device;
    SelectTargetDevice(&platform, &device);
    cl_context context = CreateOpenCLContext(&platform, &device);
    cl_command_queue queue = CreateCommandQueue(context, &device);
    cl_int err; 
    cl_event event = NULL;

    // Prepare data for OpenCL
    size_t bytesA = shape1[0] * shape1[1] * sizeof(dataType);
    size_t bytesB = shape2[0] * shape2[1] * sizeof(dataType);
    size_t bytesC = shape1[0] * shape2[1] * sizeof(dataType);
    output.resize(shape1[0] * shape2[1]); // Ensure output vector is correctly sized

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesA, input1.data(), NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesB, input2.data(), NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytesC, NULL, NULL);

    // Create and build the program
    cl_program program = clCreateProgramWithSource(context, 1, KernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    PrintKernelBuildLog(program, device); // Print any compilation errors of the Kernel source
    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", &err);

    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    err = clSetKernelArg(kernel, 3, sizeof(int), &shape1[0]);
    err = clSetKernelArg(kernel, 4, sizeof(int), &shape1[1]);
    err = clSetKernelArg(kernel, 5, sizeof(int), &shape2[1]);

    // Execute the kernel
    const int TS = 16; // Max work group size of the current device is 512. ### CAUTION: Make sure this is the same in the Kernel source code also (opencl_kernels.h) 
    size_t localSize[2] = { TS, TS };
    size_t globalSize[2] = { shape1[0], shape2[1] };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    //err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, &event);
   
    err = clWaitForEvents(1, &event);
    // Read the result back into the output vector
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytesC, output.data(), 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return;
}


void PrintKernelBuildLog(cl_program program, cl_device_id device) {
    size_t logSize;
    char* log;
    cl_int err;

    // Get the size of the compilation log
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    if (err != CL_SUCCESS) {
        printf("Failed to get program build info size. Error %d\n", err);
        // Handle error
        return;
    }

    // Allocate memory for the compilation log
    log = (char*)malloc(logSize);
    if (!log) {
        printf("Failed to allocate memory for log.\n");
        // Handle error
        return;
    }

    // Retrieve the compilation log
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get program build info. Error %d\n", err);
        // Handle error
        free(log);
        return;
    }

    // Print the compilation log
    if (strlen(log) > 0) {
        printf("Compilation Log:\n%s\n", log);
    }
    else {
        printf("Compilation Success\n");
    }

    // Cleanup
    free(log);
}