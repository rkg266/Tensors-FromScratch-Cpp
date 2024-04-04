#include "opencl_setup.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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