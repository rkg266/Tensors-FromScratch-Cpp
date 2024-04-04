#include "Operations.hpp"
#include "opencl_setup.h"
#include <cmath> // cmath header for std::isnan
#include <omp.h> // OpenMP for CPU parallel programming

/*********** CPUOperation *************/

// Assuming the kernel source is defined somewhere
// For the sake of simplicity, let's define it here as a global string.
// The kernel for matrix multiplication iterates through the elements in row-major order.
const char* matrixMultiplicationKernelSource = R"CLC(
    __kernel void matrix_multiply(const __global float* A, const __global float* B, __global float* C, 
                                  const int M, const int N, const int K) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if(row < M && col < K) {
            float sum = 0.0;
            for(int i = 0; i < N; ++i) {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
)CLC";

void CPUOperation::performOperation(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
                                    std::vector<dataType>& output, OperationType opType,
                                    ShapeCompatibility spCompat) const {
	if (ShapeCompatibility::IsScalar == spCompat)
		return OperationWithScalar(input1, input2, output, opType);
	if (ShapeCompatibility::ColVector == spCompat)
		return OperationWithColVector(input1, input2, output, opType);
	if (ShapeCompatibility::RowVector == spCompat)
		return OperationWithRowVector(input1, input2, output, opType);
	return OperationWithSameShape(input1, input2, output, opType);
}

void CPUOperation::Matrix2DMulitplication(std::vector<dataType>& input1, std::vector<int>& shape1,
    std::vector<dataType>& input2, std::vector<int>& shape2,
    std::vector<dataType>& output) {

    // Set up OpenCL environment: context, command queue, and device are assumed to be initialized already.
    cl_platform_id platform;
    cl_device_id device;
    SelectTargetDevice(&platform, &device); 
    cl_context context = CreateOpenCLContext(&platform, &device);
    cl_command_queue queue = CreateCommandQueue(context, &device);

    // Prepare data for OpenCL
    size_t bytesA = shape1[0] * shape1[1] * sizeof(dataType);
    size_t bytesB = shape2[0] * shape2[1] * sizeof(dataType);
    size_t bytesC = shape1[0] * shape2[1] * sizeof(dataType);
    output.resize(shape1[0] * shape2[1]); // Ensure output vector is correctly sized

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesA, input1.data(), NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesB, input2.data(), NULL);
     cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytesC, NULL, NULL);

    // Create and build the program
    cl_program program = clCreateProgramWithSource(context, 1, &matrixMultiplicationKernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &shape1[0]);
    clSetKernelArg(kernel, 4, sizeof(int), &shape1[1]);
    clSetKernelArg(kernel, 5, sizeof(int), &shape2[1]);

    // Execute the kernel
    size_t globalSize[2] = { shape1[0], shape2[1] };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);

    // Read the result back into the output vector
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytesC, output.data(), 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

/*************CPUOperation private *********************/
void CPUOperation::OperationWithScalar(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
									std::vector<dataType>& output, OperationType opType) const {
    // Check if input2 has exactly one element
    if (input2.size() != 1) {
        throw std::invalid_argument("Input2 must contain exactly one element for scalar operation.");
    }

    // Get the scalar value from input2
    dataType scalar = input2[0];

    // Get the number of elements in the input vector
    size_t numElements = input1.size();

    // Check if the scalar value is NaN
    if (std::isnan(scalar)) {
        // If the scalar value is NaN, set all output elements to NaN
        output.assign(input1.size(), std::numeric_limits<dataType>::quiet_NaN());
        return;
    }

    // Resize the output vector to match the input size
    output.resize(numElements);

    // Parallelize the operation using OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < numElements; ++i) {
        // Check if the input1 value is NaN
        if (std::isnan(input1[i])) {
            // If input1 is NaN, set the output to NaN
            output[i] = std::numeric_limits<dataType>::quiet_NaN();
        }
        else {
            // Perform the operation element-wise
            switch (opType) {
            case OperationType::Addition:
                output[i] = input1[i] + scalar;
                break;
            case OperationType::Subtraction:
                output[i] = input1[i] - scalar;
                break;
            case OperationType::Multiplication:
                output[i] = input1[i] * scalar;
                break;
            case OperationType::Division:
                // Check for division by zero
                if (scalar == 0) {
                    // Handle division by zero gracefully
                    output[i] = std::numeric_limits<dataType>::quiet_NaN();
                }
                else {
                    output[i] = input1[i] / scalar;
                }
                break;
            default:
                // Handle unsupported operation
                output[i] = std::numeric_limits<dataType>::quiet_NaN();
                break;
            }
        }
    }
}

void CPUOperation::OperationWithColVector(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
    std::vector<dataType>& output, OperationType opType) const {

    // Get the number of rows and columns in the matrix
    int numRows = static_cast<int>(input2.size());
    int numCols = static_cast<int>(input1.size() / numRows);
    
    // Resize the output vector to match the size of the matrix
    output.resize(input1.size());

    // Parallelize the operation using OpenMP
    #pragma omp parallel for
    for (int j = 0; j < numCols; ++j) {        
        // Perform the operation element-wise for each column
        for (int i = 0; i < numRows; ++i) {

            int rowStartIdx = i * numCols;
            // Calculate the index of the current element in the output matrix
            int outputIdx = rowStartIdx + j;

            if (std::isnan(input1[outputIdx]) || std::isnan(input2[i])) {
                // If input1 or input2 is NaN, set the output to NaN
                output[outputIdx] = std::numeric_limits<dataType>::quiet_NaN();
            }
            else {
                switch (opType) {
                case OperationType::Addition:
                    output[outputIdx] = input1[outputIdx] + input2[i];
                    break;
                case OperationType::Subtraction:
                    output[outputIdx] = input1[outputIdx] - input2[i];
                    break;
                case OperationType::Multiplication:
                    output[outputIdx] = input1[outputIdx] * input2[i];
                    break;
                case OperationType::Division:
                    // Check for division by zero
                    if (input2[i] == 0) {
                        // Handle division by zero gracefully
                        output[outputIdx] = std::numeric_limits<dataType>::quiet_NaN();
                    }
                    else {
                        output[outputIdx] = input1[outputIdx] / input2[i];
                    }
                    break;
                default:
                    // Handle unsupported operation
                    output[outputIdx] = std::numeric_limits<dataType>::quiet_NaN();
                    break;
                }
            }
        }
    }
}

void CPUOperation::OperationWithRowVector(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
    std::vector<dataType>& output, OperationType opType) const {

    // Get the number of rows and columns in the matrix
    int numCols = static_cast<int>(input2.size());
    int numRows = static_cast<int>(input1.size() / numCols);

    // Resize the output vector to match the size of the matrix
    output.resize(input1.size());

    // Parallelize the operation using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < numRows; ++i) {
        int rowStartIdx = i * numCols;
        // Perform the operation element-wise for each column
        for (int j = 0; j < numCols; ++j) {
            // Calculate the index of the current element in the output matrix
            int outputIdx = rowStartIdx + j;

            if (std::isnan(input1[outputIdx]) || std::isnan(input2[j])) {
                // If input1 or input2 is NaN, set the output to NaN
                output[outputIdx] = std::numeric_limits<dataType>::quiet_NaN();
            }
            else {
                switch (opType) {
                case OperationType::Addition:
                    output[outputIdx] = input1[outputIdx] + input2[j];
                    break;
                case OperationType::Subtraction:
                    output[outputIdx] = input1[outputIdx] - input2[j];
                    break;
                case OperationType::Multiplication:
                    output[outputIdx] = input1[outputIdx] * input2[j];
                    break;
                case OperationType::Division:
                    // Check for division by zero
                    if (input2[j] == 0) {
                        // Handle division by zero gracefully
                        output[outputIdx] = std::numeric_limits<dataType>::quiet_NaN();
                    }
                    else {
                        output[outputIdx] = input1[outputIdx] / input2[j];
                    }
                    break;
                default:
                    // Handle unsupported operation
                    output[outputIdx] = std::numeric_limits<dataType>::quiet_NaN();
                    break;
                }
            }
        }
    }
}

void CPUOperation::OperationWithSameShape(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
    std::vector<dataType>& output, OperationType opType) const {

    // Resize the output vector to match the size of the matrix
    output.resize(input1.size());

    // Parallelize the operation using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < input1.size(); ++i) {
        if (std::isnan(input1[i]) || std::isnan(input2[i])) {
            // If input1 or input2 is NaN, set the output to NaN
            output[i] = std::numeric_limits<dataType>::quiet_NaN();
        }
        else {
            switch (opType) {
            case OperationType::Addition:
                output[i] = input1[i] + input2[i];
                break;
            case OperationType::Subtraction:
                output[i] = input1[i] - input2[i];
                break;
            case OperationType::Multiplication:
                output[i] = input1[i] * input2[i];
                break;
            case OperationType::Division:
                // Check for division by zero
                if (input2[i] == 0) {
                    // Handle division by zero gracefully
                    output[i] = std::numeric_limits<dataType>::quiet_NaN();
                }
                else {
                    output[i] = input1[i] / input2[i];
                }
                break;
            default:
                // Handle unsupported operation
                output[i] = std::numeric_limits<dataType>::quiet_NaN();
                break;
            }
        }
        
    }
}
