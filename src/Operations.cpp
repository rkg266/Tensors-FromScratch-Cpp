#include "Operations.hpp"
#include "opencl_setup.h" // OpenCL seup and execution
#include "opencl_kernels.h" // Kernel implementations
#include <cmath> // cmath header for std::isnan
#include <omp.h> // OpenMP for CPU parallel programming

/*********** CPUOperation *************/

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

    //MatrixMultiplyKernelBased(input1, shape1, input2, shape2, output, &matrixMultNaiveKernelSource);

    MatrixMultiplyKernelBased(input1, shape1, input2, shape2, output, &matrixMultTilingKernelSource);
    return;
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
