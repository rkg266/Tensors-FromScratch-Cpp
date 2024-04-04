#ifndef OPS_HPP
#define OPS_HPP

#include <iostream>
#include <vector>
#include <numeric>
#include "Globals.hpp"

enum class OperationType {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    MatrixMultiplication
};

enum class ShapeCompatibility {
    ShapeMatch,
    RowVector,
    ColVector,
    ColsRowsMatch,
    IsScalar,
    Incompatible
};

// Abstract interface for parallel operations
class OperationInterface {
public:
    virtual ~OperationInterface() {}
    virtual void performOperation(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
        std::vector<dataType>& output, OperationType opType, 
        ShapeCompatibility spCompat) const = 0;
    virtual void Matrix2DMulitplication(std::vector<dataType>& input1, std::vector<int>& shape1,
        std::vector<dataType>& input2, std::vector<int>& shape2,
        std::vector<dataType>& output) = 0;
};

// CPU parallel operations
class CPUOperation : public OperationInterface {
public:
    virtual void performOperation(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
        std::vector<dataType>& output, OperationType opType,
        ShapeCompatibility spCompat) const override;

    virtual void Matrix2DMulitplication(std::vector<dataType>& input1, std::vector<int>& shape1,
        std::vector<dataType>& input2, std::vector<int>& shape2,
        std::vector<dataType>& output) override;

private:
    void OperationWithScalar(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
                            std::vector<dataType>& output, OperationType opType) const;
    void OperationWithColVector(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
                            std::vector<dataType>& output, OperationType opType) const;
    void OperationWithRowVector(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
                            std::vector<dataType>& output, OperationType opType) const;
    void OperationWithSameShape(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
                            std::vector<dataType>& output, OperationType opType) const;
};

// CUDA parallel operations
class CUDAOperation : public OperationInterface {
public:
    virtual void performOperation(const std::vector<dataType>& input1, const std::vector<dataType>& input2,
        std::vector<dataType>& output, OperationType opType,
        ShapeCompatibility spCompat) const override;
};

#endif
