#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "Operations.hpp"

struct All {};

struct Slice {
    int start;
    int end;
    Slice(int s, int e) : start(s), end(e) {}
};

class TensorAccessProxy;  // Forward declaration

class Tensor {
public:
    static inline const All all{};

    // Constructors
    Tensor(); // Default constructor
    explicit Tensor(const std::vector<int>& aShape); // Construct tensor with a given shape
    Tensor(const std::vector<int>& aShape, const std::vector<float>& aData); // Construct tensor with shape and data
    Tensor(const std::vector<std::vector<float>>& aData); // Construct tensor with 2D vector


    // Destructor
    ~Tensor() = default;

    // Copy constructor and copy assignment operator
    Tensor(const Tensor& aTensor);
    Tensor& operator=(const Tensor& aTensor);

    // Move constructor and move assignment operator
    Tensor(Tensor&& aTemp) noexcept;
    Tensor& operator=(Tensor&& aTemp) noexcept;

    // Indexing: Access/Assign
    // single value
    float& operator()(int i, int j); // for non-const Tensor
    const float& operator()(int i, int j) const; // for const Tensor

    // Methods to get a proxy for row or column access
    TensorAccessProxy operator()(int index, const All&);
    TensorAccessProxy operator()(const All&, int index);

    // Method to access a submatrix based on row and column slices
    TensorAccessProxy operator()(Slice rowSlice, Slice colSlice);

    // Tensor operations
    Tensor operator+(const Tensor& aTensor) const; // Addition
    Tensor operator-(const Tensor& aTensor) const; // Subtraction
    Tensor operator*(const Tensor& aTensor) const; // Multiplication
    Tensor operator/(const Tensor& aTensor) const; // Division

    // Operations with TensorProxy
    Tensor operator+(const TensorAccessProxy& aTensorProxy) const; // Addition
    Tensor operator-(const TensorAccessProxy& aTensorProxy) const; // Subtraction
    Tensor operator*(const TensorAccessProxy& aTensorProxy) const; // Multiplication
    Tensor operator/(const TensorAccessProxy& aTensorProxy) const; // Division

    // Matrix multiplication
    Tensor matmul(Tensor& aTensor); // const REMOVED for OpenCL. Figure out later

    // Scalar operations
    Tensor operator+(const dataType& aScalar) const; // Addition with scalar
    Tensor operator-(const dataType& aScalar) const; // Subtraction with scalar
    Tensor operator*(const dataType& aScalar) const; // Multiplication with scalar
    Tensor operator/(const dataType& aScalar) const; // Division with scalar

    // Utility functions
    void print() const; // For debugging: print tensor values
    std::vector<int> getShape() const; // Get the shape of the tensor
    int numel() const; // Number of elements in the tensor

    // friends 
    friend class TensorAccessProxy;
    friend class ParallelOperation; // from "Operations.hpp"
    
private:
    std::vector<int> shape; // Shape of the tensor
    std::vector<float> data; // Data of the tensor, stored in a flat array

    // Private methods for internal use
    ShapeCompatibility CheckShapeCompatibility(const Tensor& aTensor, const OperationType opType) const; // Check shape compatibility for operations
};

class TensorAccessProxy {
public:
    enum class AccessMode { Row, Column, Submatrix};

    // constructor
    TensorAccessProxy(Tensor& tensor, int index, std::vector<Slice> slice, AccessMode mode);

    TensorAccessProxy& operator=(const Tensor& src); // Assignment operator for both row and column
    operator Tensor() const; // Conversion operator to support extraction as a Tensor. Usage: "aTensorProxy.operator Tensor()".
                                //Works for "Tensor aTensor = aTensorProxy". 

    // Extract the Tensor
    Tensor& getTensor() const;

    // Move, copy operators?

    
    // Operations with TensorProxy
    Tensor operator+(const TensorAccessProxy& aTensorProxy) const; // Addition
    Tensor operator-(const TensorAccessProxy& aTensorProxy) const; // Subtraction
    Tensor operator*(const TensorAccessProxy& aTensorProxy) const; // Multiplication
    Tensor operator/(const TensorAccessProxy& aTensorProxy) const; // Division

    // Operations with Tensor
    Tensor operator+(const Tensor& aTensor) const; // Addition
    Tensor operator-(const Tensor& aTensor) const; // Subtraction
    Tensor operator*(const Tensor& aTensor) const; // Multiplication
    Tensor operator/(const Tensor& aTensor) const; // Division

    // Operations with Scalar
    Tensor operator+(const dataType& aScalar) const; // Addition
    Tensor operator-(const dataType& aScalar) const; // Subtraction
    Tensor operator*(const dataType& aScalar) const; // Multiplication
    Tensor operator/(const dataType& aScalar) const; // Division

    // Utility functions
    void print() const; // For debugging: print tensor values
    std::vector<int> getShape() const; // Get the shape of the tensor
    int numel() const; // Number of elements in the tensor

private:
    Tensor& tensor;
    int index; // Row or column index
    std::vector<Slice> slice;
    AccessMode mode;
};


/******************************************************** NON-MEMBER FUNCTIONS *********************************************************/
// Operations with Scalar on the left
// aScalar + aTensor
Tensor operator+(const dataType& aScalar, const Tensor& aTensor);
Tensor operator-(const dataType& aScalar, const Tensor& aTensor);
Tensor operator*(const dataType& aScalar, const Tensor& aTensor);
Tensor operator/(const dataType& aScalar, const Tensor& aTensor);

// aScalar + aTensorProxy
Tensor operator+(const dataType& aScalar, const TensorAccessProxy& aTensorProxy);
Tensor operator-(const dataType& aScalar, const TensorAccessProxy& aTensorProxy);
Tensor operator*(const dataType& aScalar, const TensorAccessProxy& aTensorProxy);
Tensor operator/(const dataType& aScalar, const TensorAccessProxy& aTensorProxy);




#endif // TENSOR_HPP