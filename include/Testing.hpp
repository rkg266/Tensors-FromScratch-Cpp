#ifndef Testing_h
#define Testing_h

#include "Tensor.hpp"
#include <random>

class Testing {
public:
    void TestIndexing() {
        // Create a 5x5 tensor initialized with some data for demonstration
        Tensor myTensor({ 5, 5 }, {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25
            });

        std::cout << "Original Tensor:" << std::endl;
        myTensor.print();

        // Access and modify a specific element in the 3rd row and 4th column (0-indexed)
        myTensor(2, 3) = 100; // Change value at row=2, col=3 to 100
        std::cout << "After modifying an element:" << std::endl;
        myTensor.print();

        // Extract and print the 2nd row as a new Tensor
        Tensor rowTensor = myTensor(1, Tensor::all);
        std::cout << "Extracted 2nd row:" << std::endl;
        rowTensor.print();

        // Assign a new row to the 1st row of myTensor
        Tensor newRow({ 1, 5 }, { 101, 102, 103, 104, 105 });
        myTensor(0, Tensor::all) = newRow;
        std::cout << "After assigning a new row:" << std::endl;
        myTensor.print();

        // Extract and print the 3rd column as a new Tensor
        Tensor colTensor = myTensor(Tensor::all, 2);
        std::cout << "Extracted 3rd column:" << std::endl;
        colTensor.print();

        // Assign a new column to the 2nd column of myTensor
        Tensor newCol({ 5, 1 }, { 106, 107, 108, 109, 110 });
        myTensor(Tensor::all, 1) = newCol;
        std::cout << "After assigning a new column:" << std::endl;
        myTensor.print();

        // Extract a 3x3 submatrix starting from (1,1) and assign a new submatrix
        Tensor subMatrix = myTensor(Slice(1, 4), Slice(1, 4));
        std::cout << "Extracted submatrix:" << std::endl;
        subMatrix.print();

        // Assign a new 3x3 submatrix to the extracted position
        Tensor newSubMatrix({ 3, 3 }, { 201, 202, 203, 204, 205, 206, 207, 208, 209 });
        myTensor(Slice(1, 4), Slice(1, 4)) = newSubMatrix;
        std::cout << "After assigning a new submatrix:" << std::endl;
        myTensor.print();
    }

    void TestElementWiseOperations() {  // NEEDS MORE TEST EXAMPLES
        // Create tensors for testing
        Tensor tensor1({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        //Tensor tensor1({ 4, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor tensor2({ 2, 3 }, { 2, 4, 5, 6, 1, 3 });
        // Tensor tensor2({ {2, 4, 6}, {8, 10, 12} }); -> this case not handled in the Tensor constructor
        Tensor scalarTensor({ 1, 1 }, { 5 }); // Scalar tensor
        std::cout << "First Tensor:" << std::endl;
        tensor1.print();
        std::cout << "Second Tensor:" << std::endl;
        tensor2.print();

        // Test addition between tensor and scalar
        std::cout << "Addition between tensor and scalar:\n";
        Tensor resultAdditionScalar = tensor1 + scalarTensor;
        resultAdditionScalar.print();

        // Test subtraction between tensor and scalar
        std::cout << "Subtraction between tensor and scalar:\n";
        Tensor resultSubtractionScalar = tensor1 - scalarTensor;
        resultSubtractionScalar.print();

        // Test multiplication between tensor and scalar
        std::cout << "Multiplication between tensor and scalar:\n";
        Tensor resultMultiplicationScalar = tensor1 * scalarTensor;
        resultMultiplicationScalar.print();

        // Test division between tensor and scalar
        std::cout << "Division between tensor and scalar:\n";
        Tensor resultDivisionScalar = tensor1 / scalarTensor;
        resultDivisionScalar.print();

        // Test addition between tensor matrix and row vector tensor
        std::cout << "Addition between tensor matrix and row vector tensor:\n";
        Tensor rowVector({1, 3}, {1, 2, 3});
        Tensor resultAdditionRowVector = tensor1 + rowVector;
        resultAdditionRowVector.print();

        // Test addition between tensor matrix and column vector tensor
        std::cout << "Addition between tensor matrix and column vector tensor:\n";
        Tensor colVector({ 2, 1 }, { 1, 4 });
        Tensor resultAdditionColVector = tensor1 + colVector;
        resultAdditionColVector.print();

        // Test addition between tensor matrices
        std::cout << "Addition between tensor matrices:\n";
        Tensor resultAdditionMatrices = tensor1 + tensor2;
        resultAdditionMatrices.print();
    }

    void TestMatrixMultiplication() {
        Tensor tensor1({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        Tensor tensor2({ 3, 2 }, { 2, 4, 5, 6, 1, 3 });
        std::cout << "First Tensor:" << std::endl;
        tensor1.print();
        std::cout << "Second Tensor:" << std::endl;
        tensor2.print();

        // Test matrix multiplication
        std::cout << "Matrix multiplication:\n";
        Tensor resultTensor = tensor1.matmul(tensor2);
        resultTensor.print();

        // Large tensors
        std::vector<int> shape1{ 512, 512 };
        std::vector<int> shape2{ 512, 256 };
        std::vector<dataType> largeVect1 = generateRandomVector<dataType>(shape1[0] * shape1[1], -25, 25);
        std::vector<dataType> largeVect2 = generateRandomVector<dataType>(shape2[0] * shape2[1], -25, 25);
        Tensor largeTensor1({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        Tensor largeTensor2({ 3, 2 }, { 2, 4, 5, 6, 1, 3 });
        std::cout << "Large Matrix multiplication: Started\n";
        Tensor resultTensor1 = largeTensor1.matmul(largeTensor2);
        std::cout << "Large Matrix multiplication: Done\n";
    }

 private:
     // Random vector generation
     template<typename T>
     std::vector<T> generateRandomVector(int size, T minVal, T maxVal) {
         std::random_device rd;
         std::mt19937 gen(rd());
         std::uniform_real_distribution<T> dis(minVal, maxVal);

         std::vector<T> result(size);
         for (int i = 0; i < size; ++i) {
             result[i] = dis(gen);
         }
         return result;
     }
};


#endif
