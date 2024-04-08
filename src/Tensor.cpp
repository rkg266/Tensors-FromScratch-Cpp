#include "Tensor.hpp"
#include <memory> // Include the memory header for std::shared_ptr


/*********TENSOR CLASS************/

// Constructors
Tensor::Tensor(const std::vector<int>& aShape) {
	shape = aShape;
}
Tensor::Tensor(const std::vector<int>& aShape, const std::vector<float>& aData) : shape(aShape), data(aData) { // list initilaization
	if (std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) != data.size()) {
		std::cerr << "Error: Shape and data size do not match." << "\n";
		std::exit(EXIT_FAILURE);
	}
}
Tensor::Tensor(const std::vector<std::vector<float>>& aData) {
	if (aData.empty()) {
		std::cerr << "Error: Cannot initialize Tensor with empty data." << "\n";
		std::exit(EXIT_FAILURE);
	}

	int rows = aData.size();
	int cols = aData[0].size();
	for (const auto& row : aData) {
		if (row.size() != cols) {
			std::cerr << "Error: All rows in the 2D vector must have the same size." << "\n";
			std::exit(EXIT_FAILURE);
		}
	}

	this->shape = { rows, cols };
	this->data.reserve(rows * cols);
	for (auto& row : aData) {
		this->data.insert(this->data.end(), row.begin(), row.end()); // Flatten the 2D vector into 1D and store it in data
	}
}

// Copy constructor and copy assignment operator
Tensor::Tensor(const Tensor& aTensor) : shape(aTensor.shape), data(aTensor.data) {} // list initilaization
Tensor& Tensor::operator=(const Tensor& aTensor) {
	shape = aTensor.shape;
	data = aTensor.data;
	return *this;
}

// Move constructor and move assignment operator
// && indicates that "aTemp" is an r-value reference. 
// Transfers resources from temporary "aTemp" to current object without copying
Tensor::Tensor(Tensor&& aTemp) noexcept : shape(std::move(aTemp.shape)), data(std::move(aTemp.data)) {}
Tensor& Tensor::operator=(Tensor&& aTemp) noexcept {
	if (this != &aTemp) { // Check for self-assignment
		shape = std::move(aTemp.shape);
		data = std::move(aTemp.data);
	}
	return *this;
}

// Indexing: Access and Modify
// single value
float& Tensor::operator()(int i, int j) {
	// for non-const Tensor
	return data[i * shape[1] + j];
}
const float& Tensor::operator()(int i, int j) const {
	// for const Tensor
	return data[i * shape[1] + j];
}
// Methods to get a proxy for row or column access
TensorAccessProxy Tensor::operator()(int index, const All&) {
	return TensorAccessProxy(*this, index, {}, TensorAccessProxy::AccessMode::Row);
}
TensorAccessProxy Tensor::operator()(const All&, int index) {
	return TensorAccessProxy(*this, index, {}, TensorAccessProxy::AccessMode::Column);
}

// Method to access a submatrix based on row and column slices
TensorAccessProxy Tensor::operator()(Slice rowSlice, Slice colSlice) {
	return TensorAccessProxy(*this, 0, std::vector<Slice>{ rowSlice, colSlice }, 
		TensorAccessProxy::AccessMode::Submatrix);
}

// Tensor operations
Tensor Tensor::operator+(const Tensor& aTensor) const {
	ShapeCompatibility curCompatability = CheckShapeCompatibility(aTensor, OperationType::Addition);
	if (ShapeCompatibility::Incompatible == curCompatability) {
		std::cerr << "Error: Operand tensor's shape is incompatible." << "\n";
		std::exit(EXIT_FAILURE);
	}

	std::shared_ptr<OperationInterface> OperationPerformer;
	std::vector<dataType> answer;
	if (UseDevice == Device::cpu) {
		OperationPerformer = std::make_shared<CPUOperation>();
		OperationPerformer->performOperation(this->data, aTensor.data,
			answer, OperationType::Addition, curCompatability);

		return Tensor(this->shape, answer);
	}
	return *this;
}
Tensor Tensor::operator-(const Tensor& aTensor) const {
	ShapeCompatibility curCompatability = CheckShapeCompatibility(aTensor, OperationType::Subtraction);
	if (ShapeCompatibility::Incompatible == curCompatability) {
		std::cerr << "Error: Operand tensor's shape is incompatible." << "\n";
		std::exit(EXIT_FAILURE);
	}

	std::shared_ptr<OperationInterface> OperationPerformer;
	std::vector<dataType> answer;
	if (UseDevice == Device::cpu) {
		OperationPerformer = std::make_shared<CPUOperation>();
		OperationPerformer->performOperation(this->data, aTensor.data,
			answer, OperationType::Subtraction, curCompatability);

		return Tensor(this->shape, answer);
	}
	return *this;
}
Tensor Tensor::operator*(const Tensor& aTensor) const {
	ShapeCompatibility curCompatability = CheckShapeCompatibility(aTensor, OperationType::Multiplication);
	if (ShapeCompatibility::Incompatible == curCompatability) {
		std::cerr << "Error: Operand tensor's shape is incompatible." << "\n";
		std::exit(EXIT_FAILURE);
	}

	std::shared_ptr<OperationInterface> OperationPerformer;
	std::vector<dataType> answer;
	if (UseDevice == Device::cpu) {
		OperationPerformer = std::make_shared<CPUOperation>();
		OperationPerformer->performOperation(this->data, aTensor.data,
			answer, OperationType::Multiplication, curCompatability);

		return Tensor(this->shape, answer);
	}
	return *this;
}
Tensor Tensor::operator/(const Tensor& aTensor) const {
	ShapeCompatibility curCompatability = CheckShapeCompatibility(aTensor, OperationType::Division);
	if (ShapeCompatibility::Incompatible == curCompatability) {
		std::cerr << "Error: Operand tensor's shape is incompatible." << "\n";
		std::exit(EXIT_FAILURE);
	}

	std::shared_ptr<OperationInterface> OperationPerformer;
	std::vector<dataType> answer;
	if (UseDevice == Device::cpu) {
		OperationPerformer = std::make_shared<CPUOperation>();
		OperationPerformer->performOperation(this->data, aTensor.data,
			answer, OperationType::Division, curCompatability);
		
		return Tensor(this->shape, answer);
	}
	return *this;
}

// Operations with TensorProxy
Tensor Tensor::operator+(const TensorAccessProxy& aTensorProxy) const {
	return this->operator+(aTensorProxy.getTensor());
}
Tensor Tensor::operator-(const TensorAccessProxy& aTensorProxy) const {
	return this->operator-(aTensorProxy.getTensor());
}
Tensor Tensor::operator*(const TensorAccessProxy& aTensorProxy) const {
	return this->operator*(aTensorProxy.getTensor());
}
Tensor Tensor::operator/(const TensorAccessProxy& aTensorProxy) const {
	return this->operator/(aTensorProxy.getTensor());
}

// Operations with Scalar
Tensor Tensor::operator+(const dataType& aScalar) const {
	return this->operator+(Tensor({ 1, 1 }, { aScalar }));
}
Tensor Tensor::operator-(const dataType& aScalar) const {
	return this->operator-(Tensor({ 1, 1 }, { aScalar }));
}
Tensor Tensor::operator*(const dataType& aScalar) const {
	return this->operator*(Tensor({ 1, 1 }, { aScalar }));
}
Tensor Tensor::operator/(const dataType& aScalar) const {
	return this->operator/(Tensor({ 1, 1 }, { aScalar }));
}


// Matrix multiplication
Tensor Tensor::matmul(Tensor& aTensor) {
	ShapeCompatibility curCompatability = CheckShapeCompatibility(aTensor, OperationType::MatrixMultiplication);
	if (ShapeCompatibility::Incompatible == curCompatability) {
		std::cerr << "Error: Operand tensor's shape is incompatible." << "\n";
		std::exit(EXIT_FAILURE);
	}

	std::shared_ptr<OperationInterface> OperationPerformer;
	std::vector<dataType> answer;
	std::vector<int> shape1(this->shape);
	std::vector<int> shape2(aTensor.shape);
	std::vector<int> shapeOut{shape1[0], shape2[1]};

	if (UseDevice == Device::cpu) {
		OperationPerformer = std::make_shared<CPUOperation>();
		OperationPerformer->Matrix2DMulitplication(this->data, shape1,
			aTensor.data, shape2, answer);

		return Tensor(shapeOut, answer);
	}
	return *this;
}

// Utility functions
void Tensor::print() const {
	std::cout << "Shape: (";
	for (size_t i = 0; i < shape.size(); ++i) {
		std::cout << shape[i];
		if (i < shape.size() - 1) std::cout << ", ";
	}
	std::cout << ")\nData: \n";

	if (shape.size() == 2) { // For 2D tensors
		for (int i = 0; i < shape[0]; ++i) {
			for (int j = 0; j < shape[1]; ++j) {
				std::cout << data[i * shape[1] + j] << " ";
			}
			std::cout << "\n";
		}
	}
	else if (shape.size() == 1) { // For 1D tensors (vectors)
		for (int i = 0; i < shape[0]; ++i) {
			std::cout << data[i] << " ";
		}
		std::cout << "\n";
	}
	else {
		// Extend this block to handle tensors with more dimensions
		std::cout << "Printing for tensors with more than 2 dimensions is not implemented.\n";
	}
	//std::cout << "UseDevice: " << static_cast<int>(UseDevice) << std::endl;
}

/*************** HELPER METHODS ****************/
// Check shape compatibility for operations
ShapeCompatibility Tensor::CheckShapeCompatibility(const Tensor& aTensor, const OperationType opType) const {
	std::vector<int> aShape = aTensor.shape;

	// for matrix multiplication
	if (OperationType::MatrixMultiplication == opType) {
		if (shape[1] == aShape[0]) return ShapeCompatibility::ColsRowsMatch;
		return ShapeCompatibility::Incompatible;
	}
	else {
	// for all other operations
		if (shape == aShape) return ShapeCompatibility::ShapeMatch;
		if (aShape[0] == 1 && aShape[1] == 1) return ShapeCompatibility::IsScalar;
		if (shape[0] == aShape[0] && aShape[1] == 1) return ShapeCompatibility::ColVector;
		if (shape[1] == aShape[1] && aShape[0] == 1) return ShapeCompatibility::RowVector;
		return ShapeCompatibility::Incompatible;
	}
	
}


/******************************************************** TENSOR ACCESS PROXY *********************************************************/

// constructor
TensorAccessProxy::TensorAccessProxy(Tensor& tensor, int index, std::vector<Slice> slice, AccessMode mode)
	: tensor(tensor), index(index), slice(slice), mode(mode) {}

// Assignment operator for both row and column
TensorAccessProxy& TensorAccessProxy::operator=(const Tensor& src) {
	if (mode == AccessMode::Row) {
		if (src.shape[1] != tensor.shape[1] || src.shape[0] != 1) {
			std::cerr << "Error: Source tensor dimensions do not match target row." << "\n";
			std::exit(EXIT_FAILURE);
		}
		for (int col = 0; col < tensor.shape[1]; ++col) {
			tensor.data[index * tensor.shape[1] + col] = src.data[col];
		}
	}
	else if (mode == AccessMode::Column) {
		if (src.shape[0] != tensor.shape[0] || src.shape[1] != 1) {
			std::cerr << "Error: Source tensor dimensions do not match target column." << "\n";
			std::exit(EXIT_FAILURE);
		}
		for (int row = 0; row < tensor.shape[0]; ++row) {
			tensor.data[row * tensor.shape[1] + index] = src.data[row];
		}
	}
	else if (mode == AccessMode::Submatrix) {
		std::vector<int> sliceShape{ slice[0].end - slice[0].start, slice[1].end - slice[1].start };
		if (src.shape != sliceShape) {
			std::cerr << "Error: Source tensor dimensions do not match target submatrix." << "\n";
			std::exit(EXIT_FAILURE);
		}
		// assign the submatrix
		int srcRowIndex = 0;
		for (int rowIndex = slice[0].start; rowIndex < slice[0].end; ++rowIndex, ++srcRowIndex) {
			int srcColIndex = 0;
			for (int colIndex = slice[1].start; colIndex < slice[1].end; ++colIndex, ++srcColIndex) {
				// Calculate the flat index for the target tensor
				int targetIndex = rowIndex * tensor.shape[1] + colIndex;
				// Calculate the flat index for the source tensor 
				int sourceIndex = srcRowIndex * src.shape[1] + srcColIndex;
				tensor.data[targetIndex] = src.data[sourceIndex];
			}
		}
	}
	return *this;
}

// Conversion operator to support extraction as a Tensor
TensorAccessProxy::operator Tensor() const {
	std::vector<float> extractedData;
	if (mode == AccessMode::Row) {
		extractedData.reserve(tensor.shape[1]);
		for (int col = 0; col < tensor.shape[1]; ++col) {
			extractedData.push_back(tensor.data[index * tensor.shape[1] + col]);
		}
		return Tensor({ 1, tensor.shape[1] }, extractedData);
	}
	else if (mode == AccessMode::Column) {
		extractedData.reserve(tensor.shape[0]);
		for (int row = 0; row < tensor.shape[0]; ++row) {
			extractedData.push_back(tensor.data[row * tensor.shape[1] + index]);
		}
		return Tensor({ tensor.shape[0], 1 }, extractedData);
	}
	else { // Submatrix
		std::vector<int> subShape = { slice[0].end - slice[0].start, slice[1].end - slice[1].start };
		extractedData.reserve(subShape[0] * subShape[1]);
		for (int i = slice[0].start; i < slice[0].end; ++i) {
			for (int j = slice[1].start; j < slice[1].end; ++j) {
				extractedData.push_back(tensor.data[i * tensor.shape[1] + j]);
			}
		}
		return Tensor(subShape, extractedData);
	}
}


// Extract the sliced-Tensor
Tensor& TensorAccessProxy::getTensor() const {
	return this->operator Tensor(); // This is how we call custom operator
}

// Operations with TensorProxy
Tensor TensorAccessProxy::operator+(const TensorAccessProxy& aTensorProxy) const {
	return (this->getTensor() + aTensorProxy.getTensor());
}
Tensor TensorAccessProxy::operator-(const TensorAccessProxy& aTensorProxy) const {
	return (this->getTensor() - aTensorProxy.getTensor());
}
Tensor TensorAccessProxy::operator*(const TensorAccessProxy& aTensorProxy) const {
	return (this->getTensor() * aTensorProxy.getTensor());
}
Tensor TensorAccessProxy::operator/(const TensorAccessProxy& aTensorProxy) const {
	return (this->getTensor() / aTensorProxy.getTensor());
}

// Operations with Tensor
Tensor TensorAccessProxy::operator+(const Tensor& aTensor) const {
	return (this->getTensor() + aTensor);
}
Tensor TensorAccessProxy::operator-(const Tensor& aTensor) const {
	return (this->getTensor() - aTensor);
}
Tensor TensorAccessProxy::operator*(const Tensor& aTensor) const {
	return (this->getTensor() * aTensor);
}
Tensor TensorAccessProxy::operator/(const Tensor& aTensor) const {
	return (this->getTensor() / aTensor);
}

// Scalar operations
Tensor TensorAccessProxy::operator+(const dataType& aScalar) const {
	return (this->getTensor() + Tensor({1, 1}, {aScalar}));
}
Tensor TensorAccessProxy::operator-(const dataType& aScalar) const {
	return (this->getTensor() - Tensor({ 1, 1 }, { aScalar }));
}
Tensor TensorAccessProxy::operator*(const dataType& aScalar) const {
	return (this->getTensor() * Tensor({ 1, 1 }, { aScalar }));
}
Tensor TensorAccessProxy::operator/(const dataType& aScalar) const {
	return (this->getTensor() / Tensor({ 1, 1 }, { aScalar }));
}

// Utility functions
void TensorAccessProxy::print() const {
	this->operator Tensor().print();
}


/******************************************************** NON-MEMBER FUNCTIONS *********************************************************/
// Operations with Scalar on the left
Tensor operator+(const dataType& aScalar, const Tensor& aTensor) {
	return (Tensor({ 1, 1 }, { aScalar }) + aTensor);
}
Tensor operator-(const dataType& aScalar, const Tensor& aTensor) {
	return (Tensor({ 1, 1 }, { aScalar }) - aTensor);
}
Tensor operator*(const dataType& aScalar, const Tensor& aTensor) {
	return (Tensor({ 1, 1 }, { aScalar }) * aTensor);
}
Tensor operator/(const dataType& aScalar, const Tensor& aTensor) {
	return (Tensor({ 1, 1 }, { aScalar }) / aTensor);
}

Tensor operator+(const dataType& aScalar, const TensorAccessProxy& aTensorProxy) {
	return (Tensor({ 1, 1 }, { aScalar }) + aTensorProxy.getTensor());
}
Tensor operator-(const dataType& aScalar, const TensorAccessProxy& aTensorProxy) {
	return (Tensor({ 1, 1 }, { aScalar }) - aTensorProxy.getTensor());
}
Tensor operator*(const dataType& aScalar, const TensorAccessProxy& aTensorProxy) {
	return (Tensor({ 1, 1 }, { aScalar }) * aTensorProxy.getTensor());
}
Tensor operator/(const dataType& aScalar, const TensorAccessProxy& aTensorProxy) {
	return (Tensor({ 1, 1 }, { aScalar }) / aTensorProxy.getTensor());
}