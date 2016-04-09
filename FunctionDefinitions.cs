using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Grammophone.Vectors;

namespace Grammophone.Optimization
{
	/// <summary>
	/// Function of vector returning a vector.
	/// </summary>
	/// <param name="w">The vector input argument.</param>
	/// <returns>Returns the vector that corresponds to the input.</returns>
	public delegate Vector VectorFunction(Vector w);

	/// <summary>
	/// Function of vector returning a scalar.
	/// </summary>
	/// <param name="w">The vector input argument.</param>
	/// <returns>Returns the scalar that corresponds to the input.</returns>
	public delegate double ScalarFunction(Vector w);

	/// <summary>
	/// Function of vector returning a linear function of vector.
	/// In other words, a function of vector returning a tensor.
	/// </summary>
	/// <param name="w">The input vector.</param>
	/// <returns>Returns the tensor that corresponds to the input vector.</returns>
	public delegate Vector.Tensor TensorFunction(Vector w);

}
