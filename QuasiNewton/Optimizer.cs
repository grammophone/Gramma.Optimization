using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Gramma.Vectors;

namespace Gramma.Optimization.QuasiNewton
{
	/// <summary>
	/// Abstract base for Quasi-Newton optimizers.
	/// </summary>
	[Serializable]
	public abstract class Optimizer
	{
		#region Public methods

		/// <summary>
		/// Minimize a convex scalar function.
		/// </summary>
		/// <param name="f">The scalar function to minimize.</param>
		/// <param name="df">The gradient of the scalar function.</param>
		/// <param name="w">Initial estimation of solution.</param>
		/// <param name="M">Optional preconditioner at point w. Default is identity.</param>
		/// <param name="outOfDomainIndicator">
		/// Optional fast function to indicate true when w is out of domain instead of 
		/// fully evaluating f(w) and testing for infinity or NaN.
		/// </param>
		/// <returns>
		/// Returns the optimum point where the function <paramref name="f"/>
		/// attains its minimum.
		/// </returns>
		public abstract Vector Minimize(
			ScalarFunction f,
			VectorFunction df,
			Vector w,
			TensorFunction M = null,
			Func<Vector, bool> outOfDomainIndicator = null);

		#endregion
	}
}
