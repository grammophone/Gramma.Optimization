using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Gramma.Vectors;

namespace Gramma.Optimization.QuasiNewton
{
	/// <summary>
	/// Optimizer using the Conjugate Gradient method.
	/// </summary>
	[Serializable]
	public class ConjugateGradientOptimizer : Optimizer
	{
		#region Private fields

		private ConjugateGradient.LineSearchMinimizeOptions options;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		public ConjugateGradientOptimizer()
		{
			options = new ConjugateGradient.LineSearchMinimizeOptions();
		}

		#endregion

		#region Public properties

		/// <summary>
		/// The options used for the Conjugate Gradient solver.
		/// </summary>
		public ConjugateGradient.LineSearchMinimizeOptions Options
		{
			get
			{
				return this.options;
			}
			set
			{
				if (value == null) throw new ArgumentNullException("value");

				this.options = value;
			}
		}

		#endregion

		#region Public methods

		public override Vector Minimize(
			ScalarFunction f, 
			VectorFunction df, 
			Vector w,
			TensorFunction M = null,
			Func<Vector, bool> outOfDomainIndicator = null)
		{
			return ConjugateGradient.LineSearchMinimize(f, df, w, this.options, M, outOfDomainIndicator);
		}

		#endregion
	}
}
