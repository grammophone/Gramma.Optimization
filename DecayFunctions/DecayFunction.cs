using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Grammophone.Optimization.DecayFunctions
{
	/// <summary>
	/// Used by <see cref="StochasticGradientDescent"/> methods.
	/// </summary>
	[Serializable]
	public abstract class DecayFunction
	{
		/// <summary>
		/// Return the decay corresponding to an iteration number.
		/// </summary>
		/// <param name="iterationNumber">The iteration number, expected to be at least 1.</param>
		/// <returns>Returns the decay.</returns>
		public abstract double Evaluate(int iterationNumber);
	}
}
