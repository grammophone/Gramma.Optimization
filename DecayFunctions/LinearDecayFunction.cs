using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Gramma.Optimization.DecayFunctions
{
	/// <summary>
	/// Implements the 1/n decay function.
	/// </summary>
	[Serializable]
	public class LinearDecayFunction : DecayFunction
	{
		/// <summary>
		/// Returns 1 / <paramref name="iterationNumber"/>.
		/// </summary>
		public override double Evaluate(int iterationNumber)
		{
			return 1.0 / iterationNumber;
		}
	}
}
