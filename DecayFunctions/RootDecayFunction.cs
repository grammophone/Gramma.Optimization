using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Grammophone.Optimization.DecayFunctions
{
	/// <summary>
	/// Implements the 1/sqrt(n) decay function.
	/// </summary>
	[Serializable]
	public class RootDecayFunction : DecayFunction
	{
		/// <summary>
		/// Returns 1/sqrt(<paramref name="iterationNumber"/>).
		/// </summary>
		public override double Evaluate(int iterationNumber)
		{
			return Math.Sqrt(iterationNumber);
		}
	}
}
