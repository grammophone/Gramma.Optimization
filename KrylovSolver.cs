using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Grammophone.Vectors;

namespace Grammophone.Optimization
{
	/// <summary>
	/// Contains method for solution of linear systems using Krylov sequence iterations.
	/// </summary>
	public static class KrylovSolver
	{
		#region Delegates definitions

		/// <summary>
		/// Stop criterion for iterations of <see cref="LinearSolve"/> method. 
		/// </summary>
		/// <param name="w">Current solution approximation.</param>
		/// <param name="w1">Next solution approximation.</param>
		/// <param name="H">The matrix tensor of the system.</param>
		/// <param name="Mg1">
		/// The preconditioned next residual M(Hw1 + b). 
		/// Corresponds to preconditioned gradient of Newton method.
		/// </param>
		/// <returns>Must return true to signal termination of algorithm.</returns>
		public delegate bool LinearSolveStopCriterion(Vector w, Vector w1, Vector.Tensor H, Vector Mg1);

		#endregion

		#region Auxilliary classes

		/// <summary>
		/// Options for method <see cref="LinearSolve"/>.
		/// </summary>
		public class LinearSolveOptions
		{
			#region Standard stop criteria

			/// <summary>
			/// Stop criterion based on squared euclidean norm of difference of
			/// solutions approximations.
			/// </summary>
			/// <param name="ε">The threshold of normalized squared euclidean norm.</param>
			/// <returns>Returns the criterion function.</returns>
			/// <remarks>
			/// Euclidean norm is normalized by division by the dimension.
			/// </remarks>
			public static LinearSolveStopCriterion GetProgressCriterion(double ε)
			{
				return (w, w1, H, Mg1) => ((w - w1).Norm2 / w.Length < ε);
			}

			/// <summary>
			/// Stop criterion based on squared euclidean norm of the
			/// preconditioned residual Mg1 = M(Hw1 + b).
			/// </summary>
			/// <param name="ε">The threshold of normalized squared euclidean norm.</param>
			/// <returns>Returns the criterion function.</returns>
			/// <remarks>
			/// Euclidean norm is normalized by division by the dimension.
			/// </remarks>
			public static LinearSolveStopCriterion GetResidualStopCriterion(double ε)
			{
				return (w, w1, H, Mg1) => (Mg1.Norm2 / Mg1.Length < ε);
			}

			/// <summary>
			/// Stop criterion based on the Hessian norm of w1 - w = Δw, which is Δw H Δw.
			/// </summary>
			/// <param name="ε">The threshold of normalized squared euclidean norm.</param>
			/// <returns>Returns the criterion function.</returns>
			/// <remarks>
			/// Euclidean norm is normalized by division by the dimension.
			/// </remarks>
			public static LinearSolveStopCriterion GetHessianNormCriterion(double ε)
			{
				return (w, w1, H, Mg1) => 
				{ 
					var Δw = w1 - w; 
					return Δw * H(Δw) / Δw.Length < ε; 
				};
			}

			#endregion

			#region Construction

			/// <summary>
			/// Create.
			/// </summary>
			public LinearSolveOptions()
			{
				this.stopCriterion = GetResidualStopCriterion(1e-5);
			}

			#endregion

			#region Private fields

			private LinearSolveStopCriterion stopCriterion;

			#endregion

			#region Public properties

			/// <summary>
			/// Stop criterion for iterations.
			/// Default is that 
			/// normalized squared norm of preconditioned residual M(Hw + b)
			/// be under threshold 1e-5.
			/// </summary>
			public LinearSolveStopCriterion StopCriterion
			{
				get
				{
					return this.stopCriterion;
				}
				set
				{
					if (value == null) throw new ArgumentNullException("value");
					this.stopCriterion = value;
				}
			}

			#endregion
		}

		#endregion

		#region Public methods

		/// <summary>
		/// Solve the linear system b + Hw = 0 for w, having H positive definite.
		/// </summary>
		/// <param name="H">
		/// Tensor compatible with vector <paramref name="b"/>.
		/// Must be linear and positive definite, otherwise we get unexpected behaviour.
		/// </param>
		/// <param name="b">
		/// A vector. Can be thought as the negative right hand side: Hw = -b.
		/// </param>
		/// <param name="w">
		/// Initial value for w. It affects convergence speed. But for 
		/// any value of w, convergence is guaranteed after N steps in theory,
		/// where N is the dimensionality of the problem.
		/// In practice, for large N there might be numerical instability.
		/// </param>
		/// <param name="options">Options for the algorithm.</param>
		/// <param name="maxIterations">
		/// The maximum iterations of the algorithm. In theory, the exact result is found
		/// after N iterations, where N is the dimensionality of the problem.
		/// In practice, for large N there might be numerical instability.
		/// If left default or zero, it is taken as N.
		/// </param>
		/// <param name="M">An optional preconditioner. Default is identity.</param>
		/// <returns>
		/// Returns the solution approximated after the designated iterations.
		/// In theory, it always exists since <paramref name="H"/> is positive definite.
		/// </returns>
		/// <remarks>
		/// It is stressed that H be positive definite.
		/// </remarks>
		public static Vector LinearSolve(
			Vector.Tensor H, 
			Vector b, 
			Vector w, 
			LinearSolveOptions options, 
			int maxIterations = 0,
			Vector.Tensor M = null)
		{
			if (H == null) throw new ArgumentNullException("H");
			if (b == null) throw new ArgumentNullException("b");
			if (w == null) throw new ArgumentNullException("w");
			if (options == null) throw new ArgumentNullException("options");
			if (M == null) M = Vector.IdentityTensor;

			if (maxIterations < 0) 
				throw new ArgumentException("maxIterations must non negative.", "maxIterations");
			if (maxIterations > b.Length)
				throw new ArgumentException("maxIterations must not be greater than vector length.", "maxIterations");

			if (maxIterations == 0) maxIterations = b.Length;

			var d = new Vector(b.Length);
			
			double μ = 0.0;

			// Solution w(n+1) of next iteration.
			Vector w1 = null;

			var g = b + H(w);

			var z = M(g);

			var gz = g * z;

			for (int k = 0; k < maxIterations; k++)
			{
				if (k > 0)
				{
					d = -z + μ * d;
				}
				else
				{
					d = -z;
				}

				var Hd = H(d);

				var dHd = d * Hd;

				if (dHd < 1e-16) return w;

				var β = gz / dHd;

				w1 = w + β * d;

				if (k < maxIterations - 1)
				{
					// Compute fast g1 = g + β * Hd reusing Hd, but recompute slow g1 = b + H(w1) 
					// from scratch every 25 iterations in order to inhibit propagation of arithmetic
					// errors.
					//var g1 =
					//  k % 25 == 0 ?
					//  b + H(w1) :
					//  g + β * Hd;
					var g1 =
						b + H(w1);

					var z1 = M(g1);

					if (options.StopCriterion(w, w1, H, z1)) break;

					var gz1 = g1 * z1;

					// Fletcher-Reeves.
					μ = gz1 / gz;

					g = g1;
					w = w1;
					z = z1;
					gz = gz1;
				}
			}

			return w1;
		}

		#endregion

	}
}
