using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using Gramma.Vectors;
using Gramma.Vectors.ExtraExtensions;

namespace Gramma.Optimization
{
	/// <summary>
	/// Contains methods for optimization using conjugate gradient algorithm variants.
	/// </summary>
	public static class ConjugateGradient
	{
		#region Delegates definitions

		/// <summary>
		/// A predicate to signal early exit of an algorithm 
		/// based on the current Newton step Δw, the Hessian H and the 
		/// preconditioned residual or gradient Mg.
		/// </summary>
		/// <param name="Δw">The calculatd newton step.</param>
		/// <param name="H">
		/// The matrix of the linear system, which, in the context of a 
		/// Newton step, is the Hessian of the function.
		/// </param>
		/// <param name="Mg">
		/// The preconditioned residual of the linear system, which, in the context of a newton step,
		/// is the preconditioned gradient.
		/// </param>
		/// <returns>
		/// Must return true if the desired accuracy is met in order to stop the iterations.
		/// </returns>
		public delegate bool NewtonStopCriterion(Vector Δw, Vector.Tensor H, Vector Mg);

		/// <summary>
		/// A predicate to signal early exit of an algorithm 
		/// based on the residual or slope g.
		/// </summary>
		/// <param name="Δw">Step updating the solution.</param>
		/// <param name="Mg">The preconditioned gradient.</param>
		/// <returns>
		/// Must return true if the desired accuracy is met in order to stop the iterations.
		/// </returns>
		public delegate bool LineSearchMinimizeStopCriterion(Vector Δw, Vector Mg);

		/// <summary>
		/// Function prototype for an oracle yielding a preconditioner
		/// for <see cref="TruncatedNewtonConstrainedMinimize"/> 
		/// and <see cref="LineSearchConstrainedMinimize"/> methods.
		/// </summary>
		/// <param name="t">
		/// Logarithmic barrier scale. See slide 12-6 of Convex Optimization I, Boyd, Stanford.
		/// </param>
		/// <returns>
		/// Returns the appropriate preconditioner as function from input space to tensor space
		/// parameterized for t.
		/// </returns>
		public delegate TensorFunction ConstrainedMinimizePreconditioner(double t);

		#endregion

		#region Auxilliary classes

		/// <summary>
		/// Base class for options used in uncostrained minimization methods.
		/// </summary>
		[Serializable]
		public abstract class MinimizeOptions
		{
			#region Standard preconditioners

			/// <summary>
			/// Get a standard Jacobi preconditioner, which is a diagonal matrix
			/// having elements the inverse of the corresponding diagonal elements
			/// of the Hessian.
			/// </summary>
			/// <param name="d2fd">
			/// A list of functions returning the diagonal elements of the hessian at a given point.
			/// </param>
			/// <returns>
			/// Returns a function from input space to the approprate preconditioner matrix.
			/// </returns>
			public static TensorFunction GetJacobiPreconditioner(IList<ScalarFunction> d2fd)
			{
				if (d2fd == null) throw new ArgumentNullException("d2fd");

				return w =>
					Vector.GetDiagonalTensor(d2fd.Select(Hdi => 1 / Hdi(w)));
			}

			/// <summary>
			/// Get a standard Jacobi preconditioner, which is a diagonal matrix
			/// having elements the inverse of the corresponding diagonal elements
			/// of the Hessian.
			/// </summary>
			/// <param name="d2fd">
			/// A function returning the diagonal elements of the hessian at a given point.
			/// </param>
			/// <returns>
			/// Returns a function from input space to the approprate proconditioner matrix.
			/// </returns>
			public static TensorFunction GetJacobiPreconditioner(VectorFunction d2fd)
			{
				if (d2fd == null) throw new ArgumentNullException("d2fd");

				return w =>
					Vector.GetDiagonalTensor(d2fd(w).Select(Hdi => 1 / Hdi));
			}

			#endregion
		}

		/// <summary>
		/// Options for the <see cref="TruncatedNewtonMinimize"/> method.
		/// </summary>
		[Serializable]
		public class TruncatedNewtonMinimizeOptions : MinimizeOptions
		{
			#region Standard stop criteria

			/// <summary>
			/// Stop criterion based on normalized Hessian norm of the
			/// Newton step Δx.
			/// </summary>
			/// <param name="ε">The error threshold.</param>
			/// <returns>Returns the stop criterion function.</returns>
			public static NewtonStopCriterion GetHessianNormCriterion(double ε)
			{
				return (Δw, H, b) => (Δw * H(Δw) / Δw.Length < ε);
			}

			/// <summary>
			/// Stop criterion based on normalized preconditioned gradient Mdf = Mb.
			/// </summary>
			/// <param name="ε">The error threshold.</param>
			/// <returns>Returns the stop criterion function.</returns>
			public static NewtonStopCriterion GetGradientCriterion(double ε)
			{
				return (Δw, H, Mb) => (Mb.Norm2 / Δw.Length < ε);
			}

			#endregion

			#region Private fields

			private double krylovIterationsCountLogOffset;

			private int minKrylovIterationsCount;

			private KrylovSolver.LinearSolveOptions linearSolveOptions;

			private NewtonStopCriterion newtonStopCriterion;

			#endregion

			#region Construction

			/// <summary>
			/// Create.
			/// </summary>
			public TruncatedNewtonMinimizeOptions()
			{
				this.krylovIterationsCountLogOffset = 0.0;
				this.newtonStopCriterion = GetGradientCriterion(1e-5);
				this.minKrylovIterationsCount = 4;
				this.linearSolveOptions = new KrylovSolver.LinearSolveOptions();
			}

			#endregion

			#region Public properties

			/// <summary>
			/// Logarithmic offset of iterations count. A zero value means that iterations count
			/// starts from 1 and ends to fourth root of dimensions. Maximum value is 0.75 which 
			/// makes the iterations count end to the number of dimensions. Default is 0.0.
			/// </summary>
			public double KrylovIterationsCountLogOffset
			{
				get
				{
					return this.krylovIterationsCountLogOffset;
				}
				set
				{
					if (value > 0.75) throw new ArgumentException("Maximum value is 0.75.", "value");

					this.krylovIterationsCountLogOffset = value;
				}
			}

			/// <summary>
			/// Minimum Krylov sequence iterations per Newton step. Default is 4.
			/// </summary>
			public int MinKrylovIterationsCount
			{
				get
				{
					return this.minKrylovIterationsCount;
				}
				set
				{
					if (value <= 0) throw new ArgumentException("value must me positive.");

					this.minKrylovIterationsCount = value;
				}
			}

			/// <summary>
			/// A predicate to signal early exit 
			/// based on the current Newton step w, the Hessian H and the residual or gradient g.
			/// Default criterion is based on residual Hw + g with threshold 1e-5.
			/// </summary>
			public NewtonStopCriterion NewtonStopCriterion
			{
				get
				{
					return this.newtonStopCriterion;
				}
				set
				{
					if (value == null) throw new ArgumentNullException("value");
					this.newtonStopCriterion = value;
				}
			}

			/// <summary>
			/// Options for linear system solution approximator using Krylov sequences.
			/// Defaults are as specified by <see cref="KrylovSolver.LinearSolveOptions"/> class.
			/// </summary>
			public KrylovSolver.LinearSolveOptions LinearSolveOptions
			{
				get
				{
					return this.linearSolveOptions;
				}
				set
				{
					if (value == null) throw new ArgumentNullException("value");
					this.linearSolveOptions = value;
				}
			}

			#endregion
		}

		/// <summary>
		/// Options for <see cref="LineSearchMinimize"/> method.
		/// </summary>
		[Serializable]
		public class LineSearchMinimizeOptions : MinimizeOptions
		{
			#region Standard stop criteria

			/// <summary>
			/// Get stop criterion based on normalized squared euclidean norm of the gradient.
			/// </summary>
			/// <param name="ε">The normalized norm threshold.</param>
			/// <returns>Returns the criterion function.</returns>
			public static LineSearchMinimizeStopCriterion GetGradientNormCriterion(double ε)
			{
				return (Δw, Mg) => (Mg.Norm2 / Mg.Length < ε);
			}

			#endregion

			#region Private fields

			private LineSearchMinimizeStopCriterion stopCriterion;

			#endregion

			#region Construction

			/// <summary>
			/// Create.
			/// </summary>
			public LineSearchMinimizeOptions()
			{
				this.stopCriterion = GetGradientNormCriterion(1e-5);
				this.LineSearchThreshold = 1e-6;
				this.MaxLineSearchIterations = 12;
				this.MinLineSearchLength = 1e-6;
			}

			#endregion

			#region Public properties

			/// <summary>
			/// Stop criterion based on the gradient.
			/// Default is criterion based on normalized squared norm of gradient
			/// with threshold 1e-5.
			/// </summary>
			public LineSearchMinimizeStopCriterion StopCriterion
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

			/// <summary>
			/// If during line search we find a point along a line
			/// whose value difference between two neighbours, left and right,
			/// varies less than this threshold, quit the search.
			/// </summary>
			public double LineSearchThreshold { get; set; }

			/// <summary>
			/// The maximum number of iterations performed per line search. Default is 12.
			/// </summary>
			public int MaxLineSearchIterations { get; set; }

			/// <summary>
			/// The minimum relative length β searching along a line for the minimum of 
			/// the function f(x + β * Δx).
			/// Below this length the search stops.
			/// Default is 1e-6;
			/// </summary>
			public double MinLineSearchLength { get; set; }

			#endregion
		}

		/// <summary>
		/// Contains the optimum feasible point and
		/// the Lagrange multipliers of the dual problem, which certify the 
		/// validity of the solution.
		/// </summary>
		[Serializable]
		public class SolutionCertificate
		{
			#region Construction

			/// <summary>
			/// Create.
			/// </summary>
			/// <param name="optimum">
			/// The input variable where the function attains its minimum.
			/// </param>
			/// <param name="λ">
			/// The Lagrange multipliers of the dual problem
			/// corresponding to the constraint inequalities.
			/// </param>
			public SolutionCertificate(Vector optimum, Vector λ)
			{
				if (optimum == null) throw new ArgumentNullException("optimum");
				if (λ == null) throw new ArgumentNullException("λ");

				this.Optimum = optimum;
				this.λ = λ;
			}

			#endregion

			#region Public properties

			/// <summary>
			/// The input variable where the function attains its minimum.
			/// </summary>
			public Vector Optimum { get; private set; }

			/// <summary>
			/// The Lagrange multipliers of the dual problem
			/// corresponding to the constraint inequalities.
			/// </summary>
			public Vector λ { get; private set; }

			#endregion
		}

		/// <summary>
		/// Common options for <see cref="TruncatedNewtonConstrainedMinimize"/>
		/// and <see cref="LineSearchConstrainedMinimize"/> methods.
		/// </summary>
		[Serializable]
		public abstract class ConstrainedMinimizeOptions
		{
			#region Standard preconditioners

			/// <summary>
			/// The identity preconditioner.
			/// </summary>
			public static readonly ConstrainedMinimizePreconditioner IdentityPreconditioner = 
				t => (x => Vector.IdentityTensor);

			/// <summary>
			/// Jacobi preconditioner, i.e. diagonal preconditioner consisting of the
			/// inverse elements of the diagonal of the total Hessian.
			/// </summary>
			/// <param name="d2fd">The diagonal of the Hessian of the goal function.</param>
			/// <param name="constraintCount">The number of constraints.</param>
			/// <param name="fc">The collection of constraint functions.</param>
			/// <param name="dfc">The collection of the derivatives of the constraint functions.</param>
			/// <param name="d2fcd">
			/// The collection of the diagonals of the Hessians of the constraint functions.
			/// </param>
			/// <returns>Returns the preconditioner as a tensor.</returns>
			public static ConstrainedMinimizePreconditioner GetJacobiPreconditioner(
				VectorFunction d2fd,
				int constraintCount,
				Func<int, ScalarFunction> fc,
				Func<int, VectorFunction> dfc,
				Func<int, VectorFunction> d2fcd)
			{
				if (d2fd == null) throw new ArgumentNullException("d2fd");
				if (fc == null) throw new ArgumentNullException("fc");
				if (dfc == null) throw new ArgumentNullException("dfc");
				if (d2fcd == null) throw new ArgumentNullException("d2fcd");

				//return delegate(double t)
				//{
				//  // t is the log barrier scale.
				//  return delegate(Vector w)
				//  {
				//    if (w == null) throw new ArgumentNullException("w");

				//    // Evaluate constraint values at w.
				//    IList<double> fcw = new List<double>(constraintCount);

				//    for (int i = 0; i < constraintCount; i++)
				//    {
				//      fcw[i] = fc(i)(w);
				//    }

				//    // Evaluate constraint derivative values at w.
				//    IList<Vector> dfcw = new List<Vector>(constraintCount);

				//    for (int i = 0; i < constraintCount; i++)
				//    {
				//      dfcw[i] = dfc(i)(w);
				//    }

				//    // Evaluate the diagonals of the Hessian of the constraints.
				//    IList<Vector> d2fcdw = new List<Vector>(d2fcd.Count);

				//    for (int i = 0; i < d2fcd.Count; i++)
				//    {
				//      d2fcdw[i] = d2fcd[i](w);
				//    }

				//    var range = Enumerable.Range(0, constraintCount);

				//    // Diagonal of the Hessian of the log barrier.
				//    // See slide 12-5, Convex Optimization I, Boyd, Stanford.
				//    Vector φ =
				//      range.Sum(i => (dfcw.Select(dfcwi => dfcwi * dfcwi) / fcw[i] - d2fcdw[i]) / fcw[i]);

				//    // Diagonal of the total Hessian.
				//    Vector Hd = t * d2fd(w) + φ;

				//    // Create a diagonal tensor consisting of the inverse of the diagonal elements
				//    // of the total Hessian.
				//    return Vector.GetDiagonalTensor(Hd.Select(Hdi => 1 / Hdi));

				//  };
				//};

				var iRange = Enumerable.Range(0, constraintCount);

				return 
					t =>
						w => Vector.GetDiagonalTensor(
							(
								t * d2fd(w) + 
								iRange.AsParallel().Sum(i => 
									1.0 / fc(i)(w) * 
										(1.0 / fc(i)(w) * dfc(i)(w).Select(x => x * x) - d2fcd(i)(w)))
							) // Up to here this is the diagonal of  t * φ0(w) + φ(w).
							.Select(Tii => 1 / Tii) // Invert.
								
						);
			}

			#endregion

			#region Private fields

			private double dualityGap;

			private double barrierScaleFactor;

			private double barrierInitialScale;

			#endregion

			#region Construction

			/// <summary>
			/// Create.
			/// </summary>
			public ConstrainedMinimizeOptions()
			{
				this.dualityGap = 1e-5;
				this.barrierScaleFactor = 10.0;
				this.barrierInitialScale = 1.0;
			}

			#endregion

			#region Public properties

			/// <summary>
			/// Desired duality gap for the solution.
			/// Must be greater than zero. Default is 1e-5.
			/// </summary>
			public double DualityGap
			{
				get
				{
					return this.dualityGap;
				}
				set
				{
					if (value <= 0.0) throw new ArgumentException("value must be positive.");
					this.dualityGap = value;
				}
			}

			/// <summary>
			/// The increase factor μ of the scale t of the log barrier.
			/// Must be greater than 1.0. Default is 10.0.
			/// </summary>
			/// <remarks>
			/// See slide 12-11, Convex Optimization I, Boyd, Stanford.
			/// </remarks>
			public double BarrierScaleFactor
			{
				get
				{
					return this.barrierScaleFactor;
				}
				set
				{
					if (value <= 1.0) throw new ArgumentException("value must be greater than 1.0.");
					this.barrierScaleFactor = value;
				}
			}

			/// <summary>
			/// The start scale t0 of the log barrier.
			/// Must be greater than zero. Default is 1.0.
			/// </summary>
			/// <remarks>
			/// See slide 12-11, Convex Optimization I, Boyd, Stanford.
			/// </remarks>
			public double BarrierInitialScale
			{
				get
				{
					return this.barrierInitialScale;
				}
				set
				{
					if (value <= 0.0) throw new ArgumentException("value must be positive.");
					this.barrierInitialScale = value;
				}
			}

			#endregion
		}

		/// <summary>
		/// Options for <see cref="TruncatedNewtonConstrainedMinimize"/> method.
		/// </summary>
		[Serializable]
		public class TruncatedNewtonConstrainedMinimizeOptions : ConstrainedMinimizeOptions
		{
			#region Private fields

			private double krylovIterationsCountLogOffsetStart;

			private double krylovIterationsCountLogOffsetEnd;

			private double krylovIterationsCountLogOffsetStep;

			private Func<double, NewtonStopCriterion> stopCriterion;

			#endregion

			#region Construction

			/// <summary>
			/// Create.
			/// </summary>
			public TruncatedNewtonConstrainedMinimizeOptions()
			{
				this.krylovIterationsCountLogOffsetStart = -0.25;
				this.krylovIterationsCountLogOffsetEnd = 0.0;
				this.krylovIterationsCountLogOffsetStep = 0.05;
				this.stopCriterion = TruncatedNewtonMinimizeOptions.GetGradientCriterion;
			}

			#endregion

			#region Public properties

			/// <summary>
			/// The early stop criterion to be applied during Krylov sequence generation,
			/// parameterized by duality gap squared. Default 
			/// <see cref="TruncatedNewtonMinimizeOptions.GetGradientCriterion"/>.
			/// Another option may be 
			/// the standard <see cref="TruncatedNewtonMinimizeOptions.GetHessianNormCriterion"/>,
			/// (although it is more difficultly met)
			/// or any other user supplied criterion.
			/// </summary>
			public Func<double, NewtonStopCriterion> StopCriterion 
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

			/// <summary>
			/// Logarithmic start offset of Krylov iterations count.
			/// Default is -0.25;
			/// </summary>
			/// <remarks>
			/// The first Newton steps don't need accuracy as the duality gap is high.
			/// Thus a lower <see cref="KrylovIterationsCountLogOffsetStart"/>
			/// is in effect diring the start of the algorithm
			/// in order to save time.
			/// The offset will progress up to <see cref="KrylovIterationsCountLogOffsetEnd"/>
			/// in increments of <see cref="KrylovIterationsCountLogOffsetStep"/>
			/// as higher accuracy is needed in subsequent iterations.
			/// </remarks>
			public double KrylovIterationsCountLogOffsetStart
			{
				get
				{
					return this.krylovIterationsCountLogOffsetStart;
				}
				set
				{
					this.krylovIterationsCountLogOffsetStart = value;
				}
			}

			/// <summary>
			/// Logarithmic end offset of Krylov iterations count.
			/// Default is 0.0.
			/// </summary>
			/// <remarks>
			/// The first Newton steps don't need accuracy as the duality gap is high.
			/// Thus a lower <see cref="KrylovIterationsCountLogOffsetStart"/>
			/// is in effect diring the start of the algorithm
			/// in order to save time.
			/// The offset will progress up to <see cref="KrylovIterationsCountLogOffsetEnd"/>
			/// in increments of <see cref="KrylovIterationsCountLogOffsetStep"/>
			/// as higher accuracy is needed in subsequent iterations.
			/// </remarks>
			public double KrylovIterationsCountLogOffsetEnd
			{
				get
				{
					return this.krylovIterationsCountLogOffsetEnd;
				}
				set
				{
					this.krylovIterationsCountLogOffsetEnd = value;
				}
			}

			/// <summary>
			/// Logarithmic offset step increment of Krylov iterations count.
			/// Default is 0.05.
			/// </summary>
			/// <remarks>
			/// The first Newton steps don't need accuracy as the duality gap is high.
			/// Thus a lower <see cref="KrylovIterationsCountLogOffsetStart"/>
			/// is in effect diring the start of the algorithm
			/// in order to save time.
			/// The offset will progress up to <see cref="KrylovIterationsCountLogOffsetEnd"/>
			/// in increments of <see cref="KrylovIterationsCountLogOffsetStep"/>
			/// as higher accuracy is needed in subsequent iterations.
			/// </remarks>
			public double KrylovIterationsCountLogOffsetStep
			{
				get
				{
					return this.krylovIterationsCountLogOffsetStep;
				}
				set
				{
					this.krylovIterationsCountLogOffsetStep = value;
				}
			}

			#endregion
		}

		/// <summary>
		/// Options for the <see cref="LineSearchConstrainedMinimize"/> method.
		/// </summary>
		[Serializable]
		public class LineSearchConstrainedMinimizeOptions : ConstrainedMinimizeOptions
		{
			#region Private fields

			private Func<double, LineSearchMinimizeStopCriterion> stopCriterion;

			#endregion

			#region Construction

			/// <summary>
			/// Create.
			/// </summary>
			public LineSearchConstrainedMinimizeOptions()
			{
				this.stopCriterion = LineSearchMinimizeOptions.GetGradientNormCriterion;
			}

			#endregion

			#region Public properties

			/// <summary>
			/// Stop criterion, parameterized by the duality gap squared.
			/// Default is <see cref="LineSearchMinimizeOptions.GetGradientNormCriterion"/>.
			/// </summary>
			public Func<double, LineSearchMinimizeStopCriterion> StopCriterion
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
		/// Unconstrained truncated Newton minimization of a function.
		/// The function is implied to be strictly convex, that is, having
		/// positive definite Hessian.
		/// </summary>
		/// <param name="df">Derivative of the function.</param>
		/// <param name="d2f">Hessian of the function.</param>
		/// <param name="w">Initial estimation of solution.</param>
		/// <param name="outOfDomainIndicator">
		/// Indicator function returning true when 
		/// w is out of the domain of the problem.
		/// </param>
		/// <param name="options">Options for the algorithm.</param>
		/// <param name="M">Optional preconditioner at point w. Default is identity.</param>
		/// <returns>
		/// Returns the optimum point where the function
		/// attains its minimum.
		/// </returns>
		public static Vector TruncatedNewtonMinimize(
			VectorFunction df,
			TensorFunction d2f,
			Vector w,
			Func<Vector, bool> outOfDomainIndicator,
			TruncatedNewtonMinimizeOptions options,
			TensorFunction M = null)
		{
			if (df == null) throw new ArgumentNullException("f");
			if (d2f == null) throw new ArgumentNullException("d2f");
			if (w == null) throw new ArgumentNullException("w");
			if (outOfDomainIndicator == null) throw new ArgumentNullException("outOfDomain");
			if (options == null) throw new ArgumentNullException("options");

			if (M == null) M = (x => Vector.IdentityTensor);

			double iterationsScale = 0.0 + options.KrylovIterationsCountLogOffset;
			double iterationsScaleEnd = 0.25 + options.KrylovIterationsCountLogOffset;

			double logDimensions = Math.Log(w.Length);

			while (true)
			{
				var b = df(w); // Derivative at w.

				var H = d2f(w); // Hessian tensor at w;

				var P = M(w); // Preconditioner at P.

				if (options.NewtonStopCriterion(w, H, P(b))) break;

				// Number of Krylov iterations. Logarithmically crank up the number up to
				// fourth root of dimensions by default.
				int krylovIterationsCount = 
					Math.Max((int)Math.Exp(logDimensions * iterationsScale), options.MinKrylovIterationsCount);

				krylovIterationsCount = Math.Min(krylovIterationsCount, b.Length);

				var Δw = KrylovSolver.LinearSolve(
					H, 
					b, 
					new Vector(b.Length), 
					options.LinearSolveOptions, 
					krylovIterationsCount,
					P);

				// Backtrack if w1 = w + Δw is out of the domain.

				double β;

				Vector w1;

				for (w1 = w + Δw, β = 1.0; outOfDomainIndicator(w1); )
				{
					β /= 2.0;
					w1 = w + β * Δw;
				}

				w = w1;

				if (iterationsScale < iterationsScaleEnd) iterationsScale += 0.05;
			}

			return w;
		}

		/// <summary>
		/// Minimize a convex scalar function using the Conjugate Gradient method.
		/// Uses line search to void computing the Hessian.
		/// </summary>
		/// <param name="f">The scalar function to minimize.</param>
		/// <param name="df">The gradient of the scalar function.</param>
		/// <param name="w">Initial estimation of solution.</param>
		/// <param name="options">Options for the algorithm.</param>
		/// <param name="M">Optional preconditioner at point w. Default is identity.</param>
		/// <param name="outOfDomainIndicator">
		/// Optional fast function to indicate true when w is out of domain instead of 
		/// fully evaluating f(w) and testing for infinity or NaN.
		/// </param>
		/// <returns>
		/// Returns the optimum point where the function <paramref name="f"/>
		/// attains its minimum.
		/// </returns>
		public static Vector LineSearchMinimize(
			ScalarFunction f,
			VectorFunction df,
			Vector w,
			LineSearchMinimizeOptions options,
			TensorFunction M = null,
			Func<Vector, bool> outOfDomainIndicator = null)
		{
			if (f == null) throw new ArgumentNullException("f");
			if (df == null) throw new ArgumentNullException("f");
			if (w == null) throw new ArgumentNullException("w");
			if (options == null) throw new ArgumentNullException("options");

			if (M == null) M = (x => Vector.IdentityTensor);

			if (outOfDomainIndicator == null) outOfDomainIndicator = (x => false);

			var g = df(w);
			var P = M(w); // Preconditioner at w.
			var z = P(g);
			var d = -z;

			var fβ = f(w);

			while (true)
			{
				double fβ1;

				/* Normalize search direction by |d| squared, and denormalize β afterwards with it.
				 * But also make sure that the normalizer isn't too small to cause exit criterion activation. */

				var searchDirectionNormalizer = Math.Max(1.0 / (Math.Sqrt(d.Norm2) + 0.00000001), 100.0 * Math.Sqrt(options.MinLineSearchLength * d.Length / d.Norm2));

				var β = ConvexMinimize(fβ, f, w, d * searchDirectionNormalizer, options, outOfDomainIndicator, out fβ1) * searchDirectionNormalizer;

				if (β * Math.Sqrt(d.Norm2 / d.Length) < options.MinLineSearchLength) break;

				var Δw = β * d;

				if (options.StopCriterion(Δw, z)) break;

				w = w + Δw;

				var g1 = df(w);

				P = M(w);

				var z1 = P(g1);

				// Reset the algorithm if we find a higher goal value.
				if (fβ1 < fβ || g * Δw < 0.0)
				{
					// Polack - Ribiere.
					var μ = g1 * (z1 - z) / (g * z);

					d = -z1 + μ * d;
				}
				else
				{
					d = -z1;
				}

				z = z1;
				g = g1;
				fβ = fβ1;
			}

			return w;
		}

		/// <summary>
		/// Minimize a convex scalar function under convex inequality constraints
		/// using the Conjugate Gradient method.
		/// </summary>
		/// <param name="f">The scalar function to minimize.</param>
		/// <param name="df">The gradient of the scalar function.</param>
		/// <param name="w">Initial estimation of solution.</param>
		/// <param name="constraintCount">Number of constraints.</param>
		/// <param name="fc">The collection of contstraints fc(i) &lt;= 0.</param>
		/// <param name="dfc">The collection of the derivatives of the consatraints.</param>
		/// <param name="options">Options for the algorithm.</param>
		/// <param name="M">
		/// Optional preconditioner as a function of barrier scale t to a TensorFunction.
		/// Default is identity, i.e. no preconditioner.
		/// See slide 12-6, Convex Optimization I, Boyd, Stanford.
		/// </param>
		/// <returns>
		/// Returns a certificate of the solution, containing the optimum point 
		/// where the function <paramref name="f"/> attains its minimum 
		/// along with the the lagrange multipliers of the dual problem to allow verification.
		/// </returns>
		public static SolutionCertificate LineSearchConstrainedMinimize(
			ScalarFunction f,
			VectorFunction df,
			Vector w,
			int constraintCount,
			Func<int, ScalarFunction> fc,
			Func<int, VectorFunction> dfc,
			LineSearchConstrainedMinimizeOptions options,
			ConstrainedMinimizePreconditioner M = null)
		{
			if (f == null) throw new ArgumentNullException("f");
			if (df == null) throw new ArgumentNullException("df");
			if (w == null) throw new ArgumentNullException("w");
			if (fc == null) throw new ArgumentNullException("fc");
			if (dfc == null) throw new ArgumentNullException("dfc");
			if (options == null) throw new ArgumentNullException("options");

			if (M == null) M = LineSearchConstrainedMinimizeOptions.IdentityPreconditioner;

			var constraintsRange = Enumerable.Range(0, constraintCount);

			ScalarFunction φ =
				x => -constraintsRange.AsParallel().Sum(i => Math.Log(-fc(i)(x)));

			VectorFunction dφ =
				x => constraintsRange.AsParallel().Sum(i => 1 / -fc(i)(x) * dfc(i)(x));

			Func<Vector, bool> outOfDomainIndicator =
				x => constraintsRange.AsParallel().Any(i => fc(i)(x) > 0.0);

			// Estimator of Lagrange multipliers as a certificate for the solution.
			// See Convex Optimization I, Stanford, Boyd, slide 12-7.
			Func<double, VectorFunction> λ =
				t =>
					x =>
						constraintsRange.AsParallel().Select(i => -1.0 / (t * fc(i)(x)));

			return LineSearchConstrainedMinimize(
				f,
				df,
				w,
				φ,
				dφ,
				λ,
				options,
				outOfDomainIndicator,
				M);
		}

		/// <summary>
		/// Minimize a convex scalar function under convex inequality constraints
		/// using the Conjugate Gradient method.
		/// </summary>
		/// <param name="f">The scalar function to minimize.</param>
		/// <param name="df">The gradient of the scalar function.</param>
		/// <param name="w">Initial estimation of solution.</param>
		/// <param name="φ">
		/// Manually specified barrier function. 
		/// This is iteratively scaled and added to the 
		/// goal function <paramref name="f"/>,
		/// forming the goal of the underlying unconstrained minimization run.
		/// </param>
		/// <param name="dφ">
		/// Manually specified gradient of the barrier function.
		/// This is iteratively scaled and added to the 
		/// gradient of the goal function <paramref name="df"/>, 
		/// forming the gradient of the goal of the underlying unconstrained minimization run.
		/// </param>
		/// <param name="λ">
		/// Lagrange multiplier estimate. Required to provide solution certificate.
		/// See slide 12-7, Convex Optimization I, Boyd, Stanford.
		/// </param>
		/// <param name="options">Options for the algorithm.</param>
		/// <param name="outOfDomainIndicator">
		/// Optional predicate function returning true if the input argument is out of domain.
		/// This is used as a fast shortcut during line search and backtracking. 
		/// Othwerise, the slower path is taken, which is to fully evaluate 
		/// the goal function and test for NaN or Infinity.
		/// </param>
		/// <param name="M">
		/// Optional preconditioner as a function of barrier scale t to a TensorFunction.
		/// Default is identity, i.e. no preconditioner.
		/// See slide 12-6, Convex Optimization I, Boyd, Stanford.
		/// </param>
		/// <returns>
		/// Returns a certificate of the solution, containing the optimum point 
		/// where the function <paramref name="f"/> attains its minimum 
		/// along with the the lagrange multipliers of the dual problem to allow verification.
		/// </returns>
		/// <remarks>
		/// The caller manually supplies the barrier function, its gradient and the 
		/// Lagrange multiplier estimator instead of the constraint functions. 
		/// This overload is intended to achieve substantial speedup 
		/// when the above functions are of special form and effeciently computable.
		/// </remarks>
		public static SolutionCertificate LineSearchConstrainedMinimize(
			ScalarFunction f,
			VectorFunction df,
			Vector w,
			ScalarFunction φ,
			VectorFunction dφ,
			Func<double, VectorFunction> λ,
			LineSearchConstrainedMinimizeOptions options,
			Func<Vector, bool> outOfDomainIndicator = null,
			ConstrainedMinimizePreconditioner M = null)
		{
			if (f == null) throw new ArgumentNullException("f");
			if (df == null) throw new ArgumentNullException("df");
			if (w == null) throw new ArgumentNullException("w");
			if (φ == null) throw new ArgumentNullException("φ");
			if (dφ == null) throw new ArgumentNullException("dφ");
			if (λ == null) throw new ArgumentNullException("λ");
			if (options == null) throw new ArgumentNullException("options");

			if (M == null) M = LineSearchConstrainedMinimizeOptions.IdentityPreconditioner;

			double μ = options.BarrierScaleFactor;

			var lineSearchMinimizeOptions = new LineSearchMinimizeOptions();

			Func<double, LineSearchMinimizeStopCriterion> lineSearchMinimizeStopCriterion =
				options.StopCriterion;

			double t;

			bool dualityGapReached = false;

			for (
				t = options.BarrierInitialScale; 
				!dualityGapReached; 
				t *= μ)
			{
				if (w.Length / t <= options.DualityGap)
				{
					dualityGapReached = true;

					// Use no steeper barrier than the one required by the desired duality gap.
					// This clipping will most probably take effect during the last scale iteration.
					t = w.Length / options.DualityGap;
				}

				var t2 = t * t;

				ScalarFunction L =
					x => t * f(x) + φ(x);

				VectorFunction dL =
					x => t * df(x) + dφ(x);

				lineSearchMinimizeOptions.StopCriterion = lineSearchMinimizeStopCriterion(w.Length / t2);
				lineSearchMinimizeOptions.MaxLineSearchIterations = Math.Min(1 + (int)Math.Sqrt(t), 32);

				double lineSearchAccuracy = 1.0 / t2;

				lineSearchMinimizeOptions.MinLineSearchLength = lineSearchAccuracy;
				lineSearchMinimizeOptions.LineSearchThreshold = lineSearchAccuracy;

				Trace.WriteLine(String.Format("Log barrier scale: {0}", t));

				w = LineSearchMinimize(L, dL, w, lineSearchMinimizeOptions, M(t), outOfDomainIndicator);
			}

			// Reverse last increment of t.
			t /= μ;

			return new SolutionCertificate(w, λ(t)(w));
		}

		/// <summary>
		/// Minimize a convex scalar function under convex inequality constraints
		/// using the Truncated Newton method.
		/// </summary>
		/// <param name="df">The gradient of the scalar function.</param>
		/// <param name="d2f">The Hessian of the scalar function.</param>
		/// <param name="w">Initial estimation of solution.</param>
		/// <param name="constraintCount">Number of constraints.</param>
		/// <param name="fc">Array of constraint functions f[i](x) &lt;= 0.</param>
		/// <param name="dfc">Array of constraint functions derivatives.</param>
		/// <param name="d2fc">Array of constraint functions hessians.</param>
		/// <param name="options">Options for the algorithm.</param>
		/// <param name="M">
		/// Optional preconditioner as a function of barrier scale t to a TensorFunction.
		/// Default is identity, i.e. no preconditioner.
		/// See slide 12-6, Convex Optimization I, Boyd, Stanford.
		/// </param>
		/// <returns>
		/// Returns a certificate of the solution, containing the optimum point 
		/// where the function attains its minimum 
		/// along with the the lagrange multipliers of the dual problem to allow verification.
		/// </returns>
		public static SolutionCertificate TruncatedNewtonConstrainedMinimize(
			VectorFunction df,
			TensorFunction d2f,
			Vector w,
			int constraintCount,
			Func<int, ScalarFunction> fc,
			Func<int, VectorFunction> dfc,
			Func<int, TensorFunction> d2fc,
			TruncatedNewtonConstrainedMinimizeOptions options,
			ConstrainedMinimizePreconditioner M = null)
		{
			if (df == null) throw new ArgumentNullException("df");
			if (d2f == null) throw new ArgumentNullException("d2f");
			if (w == null) throw new ArgumentNullException("w");
			if (fc == null) throw new ArgumentNullException("fc");
			if (dfc == null) throw new ArgumentNullException("dfc");
			if (d2fc == null) throw new ArgumentNullException("d2fc");
			if (options == null) throw new ArgumentNullException("options");

			if (M == null) M = TruncatedNewtonConstrainedMinimizeOptions.IdentityPreconditioner;

			var constraintsRange = Enumerable.Range(0, constraintCount).ToArray();

			VectorFunction dφ =
				x => constraintsRange.AsParallel().Sum(i => 1 / -fc(i)(x) * dfc(i)(x));

			var d2φ = HessianOfConstraints(constraintCount, fc, dfc, d2fc);

			Func<Vector, bool> outOfDomainIndicator =
				x => constraintsRange.AsParallel().Any(i => fc(i)(x) > 0.0);

			// Estimator of Lagrange multipliers as a certificate of the solution.
			// See Convex Optimization I, Stanford, Boyd, slide 12-7.
			Func<double, VectorFunction> λ =
				t =>
					x =>
						constraintsRange.AsParallel().Select(i => -1.0 / (t * fc(i)(x)));

			return TruncatedNewtonConstrainedMinimize(
				df,
				d2f,
				w,
				dφ,
				d2φ,
				λ,
				options,
				outOfDomainIndicator,
				M);
		}

		/// <summary>
		/// Minimize a convex scalar function under convex inequality constraints
		/// using the Truncated Newton method.
		/// </summary>
		/// <param name="df">The gradient of the scalar function.</param>
		/// <param name="d2f">The Hessian of the scalar function.</param>
		/// <param name="w">Initial estimation of solution.</param>
		/// <param name="dφ">
		/// Manually specified gradient of the barrier function.
		/// This is iteratively scaled and added to the 
		/// gradient of the goal function <paramref name="df"/>, 
		/// forming the gradient of the goal of the underlying unconstrained minimization run.
		/// </param>
		/// <param name="d2φ">
		/// Manually specified Hessian of the barrier function.
		/// This is iteratively scaled and added to the 
		/// Hessian of the goal function <paramref name="d2f"/>, 
		/// forming the Hessian of the goal of the underlying unconstrained minimization run.
		/// </param>
		/// <param name="λ">
		/// Lagrange multiplier estimate. Required to provide solution certificate.
		/// See slide 12-7, Convex Optimization I, Boyd, Stanford.
		/// </param>
		/// <param name="options">Options for the algorithm.</param>
		/// <param name="outOfDomainIndicator">
		/// Optional predicate function returning true if the input argument is out of domain.
		/// This is used as a fast shortcut during line search and backtracking. 
		/// Othwerise, the slower path is taken, which is to fully evaluate 
		/// the goal function and test for NaN or Infinity.
		/// </param>
		/// <param name="M">
		/// Optional preconditioner as a function of barrier scale t to a TensorFunction.
		/// Default is identity, i.e. no preconditioner.
		/// See slide 12-6, Convex Optimization I, Boyd, Stanford.
		/// </param>
		/// <returns>
		/// Returns a certificate of the solution, containing the optimum point 
		/// where the function attains its minimum 
		/// along with the the lagrange multipliers of the dual problem to allow verification.
		/// </returns>
		/// <remarks>
		/// The caller manually supplies the gradient of the barrier function, its Hessian and the 
		/// Lagrange multiplier estimator instead of the constraint functions. 
		/// This overload is intended to achieve substantial speedup 
		/// when the above functions are of special form and effeciently computable.
		/// </remarks>
		public static SolutionCertificate TruncatedNewtonConstrainedMinimize(
			VectorFunction df,
			TensorFunction d2f,
			Vector w,
			VectorFunction dφ,
			TensorFunction d2φ,
			Func<double, VectorFunction> λ,
			TruncatedNewtonConstrainedMinimizeOptions options,
			Func<Vector, bool> outOfDomainIndicator = null,
			ConstrainedMinimizePreconditioner M = null)
		{
			if (df == null) throw new ArgumentNullException("df");
			if (d2f == null) throw new ArgumentNullException("d2f");
			if (w == null) throw new ArgumentNullException("w");
			if (dφ == null) throw new ArgumentNullException("dφ");
			if (d2φ == null) throw new ArgumentNullException("φ");
			if (λ == null) throw new ArgumentNullException("λ");
			if (options == null) throw new ArgumentNullException("options");

			if (M == null) M = TruncatedNewtonConstrainedMinimizeOptions.IdentityPreconditioner;

			double μ = options.BarrierScaleFactor;

			var newtonMinimizeOptions = new TruncatedNewtonMinimizeOptions();

			Func<double, NewtonStopCriterion> newtonStopCriterion =
				//ε => TruncatedNewtonMinimizeOptions.GetGradientCriterion(ε);
				//ε => TruncatedNewtonMinimizeOptions.GetHessianNormCriterion(ε);
				options.StopCriterion;

			double t;

			newtonMinimizeOptions.KrylovIterationsCountLogOffset =
				options.KrylovIterationsCountLogOffsetStart;

			bool dualityGapReached = false;

			for (
				t = options.BarrierInitialScale;
				!dualityGapReached;
				t *= μ)
			{
				if (w.Length / t <= options.DualityGap)
				{
					dualityGapReached = true;

					// Use no steeper barrier than the one required by the desired duality gap.
					// This clipping will most probably take effect during the last scale iteration.
					t = w.Length / options.DualityGap;
				}

				var t2 = t * t;

				VectorFunction dL =
					x => t * df(x) + dφ(x);

				TensorFunction d2L =
					(x =>
						(y =>
							t * d2f(x)(y) + d2φ(x)(y)
						)
					);

				newtonMinimizeOptions.NewtonStopCriterion =
					newtonStopCriterion(0.5 * w.Length / t);

				Trace.WriteLine(String.Format("Log barrier scale: {0}", t));

				w = TruncatedNewtonMinimize(dL, d2L, w, outOfDomainIndicator, newtonMinimizeOptions, M(t));

				if (newtonMinimizeOptions.KrylovIterationsCountLogOffset < options.KrylovIterationsCountLogOffsetEnd)
					newtonMinimizeOptions.KrylovIterationsCountLogOffset += options.KrylovIterationsCountLogOffsetStep;
			}

			// Reverse last increment of t.
			t /= μ;

			return new SolutionCertificate(w, λ(t)(w));
		}

		#endregion

		#region Private methods

		/// <summary>
		/// Compute the Hessian of the sum of the log barriers of constraints.
		/// </summary>
		/// <param name="constraintCount">The number of constraints.</param>
		/// <param name="fc">Array of constraint functions f[i](x) &lt;= 0.</param>
		/// <param name="dfc">Array of constraint functions derivatives.</param>
		/// <param name="d2fc">Array of constraint functions Hessians.</param>
		/// <returns>
		/// Returns a function that returns the Hessian tensor at a point x.
		/// </returns>
		/// <remarks>
		/// See slide 12-5 of lecture Convex Optimization I, Boyd, Stanford.
		/// </remarks>
		private static TensorFunction HessianOfConstraints(
			int constraintCount,
			Func<int, ScalarFunction> fc, 
			Func<int, VectorFunction> dfc, 
			Func<int, TensorFunction> d2fc)
		{
			var range = Enumerable.Range(0, constraintCount);

			return x =>
				y =>
				{
					Vector sum = new Vector(x.Length);
					Vector zero = new Vector(x.Length);

					for (int i = 0; i < constraintCount; i++)
					{
						double fcix = fc(i)(x);
						Vector dfcix = dfc(i)(x);

						sum += 1.0 / fcix * (1.0 / fcix * (dfcix % dfcix)(y) - d2fc(i)(x)(y));
					}

					return sum;
				};
		}

		/// <summary>
		/// Search for approximate minimum of a scalar function
		/// accross a direction using backtracking.
		/// </summary>
		/// <param name="f">The scalar function to minimize.</param>
		/// <param name="df">The derivative of the scalar function to minimize.</param>
		/// <param name="x">The point where the search starts.</param>
		/// <param name="Δx">The direction along which the search takes place.</param>
		/// <param name="α">Slope ease factor. Must be in (0, 1/2).</param>
		/// <param name="β">Backtracking retreat factor. Must be in (0, 1).</param>
		/// <returns>
		/// Returns the factor which scales <paramref name="Δx"/> to point
		/// to the approximate minimum.
		/// </returns>
		/// <remarks>
		/// See lecture notes of Convex Optimization I class, slide 10-6, Boyd, Stanford University.
		/// </remarks>
		private static double BacktrackMinSearch(
			ScalarFunction f,
			VectorFunction df,
			Vector x,
			Vector Δx,
			double α,
			double β)
		{
			if (f == null) throw new ArgumentNullException("f");
			if (df == null) throw new ArgumentNullException("f");
			if (x == null) throw new ArgumentNullException("x");
			if (Δx == null) throw new ArgumentNullException("Δx");
			if (α <= 0.0 || α >= 0.5)
				throw new ArgumentException("α must be in (0, 1/2).", "α");
			if (β <= 0.0 || β >= 1.0)
				throw new ArgumentException("β must be in (0, 1).", "β");

			var fx = f(x);
			var αdfx = α * (df(x) * Δx);

			double t;

			for (t = 1.0; f(x + t * Δx) >= fx + t * αdfx; t *= β) ;

			return t;
		}

		private static double ConvexMinimize(
			ScalarFunction f, 
			Vector x, 
			Vector Δx, 
			LineSearchMinimizeOptions options,
			Func<Vector, bool> outOfDomainIndicator,
			out double fβ)
		{
			double f1 = f(x);

			return ConvexMinimize(f1, f, x, Δx, options, outOfDomainIndicator, out fβ);
		}

		private static double ConvexMinimize(
			double f1,
			ScalarFunction f,
			Vector x,
			Vector Δx,
			LineSearchMinimizeOptions options,
			Func<Vector, bool> outOfDomainIndicator,
			out double fβ)
		{
			double f2 = f1;
			double f3;

			double β1 = 0.0;

			double β2 = 0.0, β3 = 0.0;

			for (double Δβ = 1.0; ; )
			{
				if (Δβ * Math.Sqrt(Δx.Norm2 / Δx.Length) < options.MinLineSearchLength)
				{
					fβ = f2;
					return β2;
				}

				// Point of search.
				Vector xs = x + (β2 + Δβ) * Δx;

				// Take the quick domain test first. Are we off the domain of the function? 
				if (outOfDomainIndicator(xs))
				{
					// If yes, search closer.
					Δβ /= 4.0;
					continue;
				}

				// Evaluate the function fully now at the search point.
				fβ = f(xs);

				// Are we off the domain of the function?
				if (Double.IsNaN(fβ) || Double.IsPositiveInfinity(fβ))
				{
					// If yes, search closer.
					Δβ /= 4.0;
					continue;
				}

				// Is the value found greater than the one of the last mid-point?
				if (fβ >= f2)
				{
					// And, did we find at least one descending mid-point?
					if (β2 == 0.0)
					{
						// If not, we should first find a descending mid-point.
						// Δx is a descending direction, so we ought to search closer.
						// If we search close enough, we will surely find one.
						Δβ /= 4.0;

						continue;
					}
					else
					{
						// We have our goal: β1 < β2 < β3 = β and f1 > f2 < f3 = f.
						β3 = β2 + Δβ;
						f3 = fβ;
						break;
					}
				}
				// Is the current value less than the left value?
				else
				{
					// Else, this is the current mid-point β2 and the previous 
					// midpoint becomes left point β1.
					β1 = β2;
					f1 = f2;
					β2 = β2 + Δβ;
					f2 = fβ;

					Δβ *= 10;
				}
			}

			return ConvexMinimize(
				β => f(x + β * Δx),
				β1, β2, β3,
				f1, f2, f3,
				Δx,
				options,
				out fβ);
		}

		private static double ConvexMinimize(
			Func<double, double> f,
			double β1, double β2, double β3,
			double f1, double f2, double f3,
			Vector Δx,
			LineSearchMinimizeOptions options,
			out double fβ)
		{
			int iterationsCount = 0;

			do
			{
				// This is the determinant of the Vandermonde matrix implied by βi's.
				double detB = (β2 - β1) * (β3 - β2) * (β3 - β1);

				// Coefficient c1 of quadratic regression.
				double c1 =
					(
						f2 * β3 * β3 + f1 * β2 * β2 + f3 * β1 * β1
						- f2 * β1 * β1 - f3 * β2 * β2 - f1 * β3 * β3
					) / detB;

				// Coefficient c2 of quadratic regression.
				double c2 =
					(
						β2 * f3 + β1 * f2 + β3 * f1 - β2 * f1 - β3 * f2 - β1 * f3
					) / detB;

				// Compute minimum by setting the polynomial's derivative to zero.
				double β = -c1 / (2 * c2);

				fβ = f(β);

				if (β < β2 && fβ < f2)
				{
					β3 = β2; f3 = f2;
					β2 = β; f2 = fβ;
				}
				else if (β > β2 && fβ < f2)
				{
					β1 = β2; f1 = f2;
					β2 = β; f2 = fβ;
				}
				else if (β < β2)
				{
					β1 = β; f1 = fβ;
				}
				else if (β > β2)
				{
					β3 = β; f3 = fβ;
				}
				else return β2;

			}
			while (
				Math.Min(β2 - β1, β3 - β2) > options.MinLineSearchLength &&
				Math.Min(f1 - f2, f3 - f2) > options.LineSearchThreshold &&
				++iterationsCount < options.MaxLineSearchIterations
				);

			return β2;
		}

		#endregion
	}
}
