using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Gramma.Vectors;
using Gramma.Optimization.DecayFunctions;
using System.Threading.Tasks;
using System.Threading;
using System.Diagnostics;
using System.Collections.Concurrent;

namespace Gramma.Optimization
{
	/// <summary>
	/// Implements a parallel version of the stochastic gradient descent algorithm.
	/// </summary>
	public static class StochasticGradientDescent
	{
		#region Options for optimization

		/// <summary>
		/// Options for the optimization.
		/// </summary>
		[Serializable]
		public class Options
		{
			#region Private fields

			private DecayFunction decayFunction = new LinearDecayFunction();

			private int maxIterationsCount = 1000;

			private double stepSizeCoefficient = 1.0;

			#endregion

			#region Public properties

			/// <summary>
			/// The decay function to use for update steps. Default is <see cref="LinearDecayFunction"/>.
			/// </summary>
			public DecayFunction DecayFunction
			{
				get
				{
					return decayFunction;
				}
				set
				{
					if (value == null) throw new ArgumentNullException("value");
					decayFunction = value;
				}
			}

			/// <summary>
			/// The maximum number of iterations to perform. Default is 1000.
			/// </summary>
			public int MaxIterationsCount
			{
				get { return maxIterationsCount; }
				set { maxIterationsCount = value; }
			}

			/// <summary>
			/// The update step size coeficient.
			/// It is multiplied with the outcome of <see cref="DecayFunction"/> to determine the step size.
			/// Default is 1.0.
			/// </summary>
			public double StepSizeCoefficient
			{
				get
				{
					return stepSizeCoefficient;
				}
				set
				{
					if (value <= 0.0) throw new ArgumentException("The value must be positive.");
					stepSizeCoefficient = value;
				}
			}

			#endregion
		}

		#endregion

		#region Optimization methods

		/// <summary>
		/// Minimize using stochastic gradient descent.
		/// </summary>
		/// <typeparam name="T">The type of each sample.</typeparam>
		/// <typeparam name="V">The type of vector, must be at least <see cref="IVector"/>.</typeparam>
		/// <param name="samples">The collection of samples.</param>
		/// <param name="gradientFunction">
		/// The sample gradient function.
		/// Its two arguments are the sample, the model parameters, 
		/// and it returns the gradient with respect to the model parameters.
		/// </param>
		/// <param name="initialVector">The initial value of model parameters.</param>
		/// <param name="options">The optimization options.</param>
		/// <param name="cancellationToken">A token used to cause premature exit.</param>
		/// <param name="degreeOfParallelism">
		/// The degree of parallelism. If zero or more than the number of processors, 
		/// it is set to number of processors.
		/// </param>
		/// <returns>
		/// Returns the optimized model parameters after the maximum number of iterations specified in <paramref name="options"/>
		/// or the end of <paramref name="samples"/> collection, whichever comes first.
		/// </returns>
		/// <remarks>
		/// It is typical to have an infinite sequence in <paramref name="samples"/> to represent streaming on-line data.
		/// When having off line data loaded in an array, the infinite sequence
		/// can be produced by infinite random picking of array items as provided
		/// by the Gramma.Linq project, using CollectionExtensions.RandomPickSequence extension methods.
		/// </remarks>
		public static V ParallelMinimize<T, V>(
			IEnumerable<T> samples,
			Func<T, V, V> gradientFunction, 
			V initialVector, 
			Options options,
			CancellationToken cancellationToken,
			int degreeOfParallelism = 0)
			where V : IVector
		{
			if (samples == null) throw new ArgumentNullException("samples");
			if (gradientFunction == null) throw new ArgumentNullException("gradientFunction");
			if (initialVector == null) throw new ArgumentNullException("initialVector");
			if (options == null) throw new ArgumentNullException("options");
			
			if (degreeOfParallelism <= 0 || degreeOfParallelism > Environment.ProcessorCount) 
				degreeOfParallelism = Environment.ProcessorCount;

			var xLock = new object();

			int sampleCount = 0;

			ParallelOptions parallelOptions = new ParallelOptions();

			parallelOptions.MaxDegreeOfParallelism = degreeOfParallelism;

			var partitioner = Partitioner.Create(samples);

			var x = initialVector;

			var decayFunction = options.DecayFunction;

			double stepSizeCoefficient = options.StepSizeCoefficient;

			Parallel.ForEach(
				partitioner,
				parallelOptions,
				delegate(T sample, ParallelLoopState state)
				{
					int currentSampleIndex = Interlocked.Increment(ref sampleCount);

					if (currentSampleIndex > options.MaxIterationsCount || cancellationToken.IsCancellationRequested)
					{
						state.Break();
						return;
					}

					Trace.WriteLine(String.Format("Computing stochastic gradient sample #{0}.", currentSampleIndex));

					double β = 
						stepSizeCoefficient * decayFunction.Evaluate(currentSampleIndex) / degreeOfParallelism;

					var xCurrent = x;

					var g = gradientFunction(sample, xCurrent);

					var Δx = g.ScaleInPlace(-β);

					lock (xLock)
					{
						x.AddInPlace(Δx);
					}
				}
			);

			return x;
		}

				/// <summary>
		/// Minimize using stochastic gradient descent.
		/// </summary>
		/// <typeparam name="T">The type of each sample.</typeparam>
		/// <typeparam name="V">The type of vector, must be at least <see cref="IVector"/>.</typeparam>
		/// <param name="samples">The collection of samples.</param>
		/// <param name="gradientFunction">
		/// The sample gradient function.
		/// Its two arguments are the sample, the model parameters, 
		/// and it returns the gradient with respect to the model parameters.
		/// </param>
		/// <param name="initialVector">The initial value of model parameters.</param>
		/// <param name="options">The optimization options.</param>
		/// <param name="degreeOfParallelism">
		/// The degree of parallelism. If zero or more than the number of processors, 
		/// it is set to number of processors.
		/// </param>
		/// <returns>
		/// Returns the optimized model parameters after the maximum number of iterations specified in <paramref name="options"/>
		/// or the end of <paramref name="samples"/> collection, whichever comes first.
		/// </returns>
		/// <remarks>
		/// It is typical to have an infinite sequence in <paramref name="samples"/> to represent streaming on-line data.
		/// When having off line data loaded in an array, the infinite sequence
		/// can be produced by infinite random picking of array items as provided
		/// by the Gramma.Linq project, using CollectionExtensions.RandomPickSequence extension methods.
		/// </remarks>
		public static V ParallelMinimize<T, V>(
			IEnumerable<T> samples,
			Func<T, V, V> gradientFunction,
			V initialVector,
			Options options,
			int degreeOfParallelism = 0)
			where V : IVector
		{
			return ParallelMinimize(samples, gradientFunction, initialVector, options, new CancellationToken(false), degreeOfParallelism);
		}

		#endregion
	}
}
