# Grammophone.Optimization
This .NET library provides solving linear systems using Krylov sequences and minimizing convex functions, possibly with convex inequality constraints, using conjugate gradients. It contains methods that use Hessians as well as corresponding methods that avoid Hessians altogether.

The inputs of the modules are user-supplied functions. These specify the objective function and the constraints. Matrices are never given explicitly, but they are also abstracted as a special type of function, a Tensor, which takes a vector and yields another vector. This Tensor function type forms a kind of an opaque matrix oracle that is expected to apply a linear transformation on its input. For example, preconditioners, Hessians (if the optimization method variant requires them) are built around Tensors, specifically they are functions returning Tensors, permitting implementation abstractions such as computation parallelization, remote data access or very large dimensionality where conventional matrix storage schemes would fail to fit in memory.

The library depends on [Grammophone.Vectors](https://github.com/grammophone/Grammophone.Vectors) which should be located in a sibling directory.
