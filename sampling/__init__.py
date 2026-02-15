"""Sampling-based marginalization approaches for belief state computation.

As games scale beyond Kuhn poker, exact marginalization over the full joint
distribution of deals becomes intractable. This module provides sampling-based
alternatives that approximate marginal beliefs with controllable accuracy.

Three approaches:

1. ProjectionSampling — Projects the joint onto per-player marginals, then
   reconstructs an approximate joint by sampling deals proportional to the
   product of marginal probabilities, subject to compatibility constraints.

2. GibbsSampling — Uses Gibbs sampling to draw from the joint distribution
   by iteratively sampling each player's hand conditioned on all others.

3. ReachGuidedSparse — Uses reach probabilities to identify and prune
   low-probability deals, maintaining a sparse representation of the joint
   that focuses compute on the mass of the distribution.
"""
