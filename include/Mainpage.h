/**
 * @file Mainpage.h
 * @brief Main page Doxygen documentation for CMP++.
 */

/**
 * @mainpage CMP++: Scientific Uncertainty Quantification and Bayesian Calibration Library
 *
 * \image html logo.png "" width=550px
 *
 * @section intro_sec Introduction
 * Welcome to the official API documentation for **CMP++**, a high-performance, template-driven C++ library specifically engineered for **Uncertainty Quantification (UQ)**, **Bayesian Calibration**, and **Machine Learning** in complex scientific computing environments.
 *
 * @section overview_sec Architectural & Scientific Overview
 * Rather than solving a single model calibration equation, CMP++ implements a cohesive, modular ecosystem of algorithms that connect experimental data, high-fidelity simulators, statistical surrogate models, and global sensitivity evaluations:
 *
 * 1. **Bayesian Calibration with Discrepancy Modeling**:
 *    Solves the general computer model calibration problem by integrating physical simulator runs with experimental observations. It accounts for simulator inaccuracies using non-parametric discrepancy modeling. The discrepancy function is represented as a Gaussian Process that learns systemic errors directly from the residuals of physical simulator predictions.
 *
 * 2. **Surrogate Modeling Pipeline**:
 *    To accelerate computationally expensive physical model evaluations, CMP++ constructs either global or localized statistical surrogate models:
 *    - **Gaussian Processes (Kriging)**: Approximates smooth physical responses with robust posterior confidence bounds and statistical variance envelopes.
 *    - **Polynomial Chaos Expansion (PCE)**: Projects stochastic responses onto orthogonal polynomial bases (Hermite, Legendre, Chebyshev, Laguerre) using regression or spectral quadrature.
 *    - **Localized Mixtures (Model Clustering)**: Partitions the design space using K-Means or Dirichlet Process Mixture Models (DPMM) and blends local GP/PCE models through probabilistic weights.
 *
 * 3. **Global Sensitivity Analysis**:
 *    Decomposes total simulator variance into main and joint parameter interactions. CMP++ calculates Saltelli-Sobol sensitivity indices to determine which input parameters dominate output variance and which parameter interactions are negligible.
 *
 * 4. **MCMC & HMC/NUTS Bayesian Inference**:
 *    Performs posterior parameter sampling using adaptive step sizing, delayed rejection, and Hamiltonian dynamics. It offers both classical Metropolis-Hastings (DRAM) and state-of-the-art Hamiltonian Monte Carlo using the No-U-Turn Sampler (NUTS) with dual-averaging adaptation.
 *

 * @section categories_sec Architectural Components & Component Mapping
 *
 * @subsection surrogate_mod Surrogate Modeling & Dimensionality Reduction
 * - **Gaussian Process Regression**: @ref cmp::gp::GaussianProcess "GaussianProcess" - Non-parametric Bayesian regression modeling that provides predictive distributions, prior mean functions, and kernel covariances.
 * - **Multi-Output GP**: @ref cmp::gp::MultiOutputGaussianProcess "MultiOutputGaussianProcess" - Independent Gaussian Processes designed for multi-column output dimensions.
 * - **Polynomial Chaos Expansion**: @ref cmp::PolynomialExpansion "PolynomialExpansion" - Spectral projection representing stochastic models via orthonormal polynomials.
 * - **Clustered Local GP**: @ref cmp::ModelCluster "ModelCluster" - Localized Gaussian Process surrogates blended via classifier posterior probabilities.
 * - **Clustered Local PCE**: @ref cmp::ModelClusterPoly "ModelClusterPoly" - Localized Polynomial Chaos Expansions blended via classifier posterior probabilities.
 * - **Principal Component Analysis**: @ref cmp::scaler::PCA "PCA" - Covariance eigen-decomposition to map high-dimensional datasets to lower-dimensional latent spaces.
 *
 * @subsection sensitivity_anal Sensitivity Analysis
 * - **Saltelli Sobol Indices**: @ref cmp::sobol::SobolSaltelli "SobolSaltelli" - Variance decomposition scheme to evaluate first, second, and total order Sobol sensitivity indices.
 *
 * @subsection sampling_inf Sampling & Inference Methods
 * - **Markov Chain Monte Carlo**: @ref cmp::mcmc::MarkovChain "MarkovChain" - Metropolis-Hastings chain simulation implementing Delayed Rejection Adaptive Metropolis (DRAM).
 * - **Evolutionary MCMC**: @ref cmp::mcmc::EvolutionaryMarkovChain "EvolutionaryMarkovChain" - Multi-chain MCMC utilizing differential evolution crossover and mutations.
 * - **Hamiltonian Monte Carlo**: @ref cmp::mcmc::HamiltonianMarkovChain "HamiltonianMarkovChain" - HMC sampler utilizing the No-U-Turn Sampler (NUTS) algorithm and dual-averaging step-size adaptation.
 *
 * @subsection probability_dist Probability Distributions, Priors, & Kernels
 * - **CRTP Univariate Distribution**: @ref cmp::distribution::UnivariateDistribution "UnivariateDistribution" - CRTP base class representing continuous 1D probability density functions.
 * - **CRTP Multivariate Distribution**: @ref cmp::distribution::MultivariateDistribution "MultivariateDistribution" - CRTP base class representing joint continuous multi-dimensional probability density functions.
 * - **Prior Distributions**: @ref cmp::prior::Prior "Prior" - Interface for prior probability densities supporting joint product, uniform, and distribution-mapped priors.
 * - **Covariance Kernels**: @ref cmp::covariance::Covariance "Covariance" - Interface for GP kernels (SquaredExponential, Matern52, Matern, WhiteNoise, Sum, Product).
 * - **GP Mean Functions**: @ref cmp::mean::Mean "Mean" - Interface for GP prior mean functions (Constant, Zero).
 * - **Density Kernels**: @ref cmp::kernel::Kernel "Kernel" - Evaluates local smoothing kernels (Gaussian, Epanechnikov, Uniform) for density estimations.
 * - **Density Bandwidths**: @ref cmp::kernel::Bandwidth "Bandwidth" - Computes matrix transformations (Isotropic, Diagonal, Full) scaling multivariate distances in KDEs.
 *
 * @subsection classifiers_sec Classifiers
 * - **Classifier Base**: @ref cmp::classifier::Classifier "Classifier" - Common interface defining discrete classification and class probability estimation.
 * - **Support Vector Classifier**: @ref cmp::classifier::SVM "SVM" - Classifier constructing optimal separating hyperplanes via LIBSVM with custom kernels.
 * - **Kernel Density Classifier**: @ref cmp::classifier::KDE "KDE" - Non-parametric Bayes classifier estimating class densities using Kernel Density Estimation.
 * - **K-Nearest Neighbors**: @ref cmp::classifier::KNN "KNN" - Neighborhood helper model identifying adjacent points in design space.
 * - **Baseline Classifier**: @ref cmp::classifier::Dummy "Dummy" - Simple classifier predicting classes based on empirical training frequencies.
 *
 * @subsection clustering_sec Clustering Methods
 * - **Dirichlet Process Mixture Model**: @ref cmp::cluster::DirichletProcessMixtureModel "DirichletProcessMixtureModel" - Infinite Gaussian Mixture Model clustering using collapsed Gibbs sampling.
 * - **Geometric Clustering**: @ref cmp::cluster::GeometricCluster "GeometricCluster" - K-means clustering partitioning sample points to minimize within-cluster sum of squares.
 * - **Uniform Random Clustering**: @ref cmp::cluster::DummyCluster "DummyCluster" - Baseline clusterer assigning points to partitions uniformly at random.
 *
 * @subsection core_util Core Utilities & Integration
 * - **1D Quadrature**: @ref cmp::Quadrature1D "Quadrature1D" - Generates quadrature weights and nodes using the Golub-Welsch symmetric tridiagonal eigenvalue algorithm.
 * - **Tensor Integration**: @ref cmp::TensorIntegrator "TensorIntegrator" - Multi-dimensional integration using tensor-product quadrature rules mapped to canonical/physical domains.
 * - **K-Fold CV Partitioning**: @ref cmp::statistics::KFold "KFold" - Partitions datasets into training/testing folds for parameter cross-validation.
 * - **Bootstrap Resampling**: @ref cmp::statistics::Bootstrap "Bootstrap" - Resamples datasets with or without replacement to evaluate bootstrap estimator variance.
 * - **Wasserstein Distances**: @ref cmp::wasserstein1D "wasserstein1D" and @ref cmp::slicedWassersteinDistance "slicedWassersteinDistance" - Evaluates optimal transport distances between distributions.
 * - **Finite Differences**: @ref cmp::fd_gradient "fd_gradient" and @ref cmp::fd_hessian "fd_hessian" - Central difference stencils to approximate objective function gradients and Hessians.
 * - **IO Serialization**: @ref cmp::read_vector "read_vector" and @ref cmp::write_vector "write_vector" - Text file streaming and CSV data reading/writing utilities.
 *
 * @section quickstart_sec File Organization & Source Headers
 * - **Surrogates**:
 *   - @ref gp.h "gp.h" - Main Gaussian Process regression.
 *   - @ref multi_gp.h "multi_gp.h" - Independent multi-output GPs.
 *   - @ref poly.h "poly.h" - PCE expansion surrogates.
 *   - @ref model_cluster.h "model_cluster.h" - Clustered local GP mixtures.
 *   - @ref model_cluster_poly.h "model_cluster_poly.h" - Clustered local PCE mixtures.
 * - **Samplers & Optimization**:
 *   - @ref mcmc.h "mcmc.h" - DRAM and Evolutionary MCMC.
 *   - @ref hmc.h "hmc.h" - Hamiltonian Monte Carlo (NUTS).
 *   - @ref optimization.h "optimization.h" - NLopt optimization wrappers.
 * - **Probability & Math**:
 *   - @ref distribution.h "distribution.h" - Probability distributions.
 *   - @ref prior++.h "prior++.h" - Joint prior distributions.
 *   - @ref covariance.h "covariance.h" - Kernel covariance functions.
 *   - @ref mean++.h "mean++.h" - GP prior mean functions.
 *   - @ref kernel++.h "kernel++.h" - KDE kernels and bandwidths.
 *   - @ref kde.h "kde.h" - Kernel Density Estimation models.
 *   - @ref integrator.h "integrator.h" - Quadrature rules and tensor integration.
 * - **Sensitivity, Statistics & Utilities**:
 *   - @ref sobol.h "sobol.h" - Saltelli-Sobol sensitivity index calculator.
 *   - @ref statistics.h "statistics.h" - K-Fold, Bootstrap, and correlation.
 *   - @ref wasserstein.h "wasserstein.h" - Sliced Wasserstein distances.
 *   - @ref finite_diff.h "finite_diff.h" - Numerical gradient and Hessian stencils.
 *   - @ref classifier.h "classifier.h" - KDE, SVM, and Dummy classifiers.
 *   - @ref cluster.h "cluster.h" - K-means and Dirichlet Process Mixture Model clustering.
 *   - @ref io.h "io.h" - File read/write serialization helpers.
 *   - @ref cmp_defines.h "cmp_defines.h" - Global templates and LDLT utility functions.
 *
 * @section verification_sec Verification & Testing Suite
 * Access these files in `/tests` to see exact usage examples:
 * - `tests/distribution.cpp`: Bivariate sampling and Sliced-Wasserstein verification.
 * - `tests/mcmc.cpp`: DRAM sampling on bimodal targets.
 * - `tests/gp.cpp`: GP training and cross-validation verification.
 * - `tests/multi_gp.cpp`: Dimensionality reduction using PCA followed by multi-output GP fitting.
 * - `tests/sobol.cpp`: Sobol sensitivity analysis on the Ishigami function.
 * - `tests/calibration.cpp`: Full model calibration with discrepancy GP.
 *
 * @section eu_funding Funding & Acknowledgement
 * \htmlonly
 * <div class="eu-acknowledgement">
 *   <img class="eu-acknowledgement-logo-img" src="eu_logo.png" alt="European Union Logo" />
 *   <div class="eu-acknowledgement-text">
 *     This project has received funding from the European Union’s Horizon Europe research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101072551 (TRACES).
 *   </div>
 * </div>
 * \endhtmlonly
 */

/**
 * @defgroup surrogate Surrogate Modeling & Dimensionality Reduction
 * @brief Kriging (Gaussian Processes), Polynomial Chaos Expansion, Clustered Surrogates, and PCA.
 *
 * @defgroup sensitivity Sensitivity Analysis
 * @brief Global sensitivity analysis and Saltelli-Sobol variance decomposition.
 *
 * @defgroup sampling Sampling & Inference Methods
 * @brief MCMC, Evolutionary MCMC, and NUTS/Hamiltonian Monte Carlo.
 *
 * @defgroup probability Probability Distributions, Priors, & Kernels
 * @brief Continuous probability distributions, covariance kernels, GP mean functions, prior probability definitions, and KDE.
 *
 * @defgroup classifiers Classifiers
 * @brief Class probability models, Support Vector Machines, KDE, and KNN.
 *
 * @defgroup clustering Clustering Methods
 * @brief Dirichlet Process Mixture Models (DPMM), geometric K-Means, and baseline clusterers.
 *
 * @defgroup core Core Utilities & Integration
 * @brief Numeric quadrature, tensor integration, cross-validation, bootstrap resampling, Sliced Wasserstein distances, numerical finite differences, and serialization.
 */
