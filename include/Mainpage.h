/**
 * @file Mainpage.h
 * @brief Main page Doxygen documentation for CMP++.
 */

/**
 * @mainpage CMP++: Scientific Uncertainty Quantification and Bayesian Calibration Library
 *
 * @section intro_sec Introduction
 * Welcome to the official API documentation for **CMP++**, a high-performance, template-driven C++ library specifically engineered for **Uncertainty Quantification (UQ)**, **Bayesian Calibration**, and **Machine Learning** in complex scientific computing environments.
 *
 * @section overview_sec Architectural & Mathematical Overview
 * CMP++ solves the classical Bayesian calibration formulation for computer models:
 * \f[
 * y(\mathbf{x}) = \eta(\mathbf{x}, \boldsymbol{\theta}) + \delta(\mathbf{x}) + \varepsilon
 * \f]
 * where:
 * - \f$\eta(\mathbf{x}, \boldsymbol{\theta})\f$ is the deterministic computer simulator parameterized by calibration inputs \f$\boldsymbol{\theta}\f$.
 * - \f$\delta(\mathbf{x})\f$ is the systematic model discrepancy, represented non-parametrically as a Gaussian Process: \f$\delta(\mathbf{x}) \sim \mathcal{GP}(0, k(\mathbf{x}, \mathbf{x}'))\f$.
 * - \f$\varepsilon \sim \mathcal{N}(0, \sigma^2)\f$ is the experimental observation noise.
 *
 * @section domain_mapping_sec Mathematical Domains & UQ Methods Mapping
 *
 * | UQ Domain | Core Class | Mathematical Representation / Description | Reference Test |
 * | :--- | :--- | :--- | :--- |
 * | **Gaussian Process Regression** | @ref cmp::gp::GaussianProcess "GaussianProcess" | Non-parametric Bayesian regression: \f$ f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')) \f$ | `tests/gp.cpp` |
 * | **Multi-Output GP (PCA-reduced)** | @ref cmp::gp::MultiOutputGaussianProcess "MultiOutputGaussianProcess" | Multi-output surrogate mapping via PCA dimensionality reduction | `tests/multi_gp.cpp` |
 * | **Polynomial Chaos Expansion** | @ref cmp::PolynomialExpansion "PolynomialExpansion" | Spectral projection of stochastic outputs: \f$ Y \approx \sum_{j=0}^{P-1} c_j \Psi_j(\boldsymbol{\xi}) \f$ | `tests/poly.cpp` |
 * | **Global Sensitivity Analysis** | @ref cmp::sobol::SobolSaltelli "SobolSaltelli" | Variance decomposition: \f$ S_i = \frac{V_i}{V(Y)} \f$ and \f$ S_{Ti} \f$ via Saltelli grids | `tests/sobol.cpp` |
 * | **Markov Chain Monte Carlo** | @ref cmp::mcmc::MarkovChain "MarkovChain" | Metropolis-Hastings with DRAM (Delayed Rejection Adaptive Metropolis) | `tests/mcmc.cpp` |
 * | **Evolutionary MCMC** | @ref cmp::mcmc::EvolutionaryMarkovChain "EvolutionaryMarkovChain" | Multi-chain sampling with crossover and parallel tempering | `tests/emcmc.cpp` |
 * | **Kernel Density Estimation** | @ref cmp::classifier::KDE "KDE" | Non-parametric density estimation: \f$ \hat{f}_h(\mathbf{x}) = \frac{1}{nh^d} \sum K\left(\frac{\mathbf{x}-\mathbf{x}_i}{h}\right) \f$ | `tests/kde.cpp` |
 * | **Support Vector Machines** | @ref cmp::classifier::SVM "SVM" | Hyperparameter-tuned RBF classification with span-based bounds | `tests/svm.cpp` |
 *
 * @section quickstart_sec Quick-Start & Core Headers
 * - **Surrogate Modules**:
 *   - [gp.h](file:///Users/omarkahol/opt/CMP++/include/gp.h) - Main Gaussian Process header.
 *   - [multi_gp.h](file:///Users/omarkahol/opt/CMP++/include/multi_gp.h) - PCA Multi-output GP header.
 *   - [poly.h](file:///Users/omarkahol/opt/CMP++/include/poly.h) - PCE expansions and orthogonal polynomials.
 * - **Sampling & Inference**:
 *   - [mcmc.h](file:///Users/omarkahol/opt/CMP++/include/mcmc.h) - DRAM and adaptive MCMC chain sampling.
 *   - [distribution.h](file:///Users/omarkahol/opt/CMP++/include/distribution.h) - Prior, normal, and mixture distributions.
 *   - [hmc.h](file:///Users/omarkahol/opt/CMP++/include/hmc.h) - Hamiltonian Monte Carlo sampler.
 * - **Sensitivity & Classifiers**:
 *   - [sobol.h](file:///Users/omarkahol/opt/CMP++/include/sobol.h) - Sobol sensitivity grid constructor and indices estimator.
 *   - [classifier.h](file:///Users/omarkahol/opt/CMP++/include/classifier.h) - SVM, KDE, and Naive Bayes classifiers.
 *
 * @section verification_sec Verification & Testing Suite
 * Access these files in `/tests` to see exact usage examples:
 * - `tests/distribution.cpp`: Bivariate sampling and Sliced-Wasserstein verification.
 * - `tests/mcmc.cpp`: DRAM sampling on bimodal targets.
 * - `tests/gp.cpp`: GP training and cross-validation verification.
 * - `tests/multi_gp.cpp`: Dimensionality reduction using PCA followed by multi-output GP fitting.
 * - `tests/sobol.cpp`: Sobol sensitivity analysis on the Ishigami function.
 * - `tests/calibration.cpp`: Full model calibration with discrepancy GP.
 */
