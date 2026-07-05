#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <cmp_defines.h>
#include <list>
#include <kernel++.h>
#include <svm.h>
#include <statistics.h>
#include <optimization.h>
#include <kde.h>
#include <covariance.h>

/**
 * @addtogroup classifiers
 * @{
 */
namespace cmp::classifier {

/**
 * @brief Bandwidth selection objective criterion.
 */
enum method {
    CV_SCORE,      ///< Cross-validation score optimization based on classification accuracy
    CV_PROB_SCORE  ///< Cross-validation score optimization based on probability likelihood
};



/**
 * @brief Abstract base class for all classifiers.
 * 
 * @details Mathematical Formulation
 * A classifier maps an input vector \f$\mathbf{x} \in \mathbb{R}^D\f$ to a class label \f$y \in \{0, 1, \dots, C-1\}\f$,
 * or estimates the class posterior probabilities \f$P(y = c \mid \mathbf{x})\f$ for \f$c = 0, \dots, C-1\f$.
 * 
 * @details Implementation Algorithm
 * Provides a pure virtual interface defining:
 * - `predictProbabilities(x)`: Returns a vector of probability estimates.
 * - `predict(x)`: Returns the class label with the highest probability.
 */
class Classifier {
  public:
    virtual ~Classifier() = default;
    virtual std::vector<double> predictProbabilities(const Eigen::Ref<const Eigen::VectorXd> &x) const = 0;
    virtual size_t predict(const Eigen::Ref<const Eigen::VectorXd> &x) const = 0;
};

/**
 * @brief K-Nearest Neighbors classifier helper class.
 * 
 * @details Mathematical Formulation
 * For a query point \f$\mathbf{x}\f$, K-Nearest Neighbors identifies the set of \f$k\f$ closest points 
 * \f$\mathcal{N}_k(\mathbf{x})\f$ in the reference dataset \f$\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N\f$ under a chosen distance metric \f$d(\mathbf{x}, \mathbf{x}_i)\f$ (typically Euclidean distance):
 * \f[ d(\mathbf{x}, \mathbf{x}_i) = \|\mathbf{x} - \mathbf{x}_i\|_2 \f]
 * 
 * @details Implementation Algorithm
 * The `compute` method builds an indexing/neighbour structure. For each point in the dataset,
 * it searches and stores the indices of its \f$k\f$ nearest neighbors.
 */
class KNN {
  private:
    std::vector<std::vector<size_t>> neighbours_; ///< Cache containing nearest neighbor indices for each data point.
    size_t kNearestValue_;                        ///< The number of nearest neighbors (k).
    size_t nPoints_;                              ///< Total number of data points.

  public:
    /**
     * @brief Computes the nearest neighbor index map for the given points.
     * 
     * @param points Vector of data points.
     * @param k Number of nearest neighbors to compute.
     */
    void compute(const std::vector<Eigen::VectorXd> &points, size_t k);

    /**
     * @brief Accesses the list of nearest neighbor indices for the i-th point.
     * 
     * @param i Index of the query point.
     * @return Reference to a vector of neighbor indices.
     */
    const std::vector<size_t> &operator[](size_t i) const;

    /**
     * @brief Returns the total number of points in the dataset.
     * 
     * @return Size of the dataset.
     */
    size_t nPoints() const;

    /**
     * @brief Returns the value of k (number of neighbors).
     * 
     * @return Number of neighbors.
     */
    size_t k() const;
};



/**
 * @brief Classifier that uses Kernel Density Estimation (KDE) for classification.
 * 
 * @details Mathematical Formulation
 * Estimates the conditional probability density function for class \f$c\f$ at point \f$\mathbf{x} \in \mathbb{R}^D\f$ using kernel density estimation:
 * \f[
 * p(\mathbf{x} \mid y = c) = \frac{1}{N_c} \sum_{i \in \mathcal{D}_c} \prod_{d=1}^D \frac{1}{h_{c,d}} K\left(\frac{x_d - x_{i,d}}{h_{c,d}}\right)
 * \f]
 * where \f$N_c\f$ is the number of samples in class \f$c\f$, \f$h_{c,d}\f$ is the bandwidth for dimension \f$d\f$ in class \f$c\f$, and \f$K\f$ is a univariate kernel.
 * The class posterior probability is obtained using Bayes' rule:
 * \f[
 * P(y = c \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid y = c) P(y = c)}{\sum_{j=0}^{C-1} p(\mathbf{x} \mid y = j) P(y = j)}
 * \f]
 * where \f$P(y = c) = \frac{N_c}{N}\f$ represents the empirical prior probability of class \f$c\f$.
 * 
 * @details Implementation Algorithm
 * 1. `condition()` partitions training samples by class label and computes class priors \f$P(y=c)\f$.
 * 2. `predictProbabilities()` computes the kernel density \f$p(\mathbf{x} \mid y = c)\f$ for each class, computes the product of density and prior, and normalizes the results.
 * 3. `fit()` optimizes the bandwidth vector \f$\mathbf{h}\f$ by maximizing K-fold cross-validation likelihood or leave-one-out (LOO) log-probability using NLopt optimization routines.
 */
class KDE : public Classifier {
  private:
    Eigen::MatrixXd xObs_ = Eigen::MatrixXd(0, 0); /// Observations
    Eigen::VectorXs labels_ = Eigen::VectorXs(0);  /// Labels for the observations
    Eigen::VectorXs classCounts_;               /// Count of observations per class

    size_t nObs_ = 0;      ///< Number of training observations.
    size_t dimX_ = 0;      ///< Input features dimensionality.
    size_t nClasses_ = 1;  ///< Total number of distinct class labels.

    std::shared_ptr<cmp::kernel::Bandwidth> bandwidth_{nullptr}; /// Bandwidth object for KDE
    std::shared_ptr<kernel::Kernel> kernel_{nullptr};            /// Kernel object for KDE

  public:
    KDE() {};


    /**
     * @brief Set the kernel and bandwidth for the KDE.
     * This method allows you to set the kernel and bandwidth objects that will be used for the KDE.
     * @param kernel A shared pointer to a kernel object that defines the kernel function.
     * @param bandwidth A shared pointer to a bandwidth object that defines the bandwidth for the kernel.
     */
    void set(std::shared_ptr<kernel::Kernel> kernel, std::shared_ptr<cmp::kernel::Bandwidth> bandwidth) {
        kernel_ = kernel;
        bandwidth_ = bandwidth;
    };


    /**
     * @brief Condition the KDE with observations and labels.
     * This method sets the observations and labels for the KDE.
     * It initializes the number of observations, dimensions, and classes based on the input data.
     * @param xObs A matrix of observations where each row is an observation and each column is a feature.
     * @param labels A vector of labels corresponding to the observations.
     * @throws std::runtime_error if the number of observations and labels do not match.
     */
    void condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXs> &labels) {
        xObs_ = xObs;
        labels_ = labels;

        nObs_ = xObs_.rows();
        dimX_ = xObs_.cols();
        nClasses_ = labels_.maxCoeff() + 1;

        if(nObs_ != labels_.size()) {
            throw std::runtime_error("KDE::set: Number of observations and labels size do not match.");
        }

        classCounts_ = Eigen::VectorXs::Zero(nClasses_);
        for(size_t i = 0; i < nObs_; i++) {
            classCounts_(labels_(i))++;
        }
    }

    /**
     * @brief Get the density of a class at a given point.
     * @param x The point at which to evaluate the density.
     * @param classLabel The label of the class for which to evaluate the density.
     * @return The density of the class at the given point.
     */
    double density(const Eigen::Ref<const Eigen::VectorXd> &x, const size_t &classLabel) const;

    /**
     * @brief Predict the class probabilities at a given point.
     * @param x The point at which to predict the class probabilities.
     * @return A vector of class probabilities.
     */
    std::vector<double> predictProbabilities(const Eigen::Ref<const Eigen::VectorXd> &x) const override;

    /**
     * @brief Predict the class label at a given point.
     * @param x The point at which to predict the class label.
     * @return The predicted class label.
     */
    size_t predict(const Eigen::Ref<const Eigen::VectorXd> &x) const override {
        std::vector<double> probs = predictProbabilities(x);
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    };


    /**
     * @brief Objective function for cross-validation.
     * This function computes the objective value for cross-validation, to be maximized.
     * @param method The method to use for cross-validation (CV_SCORE or CV_PROB_SCORE).
     * @param kf The KFold object containing the cross-validation splits.
     * @return The objective value for cross-validation.
     */
    double objectiveFunctionCV(const method& method, const cmp::statistics::KFold& kf) const;

    /**
     * @brief Objective function for efficient leave-one-out optimization.
     * This function computes mean LOO log-probability over all observations, to be maximized.
     */
    double objectiveFunctionLOO() const;

    /**
     * @brief Fit the KDE using cross-validation to optimize the bandwidth.
     * This method uses the specified optimization algorithm to find the optimal bandwidth that maximizes the cross-validation score.
     * @param xObs A matrix of observations where each row is an observation and each column is a feature.
     * @param labels A vector of labels corresponding to the observations.
     * @param kf A KFold object containing the cross-validation splits.
     * @param minBw The minimum bandwidth value for optimization.
     * @param maxBw The maximum bandwidth value for optimization.
     * @param method The method to use for cross-validation (CV_SCORE or CV_PROB_SCORE).
     * @param algo The optimization algorithm to use (default is nlopt::LN_SBPLX).
     * @param ftol_rel The relative tolerance for the optimization algorithm (default is 1e-4).
     * @param logScaleFlags A vector of booleans indicating whether to use log-scaling for each bandwidth parameter (default is all false).
     *
     */
    void fit(const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& labels, cmp::statistics::KFold kf, const double& minBw, const double& maxBw, const method method = CV_PROB_SCORE, nlopt::algorithm algo = nlopt::LN_SBPLX, double ftol_rel = 1e-4, std::vector<bool> logScaleFlags = {});


    /**
     * @brief Fit the KDE by maximizing an efficient leave-one-out objective.
     * This method optimizes bandwidth using mean LOO log-probability over all observations.
     * @param xObs A matrix of observations where each row is an observation and each column is a feature.
     * @param labels A vector of labels corresponding to the observations.
     * @param minBw The minimum bandwidth value for optimization.
     * @param maxBw The maximum bandwidth value for optimization.
     * @param algo The optimization algorithm to use (default is nlopt::LN_SBPLX).
     * @param ftol_rel The relative tolerance for the optimization algorithm (default is 1e-4).
     * @param logScaleFlags A vector of booleans indicating whether to use log-scaling for each bandwidth parameter (default is all false).
     */
    void fitLOO(const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& labels, const double& minBw, const double& maxBw, nlopt::algorithm algo = nlopt::LN_SBPLX, double ftol_rel = 1e-4, std::vector<bool> logScaleFlags = {});
};



/**
 * @brief Classifier that uses Support Vector Machine (SVM) for classification.
 * 
 * @details Mathematical Formulation
 * SVM constructs a hyperparameter-tuned decision boundary in a high-dimensional feature space mapped via a kernel:
 * \f[
 * k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)
 * \f]
 * The dual optimization problem solved is:
 * \f[
 * \min_{\boldsymbol{\alpha}} \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j k(\mathbf{x}_i, \mathbf{x}_j) - \sum_{i=1}^N \alpha_i
 * \f]
 * subject to:
 * \f[
 * 0 \le \alpha_i \le C, \quad \sum_{i=1}^N \alpha_i y_i = 0
 * \f]
 * where \f$C > 0\f$ is the regularization parameter, \f$y_i \in \{-1, 1\}\f$ are class labels, and \f$\alpha_i\f$ are Lagrange multipliers.
 * Class probabilities are calculated using Platt scaling to compute the probability of class 1:
 * \f[
 * P(y=1 \mid \mathbf{x}) = \frac{1}{1 + \exp(A f(\mathbf{x}) + B)}
 * \f]
 * where \f$f(\mathbf{x}) = \sum_i \alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b\f$ is the decision function, and parameters \f$A, B\f$ are fitted via maximum likelihood on cross-validation outputs.
 * 
 * @details Implementation Algorithm
 * 1. `condition()` maps training dataset to LIBSVM nodes, constructs the precomputed kernel matrix \f$\mathbf{K}\f$, and trains the support vectors via Sequential Minimal Optimization (SMO).
 * 2. `predictProbabilities()` performs Platt scaling by passing computed decision function outputs to Platt's sigmoid model.
 * 3. `fit()` optimizes hyperparameters (including kernel lengthscales and SVM penalty \f$C\f$) using cross-validation or span-based bounds.
 */
class SVM : public Classifier {
  private:

    struct svm_model *model_ = nullptr; /// Pointer to the SVM model
    struct svm_problem prob_;            /// Structure to hold the SVM problem

    struct svm_parameter modelParameters_;                          /// Structure to hold the SVM parameters
    std::shared_ptr<cmp::covariance::Covariance> covariance_{nullptr};  /// Covariance object for the kernel
    Eigen::VectorXd hyperparameters_ = Eigen::VectorXd(0);                             /// Hyperparameters for the covariance function

    // Eigen::VectorXd to hold the original training data
    Eigen::MatrixXd xObs_; ///< Training observation points.
    size_t nObs_ = 0;      ///< Total number of training observations.

    /**
     * @brief Custom print redirect function to suppress standard console output from LIBSVM.
     * 
     * @param s The string output from LIBSVM.
     */
    static void silent_print(const char *s) {}

    /**
     * @brief Initializes default parameters for the SVM model.
     * 
     * @param C Regularization penalty parameter.
     * @param eps Tolerance of termination criterion.
     */
    void initDefaultParameters(double C, double eps) {
        modelParameters_ = svm_parameter();

        // Use a classifier with precomputed kernel
        modelParameters_.svm_type = C_SVC;
        modelParameters_.kernel_type = PRECOMPUTED;

        // Set the two parameters C and eps
        modelParameters_.eps = eps;
        modelParameters_.C = C;

        // Other parameters
        modelParameters_.weight = nullptr;
        modelParameters_.weight_label = nullptr;
        modelParameters_.shrinking = 1;
        modelParameters_.probability = 1;
        modelParameters_.nr_weight = 0;
        modelParameters_.cache_size = 200;
    }


    // Helper function to free a problem pointer
    static void freeProblem(struct svm_problem *prob) {
        if(prob->x) {
            for(int i = 0; i < prob->l; i++) {
                delete[] prob->x[i];
            }
            delete[] prob->x;
            prob->x = nullptr;
        }
        if(prob->y) {
            delete[] prob->y;
            prob->y = nullptr;
        }
        prob->l = 0;
    }

  public:

    // Initialize the SVM with default parameters
    SVM() {
        initDefaultParameters(100, 1e-3);

        // Disable LIBSVM printing by default
        svm_set_print_string_function(&silent_print);
    };

    /**
     * Call to LIBSVM function to free the model.
     */
    ~SVM() {
        svm_free_and_destroy_model(&model_);
        freeProblem(&prob_);
    };

    /**
     * @brief Set the covariance function, hyperparameters, and SVM parameters.
     * This method allows you to set the covariance function, its hyperparameters, and the SVM parameters C and eps.
     * @param covariance A shared pointer to a covariance object that defines the kernel function.
     * @param hpar A vector of hyperparameters for the covariance function.
     * @param C The SVM regularization parameter. Note, this is treated as an extra hyperparameter during fitting.
     * @param eps The stopping criterion for the SVM training algorithm (default is 1e-3).
     */
    void set(std::shared_ptr<cmp::covariance::Covariance> covariance, const Eigen::Ref<const Eigen::VectorXd> hpar, const double &C, const double &eps = 1e-3) {
        initDefaultParameters(C, eps);
        covariance_ = covariance;
        hyperparameters_ = hpar;
    }


    /**
     * @brief Condition the SVM with observations and labels.
     * This method sets the observations and labels for the SVM and trains the model using the specified kernel.
     * @param xObs A matrix of observations where each row is an observation and each column is a feature.
     * @param labels A vector of labels corresponding to the observations.
     * @throws std::runtime_error if the number of observations and labels do not match.
     */
    void condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXs> &labels);


    /**
     * @brief Predict the class label at a given point.
     * @param x The point at which to predict the class label.
     * @return The predicted class label.
     */
    size_t predict(const Eigen::Ref<const Eigen::VectorXd> &x) const override;

    /**
     * @brief Predict the class probabilities at a given point.
     * @param x The point at which to predict the class probabilities.
     * @return A vector of class probabilities.
     */
    std::vector<double> predictProbabilities(const Eigen::Ref<const Eigen::VectorXd> &x) const override;


    /**
     * @brief Get the current hyperparameters and SVM regularization parameter C.
     */
    std::pair<Eigen::VectorXd, double> getHyperparameters() const {
        return {hyperparameters_, modelParameters_.C};
    }


    /**
     * @brief Fit the SVM using cross-validation to optimize hyperparameters and C.
     * This method uses the specified optimization algorithm to find the optimal hyperparameters and C that maximize the cross-validation score.
     * @param method The method to use for cross-validation (CV_SCORE or CV_PROB_SCORE).
     * @param kf A KFold object containing the cross-validation splits.
     * @param xObs A matrix of observations where each row is an observation and each column is a feature.
     * @param membershipTable A vector of membership values (class labels) corresponding to the observations.
     * @param lb A vector of lower bounds for the hyperparameters and C.
     * @param ub A vector of upper bounds for the hyperparameters and C.
     * @param algo The optimization algorithm to use (default is nlopt::LN_SBPLX).
     * @param ftol_rel The relative tolerance for the optimization algorithm (default is 1e-4).
     * @param logScaleFlags A vector of booleans indicating whether to use log-scaling for each bandwidth parameter (default is all false).
     */
    void fit(const method& method, const cmp::statistics::KFold& kf, const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& membershipTable, Eigen::VectorXd lb, Eigen::VectorXd ub, nlopt::algorithm algo = nlopt::LN_SBPLX, double ftol_rel = 1e-4, std::vector<bool> logScaleFlags = {});

    /**
     * @brief Fit the SVM using span-based optimization to optimize hyperparameters and C.
     * This method uses the specified optimization algorithm to find the optimal hyperparameters and C that minimizes the span of the support vectors.
     * @param xObs A matrix of observations where each row is an observation and each column is a feature.
     * @param membershipTable A vector of membership values (class labels) corresponding to the observations.
     * @param lb A vector of lower bounds for the hyperparameters and C.
     * @param ub A vector of upper bounds for the hyperparameters and C.
     * @param algo The optimization algorithm to use (default is nlopt::LN_SBPLX).
     * @param ftol_rel The relative tolerance for the optimization algorithm (default is 1e-4).
     * @param logScaleFlags A vector of booleans indicating whether to use log-scaling for each bandwidth parameter (default is all false).
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> xObs, Eigen::Ref<const Eigen::VectorXs> membershipTable, Eigen::Ref<const Eigen::VectorXd> lb, Eigen::Ref<const Eigen::VectorXd> ub, nlopt::algorithm algo = nlopt::LN_SBPLX, double ftol_rel = 1e-4, std::vector<bool> logScaleFlags = {});


    /**
     * @brief Objective function for cross-validation.
     * This function computes the objective value for cross-validation, to be maximized.
     * @param method The method to use for cross-validation (CV_SCORE or CV_PROB_SCORE).
     * @param kf The KFold object containing the cross-validation splits.
     * @return The objective value for cross-validation.
     */
    double objectiveFunctionCV(const method& method, const cmp::statistics::KFold& kf);

    /**
     * @brief Objective function for span-based optimization.
     * This function returns the negative mean support-vector span so it can be maximized.
     */
    double objectiveFunctionSpan();
};


/**
 * @brief Baseline classifier predicting class labels based on training frequency distributions.
 * 
 * @details Mathematical Formulation
 * The predicted probability for class \f$c\f$ at any input point \f$\mathbf{x}\f$ is equal to the class prior:
 * \f[
 * P(y=c \mid \mathbf{x}) = \frac{N_c}{N}
 * \f]
 * where \f$N_c\f$ is the number of occurrences of class \f$c\f$ in the training set and \f$N\f$ is the total number of training observations.
 * 
 * @details Implementation Algorithm
 * 1. `condition()` computes class counts \f$N_c\f$ from the labels training vector.
 * 2. `predictProbabilities()` returns a vector of class proportions.
 */
class Dummy : public Classifier {
  private :
    size_t nClasses_{1}; ///< Total number of class labels.

  public :
    /**
     * @brief Computes the class frequencies from training labels.
     * 
     * @param xObs A matrix of training observations (unused).
     * @param labels A vector of training labels.
     */
    void condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXs> &labels);

    /**
     * @brief Predicts the class label with the highest prior frequency.
     * 
     * @param x The query point (unused).
     * @return The most frequent class label.
     */
    size_t predict(const Eigen::Ref<const Eigen::VectorXd> &x) const override;

    /**
     * @brief Estimates the class prior probabilities based on training frequencies.
     * 
     * @param x The query point (unused).
     * @return A vector of class prior probabilities.
     */
    std::vector<double> predictProbabilities(const Eigen::Ref<const Eigen::VectorXd> &x) const override;

};

} // namespace cmp::classifier

/** @} */

#endif // CLASSIFIER_H