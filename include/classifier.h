#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <cmp_defines.h>
#include <list>
#include <kernel++.h>
#include <svm.h>
#include <statistics.h>
#include <optimization.h>
#include <kde.h>

namespace cmp::classifier {

enum method {CV_SCORE, CV_PROB_SCORE};


// Abstract class for classifier
class Classifier {
  public:
    virtual ~Classifier() = default;
    virtual std::vector<double> predictProbabilities(const Eigen::Ref<const Eigen::VectorXd> &x) const = 0;
    virtual size_t predict(const Eigen::Ref<const Eigen::VectorXd> &x) const = 0;
};

class KNN {
  private:
    std::vector<std::vector<size_t>> neighbours_;
    size_t kNearestValue_;
    size_t nPoints_;

  public:
    KNN() = default;

    void compute(const std::vector<Eigen::VectorXd> &points, size_t k);

    const std::vector<size_t> &operator[](size_t i) const;
    size_t nPoints() const;
    size_t k() const;
};



/**
 * \brief Classifier that uses Kernel Density Estimation (KDE) for classification.
 * This class implements a non-parametric method to estimate the probability density function of a random variable.
 * It uses a kernel function to smooth the observations and estimate the density.
 * It can be used for classification by predicting the class probabilities based on the estimated densities.
 */
class KDE : public Classifier {
  private:
    Eigen::MatrixXd xObs_ = Eigen::MatrixXd(0, 0); /// Observations
    Eigen::VectorXs labels_ = Eigen::VectorXs(0);  /// Labels for the observations
    Eigen::VectorXs classCounts_;               /// Count of observations per class

    size_t nObs_ = 0;
    size_t dimX_ = 0;
    size_t nClasses_ = 1;

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
     * @param T Temperature scaling factor for the probabilities.
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
     * @brief Objective function for entropy-based optimization.
     * This function computes the objective value based on the entropy of the predicted class probabilities.
     * @param targetEntropy The target entropy value to achieve.
     * @return The objective value based on the entropy of the predicted class probabilities: (entropy - targetEntropy)^2.
     */
    double objectiveFunctionEntropy(const double &targetEntropy) const;

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
     *
     */
    void fit(const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& labels, cmp::statistics::KFold kf, const double& minBw, const double& maxBw, const method method = CV_PROB_SCORE, nlopt::algorithm algo = nlopt::LN_SBPLX, double ftol_rel = 1e-4);


    /**
     * @brief Fit the KDE using entropy-based optimization.
     * This method optimizes the bandwidth using the specified optimization algorithm to minimize the difference between the predicted and target entropy.
     * @param targetEntropy The target entropy value to achieve.
     * @param xObs A matrix of observations where each row is an observation and each column is a feature.
     * @param labels A vector of labels corresponding to the observations.
     * @param minBw The minimum bandwidth value for optimization.
     * @param maxBw The maximum bandwidth value for optimization.
     * @param algo The optimization algorithm to use (default is nlopt::LN_SBPLX).
     * @param ftol_rel The relative tolerance for the optimization algorithm (default is 1e-4).
     *
     */
    void fit(const double &targetEntropy, const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& labels, const double& minBw, const double& maxBw, nlopt::algorithm algo = nlopt::LN_SBPLX, double ftol_rel = 1e-4);
};



/**
 * @brief Classifier that uses Support Eigen::VectorXd Machine (SVM) for classification.
 * This class implements a supervised learning algorithm for classification tasks using LIBSVM.
 */
class SVM : public Classifier {
  private:

    struct svm_model *model_ = nullptr; /// Pointer to the SVM model
    struct svm_problem prob_;            /// Structure to hold the SVM problem

    struct svm_parameter modelParameters_;                          /// Structure to hold the SVM parameters
    std::shared_ptr<cmp::covariance::Covariance> covariance_{nullptr};  /// Covariance object for the kernel
    Eigen::VectorXd hyperparameters_ = Eigen::VectorXd(0);                             /// Hyperparameters for the covariance function

    // Eigen::VectorXd to hold the original training data
    Eigen::MatrixXd xObs_;
    size_t nObs_ = 0;

    // Custom print function that does nothing
    static void silent_print(const char *s) {}

    // Initialize default parameters for the SVM model
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
     * @param labels A vector of labels corresponding to the observations.
     * @param lb A vector of lower bounds for the hyperparameters and C.
     * @param ub A vector of upper bounds for the hyperparameters and C.
     * @param algo The optimization algorithm to use (default is nlopt::LN_SBPLX).
     * @param ftol_rel The relative tolerance for the optimization algorithm (default is 1e-4).
     */
    void fit(const method& method, const cmp::statistics::KFold& kf, const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& membershipTable, Eigen::VectorXd lb, Eigen::VectorXd ub, nlopt::algorithm algo = nlopt::LN_SBPLX, double ftol_rel = 1e-4);

    /**
     * @brief Fit the SVM using span-based optimization to optimize hyperparameters and C.
     * This method uses the specified optimization algorithm to find the optimal hyperparameters and C that minimizes the span of the support vectors.
     * @param xObs A matrix of observations where each row is an observation and each column is a feature.
     * @param labels A vector of labels corresponding to the observations.
     * @param lb A vector of lower bounds for the hyperparameters and C.
     * @param ub A vector of upper bounds for the hyperparameters and C.
     * @param algo The optimization algorithm to use (default is nlopt::LN_SBPLX).
     * @param ftol_rel The relative tolerance for the optimization algorithm (default is 1e-4).
     */
    void fit(Eigen::Ref<const Eigen::MatrixXd> xObs, Eigen::Ref<const Eigen::VectorXs> membershipTable, Eigen::Ref<const Eigen::VectorXd> lb, Eigen::Ref<const Eigen::VectorXd> ub, nlopt::algorithm algo = nlopt::LN_SBPLX, double ftol_rel = 1e-4);


    /**
     * @brief Fit the SVM using entropy-based optimization to optimize hyperparameters and C.
     * This method uses the specified optimization algorithm to find the optimal hyperparameters and C that achieves the target entropy.
     * @param targetEntropy The target entropy value to achieve.
     * @param xObs A matrix of observations where each row is an observation and each column is a feature.
     * @param labels A vector of labels corresponding to the observations.
     * @param lb A vector of lower bounds for the hyperparameters and C.
     * @param ub A vector of upper bounds for the hyperparameters and C.
     * @param algo The optimization algorithm to use (default is nlopt::LN_SBPLX).
     * @param ftol_rel The relative tolerance for the optimization algorithm (default is 1e-4).
     */
    void fit(const double & targetEntropy, const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& membershipTable, const Eigen::Ref<const Eigen::VectorXd> &lb, const Eigen::Ref<const Eigen::VectorXd> &ub, nlopt::algorithm algo = nlopt::LN_SBPLX, double ftol_rel = 1e-4);


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
     * This function computes the -log(1 + span) to be maximized.
     */
    double objectiveFunctionSpan();

    /**
     * @brief Objective function for entropy-based optimization.
     * This function computes the objective value based on the entropy of the predicted class probabilities.
     * @param targetEntropy The target entropy value to achieve.
     * @return The objective value based on the entropy of the predicted class probabilities: (entropy - targetEntropy)^2.
     */
    double objectiveFunctionEntropy(const double & targetEntropy);
};


class Dummy : public Classifier {
  private :
    size_t nClasses_{1};

  public :
    void condition(const Eigen::Ref<const Eigen::MatrixXd> &xObs, const Eigen::Ref<const Eigen::VectorXs> &labels);
    size_t predict(const Eigen::Ref<const Eigen::VectorXd> &x) const override;
    std::vector<double> predictProbabilities(const Eigen::Ref<const Eigen::VectorXd> &x) const override;

};

} // namespace cmp::classifier

#endif // CLASSIFIER_H