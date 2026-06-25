#include <functional>
#include <random>
#include <Eigen/Dense>
#include <cmp_defines.h>

namespace cmp::mcmc {


/**
 * @brief The HamiltonianMarkovChain class
 * Implements the Hamiltonian Monte Carlo (HMC) algorithm for sampling from a target distribution.
 *
 * This class uses the leapfrog integrator to propose new states in the Markov chain, and it employs
 * the No-U-Turn Sampler (NUTS) to adaptively determine the number of leapfrog steps.
 *
 * The user must provide a score function (log-posterior) and its gradient.
 */
class HamiltonianMarkovChain {
  private:
    std::default_random_engine& rng_; // Reference to the random number generator

    Eigen::VectorXd currentPosition_; // The current position (parameters)
    Eigen::VectorXd currentMomentum_; // The current momentum (auxiliary variable)
    double currentScore_;             // The current Score (log-posterior) value
    size_t dim_;                      // Dimension of the parameter space (not sure if this is needed)

    // keep track of the mean and covariance of the samples
    Eigen::VectorXd mean_;   // Sample-mean vector
    Eigen::MatrixXd cov_;    // Sample-covariance matrix

    double epsilon_;                  // Step size for the leapfrog integrator

    size_t nSteps_ = 0;                   // Number of steps taken in the Markov chain
    double sumAcceptanceProb_ = 0.0;      // Sum of acceptance probabilities for adaptation

    // Dual Averaging parameters for step size adaptation
    double mu_;
    double gamma_ = 0.05;
    double t0_ = 10.0;
    double kappa_ = 0.75;

    // Dual Averaging State Trackers
    double H_bar_ = 0.0;
    double log_epsilon_bar_ = 0.0;
    double target_accept_ = 0.80; // This should be set by the user, but we can provide a default value


    // Proposal distributions for momentum and uniform random numbers
    std::normal_distribution<double> distN_{0.0, 1.0};
    std::uniform_real_distribution<double> distU_{0.0, 1.0};


    /**
     * @brief Perform a single leapfrog step
     * @param q The current position (parameters)
     * @param p The current momentum (auxiliary variable)
     * @param grad The gradient of the score function at the current position
     * @param epsilon The step size for the leapfrog integrator
     * @param getGradient The gradient function of the score function
     * @note The gradient is passed by reference and will be updated to the new gradient after the leapfrog step.
     */
    void leapfrog(Eigen::VectorXd& q, Eigen::VectorXd& p, Eigen::VectorXd& grad, double epsilon, const gradient_t& getGradient) const;


    /**
     * Struct to hold the state of the tree during the NUTS algorithm.
     *
     * q -> Position
     * p -> Momentum
     * grad -> Gradient of the score function
     * n -> Number of valid points in the subtree
     * s -> Stop flag (1 if the subtree is valid, 0 otherwise)
     * alpha -> Sum of acceptance probabilities
     * n_alpha -> Number of valid points for acceptance probability
     */
    struct TreeState {
        Eigen::VectorXd q_minus;
        Eigen::VectorXd p_minus;
        Eigen::VectorXd grad_minus;

        Eigen::VectorXd q_plus;
        Eigen::VectorXd p_plus;
        Eigen::VectorXd grad_plus;

        Eigen::VectorXd q_prime;
        size_t n;
        int s;
        double alpha;
        size_t n_alpha;
    };

    /**
     * @brief Build a binary tree for the NUTS algorithm
     * @param q The current position (parameters)
     * @param p The current momentum (auxiliary variable)
     * @param grad The gradient of the score function at the current position
     * @param log_u The log of the slice variable
     * @param v The direction of tree expansion (-1 for backward, +1 for forward)
     * @param j The depth of the tree
     * @param epsilon The step size for the leapfrog integrator
     * @param getScore The score function (log-posterior)
     * @param getGradient The gradient function of the score function
      * @return The state of the tree after expansion
      * @note This function is called recursively to build the tree. It returns the state of the tree after expansion, which includes the new position, momentum, gradient, and acceptance probabilities.
      * The function also checks for U-turns and updates the stop flag accordingly. The acceptance probabilities are accumulated to compute the overall acceptance ratio for the NUTS algorithm.
      * The gradient is passed by reference and will be updated to the new gradient after the leapfrog step. This is important for efficiency, as it avoids redundant gradient calculations during the tree expansion.
      * The log of the slice variable is used to determine whether the proposed states are within the slice defined by the current Hamiltonian level. This is crucial for the correctness of the NUTS algorithm, as it ensures that the samples are drawn from the correct distribution.
      * The direction of tree expansion (v) determines whether the tree is built forward (v=+1) or backward (v=-1) in time. This allows the NUTS algorithm to explore the parameter space more effectively and adaptively determine the number of leapfrog
     */
    TreeState build_tree(const Eigen::VectorXd& q, const Eigen::VectorXd& p, const Eigen::VectorXd& grad, double log_u, int v, int j, double epsilon, const score_t& getScore, const gradient_t& getGradient);

    /**
     * Updates the mean and covariance of the samples based on the current position.
     */
    void update();

  public:
    HamiltonianMarkovChain(Eigen::VectorXd initialState, std::default_random_engine& rng, double epsilon);

    /**
     * @brief Perform a single MCMC step using the NUTS algorithm
     * @param getScore The score function (log-posterior)
     * @param getGradient The gradient function of the score function
     * @param adapt Whether to adapt the step size using dual averaging (default is false)
     * @note If adapt is true, the step size will be adapted using dual averaging to achieve the target acceptance ratio. If adapt is false, the step size will remain fixed.
     */
    void step(const score_t& getScore, const gradient_t& getGradient, bool adapt = false);
    void reset();

    Eigen::VectorXd getCurrent() const;
    size_t getSteps() const;
    double getAcceptanceRatio() const;

    double getStepSize() const {
        return epsilon_;
    }

    Eigen::VectorXd getMean() const {
        return mean_;
    }

    Eigen::MatrixXd getCovariance() const {
        return cov_ / static_cast<double>(std::max(nSteps_ - 1, (size_t)1));
    }

    /**
     * Print the current state of the Markov chain, including the current position, step size, number of steps, and acceptance ratio.
     */
    void info() const;
};

}