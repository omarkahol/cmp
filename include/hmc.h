#include <functional>
#include <random>
#include <Eigen/Dense>
#include <cmp_defines.h>

/**
 * @addtogroup sampling
 * @{
 */
namespace cmp::mcmc {


/**
 * @class HamiltonianMarkovChain
 * @brief Implements the Hamiltonian Monte Carlo (HMC) algorithm with the No-U-Turn Sampler (NUTS) and Dual Averaging.
 *
 * @details
 * ### Mathematical Foundations
 * Hamiltonian Monte Carlo (HMC) maps the target probability density \f$p(\mathbf{q})\f$ to the potential energy
 * of a physical system: \f$U(\mathbf{q}) = -\log p(\mathbf{q})\f$. We introduce auxiliary momentum variables
 * \f$\mathbf{p} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})\f$ with kinetic energy \f$K(\mathbf{p}) = \frac{1}{2} \mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}\f$.
 *
 * The total energy is given by the Hamiltonian:
 * \f[ H(\mathbf{q}, \mathbf{p}) = U(\mathbf{q}) + K(\mathbf{p}) \f]
 *
 * The joint system evolves according to Hamilton's equations:
 * \f[ \frac{d\mathbf{q}}{dt} = \frac{\partial H}{\partial \mathbf{p}} = \mathbf{M}^{-1} \mathbf{p}, \quad \frac{d\mathbf{p}}{dt} = -\frac{\partial H}{\partial \mathbf{q}} = -\nabla U(\mathbf{q}) \f]
 *
 * ### Implementation Algorithms
 * 1. **Leapfrog Integrator**:
 *    To integrate the system numerically with step size \f$\epsilon\f$, we use the symplectic Leapfrog scheme:
 *    \f[ \mathbf{p}\left(t + \frac{\epsilon}{2}\right) = \mathbf{p}(t) - \frac{\epsilon}{2} \nabla U(\mathbf{q}(t)) \f]
 *    \f[ \mathbf{q}(t + \epsilon) = \mathbf{q}(t) + \epsilon \mathbf{M}^{-1} \mathbf{p}\left(t + \frac{\epsilon}{2}\right) \f]
 *    \f[ \mathbf{p}(t + \epsilon) = \mathbf{p}\left(t + \frac{\epsilon}{2}\right) - \frac{\epsilon}{2} \nabla U(\mathbf{q}(t + \epsilon)) \f]
 * 2. **No-U-Turn Sampler (NUTS)**:
 *    Avoids manual tuning of the number of integration steps \f$L\f$. It recursively builds a binary tree of leapfrog steps
 *    forward and backward in time. The tree stops growing when the trajectory starts to turn back on itself:
 *    \f[ (\mathbf{q}_{\text{plus}} - \mathbf{q}_{\text{minus}})^T \mathbf{p}_{\text{minus}} < 0 \quad \text{or} \quad (\mathbf{q}_{\text{plus}} - \mathbf{q}_{\text{minus}})^T \mathbf{p}_{\text{plus}} < 0 \f]
 * 3. **Dual Averaging Step Size Adaptation**:
 *    Uses Nesterov's dual averaging scheme to adaptively tune the step size \f$\epsilon\f$ to match a target acceptance probability \f$\delta\f$ (default 0.8):
 *    \f[ H_t = \delta - \alpha_t, \quad \bar{H}_t = \left(1 - \frac{1}{t + t_0}\right) \bar{H}_{t-1} + \frac{1}{t + t_0} H_t \f]
 *    \f[ \log \epsilon_t = \mu - \frac{\sqrt{t}}{\gamma} \bar{H}_t, \quad \log \bar{\epsilon}_t = t^{-\kappa} \log \epsilon_t + (1 - t^{-\kappa}) \log \bar{\epsilon}_{t-1} \f]
 *
 * ### Constraints & Invariants
 * - **Symplectic Flow**: The Leapfrog scheme must preserve volume in phase space and hold total energy approximately constant.
 * - **U-turn Condition**: Tree growth is terminated immediately if a U-turn is detected to ensure detailed balance is maintained.
 */
class HamiltonianMarkovChain {
  private:
    std::default_random_engine& rng_; ///< Reference to the random number generator.

    Eigen::VectorXd currentPosition_; ///< The current position (parameter state).
    Eigen::VectorXd currentMomentum_; ///< The current momentum auxiliary variable.
    double currentScore_;             ///< The current log-posterior score value.
    size_t dim_;                      ///< Dimension of the parameter space.

    // keep track of the mean and covariance of the samples
    Eigen::VectorXd mean_;            ///< Running mean of the parameter samples.
    Eigen::MatrixXd cov_;             ///< Running covariance of the parameter samples.

    double epsilon_;                  ///< Step size for the leapfrog integrator.

    size_t nSteps_ = 0;               ///< Number of steps taken in the Markov chain.
    double sumAcceptanceProb_ = 0.0;  ///< Sum of acceptance probabilities for step size adaptation.

    // Dual Averaging parameters for step size adaptation
    double mu_;                       ///< Target value for log step size.
    double gamma_ = 0.05;             ///< Adaptation shrinkage parameter.
    double t0_ = 10.0;                ///< Adaptation stabilization parameter.
    double kappa_ = 0.75;             ///< Adaptation exponential decay parameter.

    // Dual Averaging State Trackers
    double H_bar_ = 0.0;              ///< Running average of difference between target and accept probability.
    double log_epsilon_bar_ = 0.0;    ///< Log of the adapted step size.
    double target_accept_ = 0.80;     ///< Target acceptance probability.


    // Proposal distributions for momentum and uniform random numbers
    std::normal_distribution<double> distN_{0.0, 1.0};      ///< Normal generator helper for momentum initialization.
    std::uniform_real_distribution<double> distU_{0.0, 1.0}; ///< Uniform generator helper for transition accept checks.


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
        Eigen::VectorXd q_minus;    ///< Position at the leftmost leaf.
        Eigen::VectorXd p_minus;    ///< Momentum at the leftmost leaf.
        Eigen::VectorXd grad_minus; ///< Gradient at the leftmost leaf.

        Eigen::VectorXd q_plus;     ///< Position at the rightmost leaf.
        Eigen::VectorXd p_plus;     ///< Momentum at the rightmost leaf.
        Eigen::VectorXd grad_plus;  ///< Gradient at the rightmost leaf.

        Eigen::VectorXd q_prime;    ///< Proposed next sample position.
        size_t n;                   ///< Number of valid states in the subtree.
        int s;                      ///< Trajectory validity flag (0 = U-turn/divergence, 1 = valid).
        double alpha;               ///< Accumulated acceptance probabilities in subtree.
        size_t n_alpha;             ///< Number of states for calculating average acceptance probability.
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
/** @} */

