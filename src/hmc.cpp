#include "hmc.h"
#include <cmath>
#include <iostream>

void cmp::mcmc::HamiltonianMarkovChain::leapfrog(Eigen::VectorXd& q, Eigen::VectorXd& p, Eigen::VectorXd& grad,  double epsilon, const gradient_t& getGradient) const {

    // Half-step momentum update using the current gradient
    p += 0.5 * epsilon * grad;

    // Full step in position update using the updated momentum
    q += epsilon * p;

    // Now we need to calculate the new gradient at the updated position q
    grad = getGradient(q);

    // Another half-step momentum update using the new gradient
    p += 0.5 * epsilon * grad;

    // At this point, q and p have been updated according to the leapfrog integrator, and grad contains the gradient at the new position q.
}

void cmp::mcmc::HamiltonianMarkovChain::update() {

    if(nSteps_ == 1) {
        mean_ = currentPosition_;
        cov_.setZero(dim_, dim_);
    } else {
        // Welford's stable 1-pass multi-dimensional algorithm
        Eigen::VectorXd delta_old = currentPosition_ - mean_;
        mean_ += delta_old / static_cast<double>(nSteps_);
        Eigen::VectorXd delta_new = currentPosition_ - mean_;

        cov_ += delta_old * delta_new.transpose();
    }
}

// Constructor
cmp::mcmc::HamiltonianMarkovChain::HamiltonianMarkovChain(Eigen::VectorXd initialState, std::default_random_engine& rng, double epsilon)
    : rng_(rng),
      currentPosition_(std::move(initialState)),
      currentScore_(-std::numeric_limits<double>::infinity()), // Init to -inf to ensure the first score is always better
      currentMomentum_(currentPosition_.size()), // Initialize momentum vector with the same size as the
      dim_(currentPosition_.size()),
      epsilon_(epsilon) {

    // Initialize Dual Averaging states
    mu_ = std::log(10.0 * epsilon_);
    log_epsilon_bar_ = std::log(epsilon_);
}


// Build tree function for NUTS
cmp::mcmc::HamiltonianMarkovChain::TreeState cmp::mcmc::HamiltonianMarkovChain::build_tree(const Eigen::VectorXd& q, const Eigen::VectorXd& p, const Eigen::VectorXd& grad, double log_u, int v, int j, double epsilon, const score_t& getScore, const gradient_t& getGradient) {

    // Initialize the state of the tree
    TreeState state;

    // Base case: Take one leapfrog step in the direction v
    if(j == 0) {
        Eigen::VectorXd q_prime = q;
        Eigen::VectorXd p_prime = p;
        Eigen::VectorXd grad_prime = grad; // Copy the incoming gradient

        // Perform one leapfrog step in the direction v
        leapfrog(q_prime, p_prime, grad_prime, v * epsilon, getGradient);

        // Update the tree state with the new position, momentum, and gradient
        // Notice that + and - boundaries are the same in the base case because we only take one step and don't expand in both directions yet.
        state.q_minus = q_prime;
        state.p_minus = p_prime;
        state.grad_minus = grad_prime;
        state.q_plus = q_prime;
        state.p_plus = p_prime;
        state.grad_plus = grad_prime;
        state.q_prime = q_prime;

        // Compute the new Hamiltonian and check if the new state is in the slice
        double U_new = -getScore(q_prime);
        double K_new = 0.5 * p_prime.squaredNorm();
        double H_new = U_new + K_new;

        state.n = (log_u <= -H_new) ? 1 : 0;
        state.s = (log_u < -H_new + 1000.0) ? 1 : 0;

        double H_old = -currentScore_ + 0.5 * currentMomentum_.squaredNorm();
        state.alpha = std::min(1.0, std::exp(H_old - H_new));
        state.n_alpha = 1;

        return state;
    }

    // Recursive case: Build the left and right subtrees
    state = build_tree(q, p, grad, log_u, v, j - 1, epsilon, getScore, getGradient);

    // If the left subtree is valid, we need to build the right subtree and combine the results
    if(state.s == 1) {
        TreeState state_prime;

        if(v == -1) {
            // Build backwards from the left boundary using the left gradient
            state_prime = build_tree(state.q_minus, state.p_minus, state.grad_minus, log_u, v, j - 1, epsilon, getScore, getGradient);
            state.q_minus = state_prime.q_minus;
            state.p_minus = state_prime.p_minus;
            state.grad_minus = state_prime.grad_minus; // Update boundary gradient
        } else {
            // Build forwards from the right boundary using the right gradient
            state_prime = build_tree(state.q_plus, state.p_plus, state.grad_plus, log_u, v, j - 1, epsilon, getScore, getGradient);
            state.q_plus = state_prime.q_plus;
            state.p_plus = state_prime.p_plus;
            state.grad_plus = state_prime.grad_plus; // Update boundary gradient
        }

        // Reservoir sampling
        double accept_prob = static_cast<double>(state_prime.n) / static_cast<double>(std::max(state.n + state_prime.n, (size_t)1));
        if(distU_(rng_) < accept_prob) {
            state.q_prime = state_prime.q_prime;
        }

        state.n += state_prime.n;
        state.alpha += state_prime.alpha;
        state.n_alpha += state_prime.n_alpha;

        Eigen::VectorXd delta_q = state.q_plus - state.q_minus;
        bool uturn_right = delta_q.dot(state.p_plus) >= 0;
        bool uturn_left = delta_q.dot(state.p_minus) >= 0;

        // Check for U-turn condition and update the stop flag accordingly
        state.s = state_prime.s * uturn_right * uturn_left;
    }

    // Return the combined state of the tree after expansion
    return state;
}

// Perform a single MCMC step using the NUTS algorithm
void cmp::mcmc::HamiltonianMarkovChain::step(const score_t& getScore, const gradient_t& getGradient, bool adapt) {

    // Increment the step counter
    nSteps_++;

    // Sample a random momentum from N(0, I) (Is there a more efficient way in terms of memory or statistics ???) -> Check
    currentMomentum_ = Eigen::VectorXd::Zero(dim_);
    for(size_t i = 0; i < dim_; ++i) {
        currentMomentum_(i) = distN_(rng_);
    }

    // Compute the current gradient for the direction
    Eigen::VectorXd current_grad = getGradient(currentPosition_);

    // Compute the current Hamiltonian components
    double U_old = -currentScore_;
    double K_old = 0.5 * currentMomentum_.squaredNorm();
    double H_old = U_old + K_old;

    // Sample a slice variable u ~ Uniform(0, exp(-H_old)) and compute log_u for numerical stability
    double log_u = std::log(distU_(rng_)) - H_old;

    // Initialize the tree state for the NUTS algorithm
    Eigen::VectorXd q_minus = currentPosition_, q_plus = currentPosition_;
    Eigen::VectorXd p_minus = currentMomentum_, p_plus = currentMomentum_;
    Eigen::VectorXd grad_minus = current_grad, grad_plus = current_grad; // Initialized!


    // Initial depth of 0 means we will take one leapfrog step in either direction, and then we will expand the tree until we hit a stopping condition (like a U-turn or max depth).
    // N indicates the number of valid points in the tree, and s is the stop flag that indicates whether we should continue expanding the tree or not.
    // The S flag is set to 1 if the subtree is valid (i.e., it contains at least one point in the slice and does not violate the U-turn condition), and 0 otherwise. The algorithm continues to expand the tree until s becomes 0, which indicates that we have reached a stopping condition.
    int j = 0;
    size_t n = 1;
    int s = 1;

    Eigen::VectorXd q_next = currentPosition_;

    // Accumulate the total acceptance probability and the number of valid points for dual averaging adaptation
    double total_alpha = 0.0;
    size_t total_n_alpha = 0;

    while(s == 1) {

        // Randomly choose to expand the tree in the forward or backward direction with equal probability
        int v = (distU_(rng_) < 0.5) ? -1 : 1;

        TreeState state_prime;
        if(v == -1) {
            state_prime = build_tree(q_minus, p_minus, grad_minus, log_u, v, j, epsilon_, getScore, getGradient);
            q_minus = state_prime.q_minus;
            p_minus = state_prime.p_minus;
            grad_minus = state_prime.grad_minus;  // Capture the updated boundary gradient for the backward direction
        } else {
            state_prime = build_tree(q_plus, p_plus, grad_plus, log_u, v, j, epsilon_, getScore, getGradient);
            q_plus = state_prime.q_plus;
            p_plus = state_prime.p_plus;
            grad_plus = state_prime.grad_plus;   // Capture the updated boundary gradient for the forward direction
        }

        // If the subtree is valid, we need to decide whether to accept the new candidate point q_prime from the subtree. The acceptance probability is based on the number of valid points in the subtree compared to the total number of valid points in both subtrees. This is a form of reservoir sampling that ensures we sample uniformly from all valid points in the tree.
        if(state_prime.s == 1) {
            double accept_prob = static_cast<double>(state_prime.n) / static_cast<double>(n + state_prime.n);
            if(distU_(rng_) < accept_prob) {
                q_next = state_prime.q_prime;
            }
        }

        // Update the total number of valid points in the tree and the acceptance probability statistics
        n += state_prime.n;

        total_alpha += state_prime.alpha;
        total_n_alpha += state_prime.n_alpha;

        // Check for U-turn condition to determine if we should stop expanding the tree. The U-turn condition checks if the trajectory has made a U-turn in the parameter space, which would indicate that further expansion in that direction is unlikely to yield valid points. If a U-turn is detected, we set the stop flag s to 0 to terminate the tree expansion.
        Eigen::VectorXd delta_q = q_plus - q_minus;
        bool uturn_right = delta_q.dot(p_plus) >= 0;
        bool uturn_left = delta_q.dot(p_minus) >= 0;

        // This is if we have a U-turn in either direction, we should stop expanding the tree. The stop flag s is updated based on the validity of the subtree and the U-turn conditions. If either subtree is invalid (s = 0) or if a U-turn is detected (uturn_right or uturn_left is false), then s will be set to 0, which will terminate the while loop and stop further expansion of the tree.
        s = state_prime.s * uturn_right * uturn_left;
        j++;

        // We limit the maximum depth of the tree to prevent infinite loops in pathological cases. If we reach a certain depth (e.g., 10), we will stop expanding the tree regardless of the U-turn condition. This is a safety measure to ensure that the algorithm terminates even if it encounters a situation where it keeps expanding without finding a valid point or detecting a U-turn.
        if(j >= 10) s = 0;
    }

    // After the tree expansion is complete, we update the current position to the new candidate point q_next that was selected during the tree building process. We also compute the new score at this position and update the acceptance count if the new point was accepted.
    currentPosition_ = q_next;
    currentScore_ = getScore(currentPosition_);

    // Dual Averaging Adaptation
    double alpha_t = total_alpha / static_cast<double>(std::max(total_n_alpha, (size_t)1));
    sumAcceptanceProb_ += alpha_t;
    if(adapt) {
        double t = static_cast<double>(nSteps_);

        // 1. Update the smoothed error
        H_bar_ = (1.0 - 1.0 / (t + t0_)) * H_bar_ + (1.0 / (t + t0_)) * (target_accept_ - alpha_t);

        // 2. Aggressively update the active step size (Calculate and exponentiate in one line)
        epsilon_ = std::exp(mu_ - (std::sqrt(t) / gamma_) * H_bar_);

        // 3. Update the smoothed step size tracker
        double weight = std::pow(t, -kappa_);
        log_epsilon_bar_ = weight * std::log(epsilon_) + (1.0 - weight) * log_epsilon_bar_;

    }

    // Update the mean and covariance of the samples based on the current position
    update();
}

void cmp::mcmc::HamiltonianMarkovChain::reset() {
    nSteps_ = 0;
    sumAcceptanceProb_ = 0.0;

    mean_.setZero(dim_);
    cov_.setZero(dim_, dim_);
}

Eigen::VectorXd cmp::mcmc::HamiltonianMarkovChain::getCurrent() const {
    return currentPosition_;
}
size_t cmp::mcmc::HamiltonianMarkovChain::getSteps() const {
    return nSteps_;
}
double cmp::mcmc::HamiltonianMarkovChain::getAcceptanceRatio() const {
    return sumAcceptanceProb_ / static_cast<double>(std::max(nSteps_, (size_t)1));
}

void cmp::mcmc::HamiltonianMarkovChain::info() const {

    std::cout << "run " << nSteps_ << " steps\n"
              << "acceptance ratio: " << std::fixed << std::setprecision(3) << getAcceptanceRatio() << "\n"
              << "Data covariance: \n" << getCovariance() << "\n"
              << "Data mean: \n" << getMean() << "\n"
              << "Step Size: " << std::fixed << std::setprecision(6) << getStepSize() << "\n";

}
