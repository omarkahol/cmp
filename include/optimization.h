/**
 * @file optimization.h
 * @brief Functor wrapper for NLopt with automatic, transparent log-scaling.
 */

#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <nlopt.hpp>

/**
 * @addtogroup core
 * @{
 */
namespace cmp {

/**
 * @brief Functor wrapper for NLopt with automatic, transparent parameter space mapping (e.g., log-scaling).
 * 
 * @details Mathematical Formulation
 * When optimization parameters span multiple orders of magnitude, log-scaling maps the optimization coordinate \f$y_k\f$ to the physical parameter \f$x_k\f$:
 * \f[
 * x_k = \exp(y_k)
 * \f]
 * By the chain rule, the gradient evaluated by NLopt with respect to \f$y_k\f$ is:
 * \f[
 * \frac{\partial f}{\partial y_k} = \frac{\partial f}{\partial x_k} \frac{\partial x_k}{\partial y_k} = \frac{\partial f}{\partial x_k} \exp(y_k) = \frac{\partial f}{\partial x_k} x_k
 * \f]
 * 
 * @details Implementation Algorithm
 * 1. `mapToReal()` performs element-wise exponential transformation if log-scaling is enabled for a given parameter index.
 * 2. `mapGradientToOpt()` multiplies each computed gradient entry \f$\frac{\partial f}{\partial x_k}\f$ by \f$x_k\f$ if log-scaling is active.
 * 3. `NLoptCallback` coordinates the mapping of std::vector to Eigen::Map mapping and triggers evaluations.
 */
class ObjectiveFunctor {
  public:
    // Gradient-free constructor
    explicit ObjectiveFunctor(std::function<double(Eigen::Ref<const Eigen::VectorXd>)> fval)
        : fval_only_(std::move(fval)), fval_grad_inplace_(nullptr), use_gradient_(false) {}

    // Gradient-based constructor
    explicit ObjectiveFunctor(std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> fval_grad)
        : fval_only_(nullptr), fval_grad_inplace_(std::move(fval_grad)), use_gradient_(true) {}

    /**
     * @brief Set which parameters should be optimized in log-space.
     * @param log_scale A boolean mask where true means the parameter is log-scaled.
     */
    void setLogScale(const std::vector<bool>& log_scale) {
        log_scale_ = log_scale;
    }

    const std::vector<bool>& getLogScale() const {
        return log_scale_;
    }

    // Call operator evaluated by NLoptCallback
    double operator()(Eigen::Ref<const Eigen::VectorXd> x_opt, Eigen::Ref<Eigen::VectorXd> grad_opt) const {
        // 1. Transform optimizer-space parameters to physical real-space parameters
        Eigen::VectorXd x_real = mapToReal(x_opt);

        double val;
        if(use_gradient_ && grad_opt.size() > 0) {
            // 2. Evaluate the user's function with real parameters
            val = fval_grad_inplace_(x_real, grad_opt);
            // 3. Chain Rule: d/dx_opt = d/dx_real * dx_real/dx_opt
            mapGradientToOpt(x_real, grad_opt);
        } else if(use_gradient_) {
            Eigen::VectorXd dummy_grad(x_opt.size());
            val = fval_grad_inplace_(x_real, dummy_grad);
        } else {
            val = fval_only_(x_real);
        }
        return val;
    }

    // Static Callback for NLopt Objective
    static double NLoptCallback(const std::vector<double> &x, std::vector<double> &grad, void *data) {
        ObjectiveFunctor* functor = static_cast<ObjectiveFunctor*>(data);
        Eigen::Map<const Eigen::VectorXd> x_eig(x.data(), x.size());
        Eigen::Map<Eigen::VectorXd> grad_map(grad.data(), grad.size());
        return (*functor)(x_eig, grad_map);
    }

    /**
     * @brief Checks if the functor utilizes gradient information.
     * @return True if gradients are used, false otherwise.
     */
    bool usesGradient() const {
        return use_gradient_;
    }

    // --- Constraints Section ---
    void addInequalityConstraint(std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> g) {
        ineq_constraints_.push_back(std::move(g));
    }

    void addEqualityConstraint(std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> h) {
        eq_constraints_.push_back(std::move(h));
    }

    /**
     * @brief Context struct for NLopt constraint evaluation callback.
     */
    struct ConstraintContext {
        ObjectiveFunctor* functor; ///< Pointer to the parent functor.
        size_t index;              ///< Index of the constraint in the vector.
        bool is_inequality;        ///< True if it is an inequality constraint, false if equality.
    };

    // Unified static wrapper for evaluating constraints with log-scaling awareness
    static double NLoptConstraintWrapper(const std::vector<double> &x, std::vector<double> &grad, void *data) {
        ConstraintContext* ctx = static_cast<ConstraintContext*>(data);
        Eigen::Map<const Eigen::VectorXd> x_opt(x.data(), x.size());
        Eigen::Map<Eigen::VectorXd> grad_opt(grad.data(), grad.size());

        return ctx->functor->evaluateConstraint(ctx->index, ctx->is_inequality, x_opt, grad_opt);
    }

    std::vector<std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)>> getInequalityConstraints() const {
        return ineq_constraints_;
    }

    std::vector<std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)>> getEqualityConstraints() const {
        return eq_constraints_;
    }



  private:
    std::function<double(Eigen::Ref<const Eigen::VectorXd>)> fval_only_;
    std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> fval_grad_inplace_;

    std::vector<std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)>> ineq_constraints_;
    std::vector<std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)>> eq_constraints_;

    bool use_gradient_;
    std::vector<bool> log_scale_; // Mask indicating which dims are log-scaled

    /**
     * @brief Maps parameters from optimization space (potentially log-scaled) to physical space.
     * 
     * @details Mathematical Formulation
     * For dimensions flagged as log-scaled:
     * \f[
     * \theta_{\text{real}, i} = \exp\left( \theta_{\text{opt}, i} \right)
     * \f]
     * For unscaled dimensions:
     * \f[
     * \theta_{\text{real}, i} = \theta_{\text{opt}, i}
     * \f]
     * 
     * @param x_opt Parameter vector in optimization space.
     * @return Parameter vector in real space.
     */
    Eigen::VectorXd mapToReal(Eigen::Ref<const Eigen::VectorXd> x_opt) const {
        if(log_scale_.empty()) return x_opt;
        Eigen::VectorXd x_real = x_opt;
        for(int i = 0; i < x_real.size(); ++i) {
            if(i < (int)log_scale_.size() && log_scale_[i]) {
                x_real(i) = std::exp(x_opt(i));
            }
        }
        return x_real;
    }

    /**
     * @brief Transforms gradients from real space back to optimization space using the chain rule.
     * 
     * @details Mathematical Formulation
     * By the chain rule, the gradient with respect to the optimization parameter is:
     * \f[
     * \frac{\partial f}{\partial \theta_{\text{opt}, i}} = \frac{\partial f}{\partial \theta_{\text{real}, i}} \frac{d \theta_{\text{real}, i}}{d \theta_{\text{opt}, i}} = \frac{\partial f}{\partial \theta_{\text{real}, i}} \theta_{\text{real}, i}
     * \f]
     * 
     * @param x_real Parameter vector in real space.
     * @param grad_real Gradient vector in real space (modified in-place to optimization space gradient).
     */
    void mapGradientToOpt(Eigen::Ref<const Eigen::VectorXd> x_real, Eigen::Ref<Eigen::VectorXd> grad_real) const {
        if(log_scale_.empty() || grad_real.size() == 0) return;
        for(int i = 0; i < grad_real.size(); ++i) {
            if(i < (int)log_scale_.size() && log_scale_[i]) {
                // If x_real = exp(x_opt), then d(x_real)/d(x_opt) = exp(x_opt) = x_real
                grad_real(i) = grad_real(i) * x_real(i);
            }
        }
    }

    /**
     * @brief Evaluates a constraint while properly translating inputs and mapping output gradients.
     * 
     * @param index Index of the constraint in the vector.
     * @param is_ineq True if inequality, false if equality constraint.
     * @param x_opt Input vector in optimization space.
     * @param grad_opt Output gradient vector in optimization space.
     * @return The evaluated constraint value.
     */
    double evaluateConstraint(size_t index, bool is_ineq, Eigen::Ref<const Eigen::VectorXd> x_opt, Eigen::Ref<Eigen::VectorXd> grad_opt) const {
        Eigen::VectorXd x_real = mapToReal(x_opt);
        double val = 0.0;

        const auto& constraint_func = is_ineq ? ineq_constraints_[index] : eq_constraints_[index];

        if(grad_opt.size() > 0) {
            val = constraint_func(x_real, grad_opt);
            mapGradientToOpt(x_real, grad_opt);
        } else {
            Eigen::VectorXd dummy_grad(x_opt.size());
            val = constraint_func(x_real, dummy_grad);
        }
        return val;
    }
};

/**
 * @brief Global helper function to execute NLopt maximization, managing the log-scaling wrapper.
 * Users pass physical, real-world bounds and initial guesses. The function manages the log translations.
 */
double nlopt_max(cmp::ObjectiveFunctor &f, Eigen::Ref<Eigen::VectorXd> x0, const Eigen::Ref<const Eigen::VectorXd> &lb, const Eigen::Ref<const Eigen::VectorXd> &ub, nlopt::algorithm alg = nlopt::LN_SBPLX, double ftol_rel = 1e-6);

} // namespace cmp
/** @} */

