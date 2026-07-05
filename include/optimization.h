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

namespace cmp {

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




    struct ConstraintContext {
        ObjectiveFunctor* functor;
        size_t index;
        bool is_inequality;
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

    // Helper: Map optimizer space x (log) to real physical space x
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

    // Helper: Map real gradient back to optimizer space gradient via Chain Rule
    void mapGradientToOpt(Eigen::Ref<const Eigen::VectorXd> x_real, Eigen::Ref<Eigen::VectorXd> grad_real) const {
        if(log_scale_.empty() || grad_real.size() == 0) return;
        for(int i = 0; i < grad_real.size(); ++i) {
            if(i < (int)log_scale_.size() && log_scale_[i]) {
                // If x_real = exp(x_opt), then d(x_real)/d(x_opt) = exp(x_opt) = x_real
                grad_real(i) = grad_real(i) * x_real(i);
            }
        }
    }

    // Helper to evaluate constraints dynamically while respecting log scaling
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