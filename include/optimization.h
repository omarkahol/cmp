#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include "cmp_defines.h"
#include "gp.h"

namespace cmp {


/**
 * @brief The ObjectiveFunctor class
 * A functor that can be used with NLopt for optimization.
 * It can handle both gradient-free and gradient-based optimization.
 * The gradient-based version writes the gradient in-place to avoid unnecessary allocations.
 * It also provides a static callback function for NLopt.
 *
 * Usage:
 * 1. For gradient-free optimization: Call the constructor with a function of type double f(const Eigen::Ref<const Eigen::VectorXd>&).
 * 2. For gradient-based optimization: Call the constructor with a function of type double f(const Eigen::Ref<const Eigen::VectorXd>&, Eigen::Ref<Eigen::VectorXd>).
 * 3. Use the operator() to evaluate the function and optionally compute the gradient.
 * 4. Use the static NLoptCallback function as the callback for NLopt.
 */
class ObjectiveFunctor {
  public:

    // Gradient-free constructor
    explicit ObjectiveFunctor(std::function<double(Eigen::Ref<const Eigen::VectorXd>)> fval)
        : fval_only_(fval), fval_grad_inplace_(nullptr), use_gradient_(false)
    {}

    // Gradient-based constructor (gradient written in-place)
    explicit ObjectiveFunctor(std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> fval_grad)
        : fval_only_(nullptr), fval_grad_inplace_(fval_grad), use_gradient_(true)
    {}

    // Call operator
    double operator()(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> grad) const {
        if(use_gradient_ && grad.size() > 0) {
            return fval_grad_inplace_(x, grad); // gradient is written in-place
        } else if(use_gradient_) {
            Eigen::VectorXd dummy_grad(x.size());
            return fval_grad_inplace_(x, dummy_grad);
        } else {
            return fval_only_(x);
        }
    }

    // Callback for NLopt
    static double NLoptCallback(const std::vector<double> &x, std::vector<double> &grad, void *data) {

        // Cast the data pointer back to ObjectiveFunctor
        ObjectiveFunctor* functor = static_cast<ObjectiveFunctor*>(data);

        // NLopt uses only double, so we need to cast to double
        Eigen::Map<const Eigen::VectorXd> x_eig(x.data(), x.size());
        Eigen::Map<Eigen::VectorXd> grad_map(grad.data(), grad.size());

        return (*functor)(x_eig, grad_map);
    }

    const std::vector<std::function<double(Eigen::Ref<const Eigen::VectorXd>,
    Eigen::Ref<Eigen::VectorXd>)>>& getIneqConstraints() const {
        return ineq_constraints_;
    }

    const std::vector<std::function<double(Eigen::Ref<const Eigen::VectorXd>,
    Eigen::Ref<Eigen::VectorXd>)>>& getEqConstraints() const {
        return eq_constraints_;
    }

    bool usesGradient() const {
        return use_gradient_;
    }

    // Add an inequality constraint: g(x) <= 0
    void addInequalityConstraint(
        std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> g) {
        ineq_constraints_.push_back(g);
    }

    // Add an equality constraint: h(x) = 0
    void addEqualityConstraint(
        std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> h) {
        eq_constraints_.push_back(h);
    }

    // Define the wrapper once
    static double NLoptConstraintWrapper(const std::vector<double> &x, std::vector<double> &grad, void *data) {
        auto* g = static_cast <std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)>* > (data);

        Eigen::Map<const Eigen::VectorXd> x_eig(x.data(), x.size());
        Eigen::Map<Eigen::VectorXd> grad_map(grad.data(), grad.size());
        return (*g)(x_eig, grad_map);
    }



  private:

    // Function for value only
    std::function<double(Eigen::Ref<const Eigen::VectorXd>)> fval_only_;

    // Function for value and gradient (gradient written in-place)
    std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)> fval_grad_inplace_;

    // Inequality constraints
    std::vector<std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)>> ineq_constraints_;

    // Equality constraints
    std::vector<std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)>> eq_constraints_;

    // Flag to indicate if gradient is used
    bool use_gradient_;
};

double nlopt_max(ObjectiveFunctor &f, Eigen::Ref<Eigen::VectorXd> x0, const Eigen::Ref<const Eigen::VectorXd> &lb, const Eigen::Ref<const Eigen::VectorXd> &ub, nlopt::algorithm alg = nlopt::LN_SBPLX, double ftol_rel = 1e-6);


}


#endif