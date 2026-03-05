/**
 * Test simple optimization functions using NLopt library.
 */

// Standard library includes
#include <cmp_defines.h>
#include <optimization.h>
#include <chrono>

int main() {

    auto objectiveFunction = [](Eigen::Ref<const Eigen::VectorXd> x,
    Eigen::Ref<Eigen::VectorXd> grad) {
        double a = 1.0;
        double b = 100.0;

        // Rosenbrock function (negated for maximization)
        double f = -((a - x[0]) * (a - x[0]) + b * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]));

        if(grad.size() > 0) {
            grad[0] = -(-2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0] * x[0])); // negate gradient too
            grad[1] = -(2.0 * b * (x[1] - x[0] * x[0]));
        }

        return f;
    };

    // Constraint: x0 + x1 <= 1
    auto constraint = [](Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> grad) {
        grad[0] = 1.0;
        grad[1] = 1.0;
        return x[0] + x[1] - 1.0;
    };


    // Initial guess
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;
    std::cout << "Initial guess: " << x0.transpose() << std::endl;
    Eigen::VectorXd grad0 = Eigen::VectorXd::Zero(2);
    std::cout << "Initial objective value: " << objectiveFunction(x0, grad0) << std::endl;

    // Define bounds
    Eigen::VectorXd lb(2);
    lb << -5, -5;
    Eigen::VectorXd ub(2);
    ub << 5, 5;

    // Create the objective functor
    cmp::ObjectiveFunctor functor(objectiveFunction);
    functor.addInequalityConstraint(constraint);

    // Optimize using NLopt and measure time, try the LD_MMA and another constraint-based algorithm i.e.
    std::vector<nlopt::algorithm> algorithms = {nlopt::LD_MMA};
    for(const auto &alg : algorithms) {
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "Using algorithm: " << alg << std::endl;
        Eigen::VectorXd x_opt = x0; // reset to initial guess

        auto start = std::chrono::high_resolution_clock::now();
        cmp::nlopt_max(functor, x_opt, lb, ub, alg, 1e-6);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Optimized parameters: " << x_opt.transpose() << std::endl;
        Eigen::VectorXd grad_opt = Eigen::VectorXd::Zero(2);
        std::cout << "Optimized objective value: " << objectiveFunction(x_opt, grad_opt) << std::endl;
        std::cout << "Gradient at optimum: " << grad_opt.transpose() << std::endl;
        std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
        std::cout << std::endl;
    }

    // Now try another case
    std::cout << "===================================" << std::endl;
    std::cout << "Testing another optimization problem with equality constraint." << std::endl;

    algorithms = {nlopt::LD_SLSQP};

    // Objective gradient: f(x,y) = -exp(x)y^2
    auto objective_grad = [](Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> grad) {
        // Check if gradient is requested
        if(grad.size() > 0) {
            grad[0] = -std::exp(x[0]) * x[1] * x[1];
            grad[1] = -2 * std::exp(x[0]) * x[1];
        }
        return -std::exp(x[0]) * x[1] * x[1];
    };

    // Constraint gradient: x^2 + y^2 - 1 = 0
    auto constraint_grad = [](Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> grad) {
        if(grad.size() > 0) {
            grad[0] = 2 * x[0];
            grad[1] = 2 * x[1];
        }
        return x[0] * x[0] + x[1] * x[1] - 1;
    };


    // Initial guess
    Eigen::VectorXd x1(2);
    x1 << 0.0, 1.0;
    Eigen::VectorXd grad1 = Eigen::VectorXd::Zero(2);
    std::cout << "Initial guess: " << x1.transpose() << std::endl
              << "Initial objective value: " << objective_grad(x1, grad1) << std::endl;
    std::cout << "Initial gradient: " << grad1.transpose() << std::endl
              << "Constraint value at initial guess: " << constraint_grad(x1, grad1) << std::endl;
    std::cout << "Initial constraint gradient: " << grad1.transpose() << std::endl;

    // Create the objective functor
    cmp::ObjectiveFunctor functor2(objective_grad);
    functor2.addEqualityConstraint(constraint_grad);

    // Define bounds
    Eigen::VectorXd lb2(2);
    lb2 << -5, -5;
    Eigen::VectorXd ub2(2);
    ub2 << 5, 5;

    // Optimize using NLopt and measure time
    for(const auto &alg : algorithms) {
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "Using algorithm: " << alg << std::endl;

        Eigen::VectorXd x_opt2 = x1; // reset to initial guess
        auto start2 = std::chrono::high_resolution_clock::now();
        cmp::nlopt_max(functor2, x_opt2, lb2, ub2, alg, 1e-6);
        auto end2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed2 = end2 - start2;
        Eigen::VectorXd grad_opt2 = Eigen::VectorXd::Zero(2);
        std::cout << "Optimized parameters: " << x_opt2.transpose() << std::endl;
        std::cout << "Optimized objective value: " << objective_grad(x_opt2, grad_opt2) << std::endl;
        std::cout << "Gradient at optimum: " << grad_opt2.transpose() << std::endl;
        std::cout << "Constraint value at optimum: " << constraint_grad(x_opt2, grad_opt2) << std::endl;
        std::cout << "Constraint gradient at optimum: " << grad_opt2.transpose() << std::endl;
        std::cout << "Time taken: " << elapsed2.count() << " seconds" << std::endl;
        std::cout << std::endl;
    }


    return 0;
}