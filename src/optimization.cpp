#include "optimization.h"

double cmp::nlopt_max(ObjectiveFunctor &f, Eigen::Ref<Eigen::VectorXd> x0, const Eigen::Ref<const Eigen::VectorXd> &lb, const Eigen::Ref<const Eigen::VectorXd> &ub, nlopt::algorithm alg, double ftol_rel) {

    // Convert to std::vector<double> for nlopt
    std::vector<double> x_std(x0.data(), x0.data() + x0.size());
    std::vector<double> lb_std(lb.data(), lb.data() + lb.size());
    std::vector<double> ub_std(ub.data(), ub.data() + ub.size());

    nlopt::opt opt(alg, x0.size());
    opt.set_max_objective(ObjectiveFunctor::NLoptCallback, &f);
    opt.set_lower_bounds(lb_std);
    opt.set_upper_bounds(ub_std);
    opt.set_ftol_rel(ftol_rel);

    // Store the actual constraints in a vector that lives until optimization finishes
    std::vector<std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)>> ieq_constraint_storage;
    for(auto &g : f.getIneqConstraints()) {
        ieq_constraint_storage.push_back(g); // store a copy
        opt.add_inequality_constraint(&ObjectiveFunctor::NLoptConstraintWrapper, &ieq_constraint_storage.back(), 1e-8);
    }

    std::vector<std::function<double(Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>)>> eq_constraint_storage;
    for(auto &h : f.getEqConstraints()) {
        eq_constraint_storage.push_back(h);
        opt.add_equality_constraint(&ObjectiveFunctor::NLoptConstraintWrapper, &eq_constraint_storage.back(), 1e-8);
    }

    double fval = 0.0;
    try {
        opt.optimize(x_std, fval);
    } catch(std::runtime_error &e) {
        std::cerr << "NLopt failed: " << e.what() << std::endl;
    }

    // Copy back to Eigen
    for(size_t i = 0; i < x0.size(); ++i) {
        x0[i] = x_std[i];
    }
    return fval;
}