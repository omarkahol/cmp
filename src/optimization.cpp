#include "optimization.h"

double cmp::nlopt_max(cmp::ObjectiveFunctor &f, Eigen::Ref<Eigen::VectorXd> x0, const Eigen::Ref<const Eigen::VectorXd> &lb, const Eigen::Ref<const Eigen::VectorXd> &ub, nlopt::algorithm alg, double ftol_rel) {

    // 1. Transform bounds and initial guess into optimizer space
    std::vector<double> x_std(x0.size());
    std::vector<double> lb_std(lb.size());
    std::vector<double> ub_std(ub.size());

    const auto& log_mask = f.getLogScale();

    for(size_t i = 0; i < x0.size(); ++i) {
        bool is_log = (i < log_mask.size() && log_mask[i]);
        x_std[i]  = is_log ? std::log(x0(i)) : x0(i);
        lb_std[i] = is_log ? std::log(lb(i)) : lb(i);
        ub_std[i] = is_log ? std::log(ub(i)) : ub(i);
    }

    nlopt::opt opt(alg, x0.size());
    opt.set_max_objective(cmp::ObjectiveFunctor::NLoptCallback, &f);
    opt.set_lower_bounds(lb_std);
    opt.set_upper_bounds(ub_std);
    opt.set_ftol_rel(ftol_rel);

    // 2. Store the ConstraintContexts so they live until optimization finishes
    std::vector<cmp::ObjectiveFunctor::ConstraintContext> ieq_ctx;
    for(size_t i = 0; i < f.getInequalityConstraints().size(); ++i) {
        ieq_ctx.push_back({&f, i, true});
        opt.add_inequality_constraint(&cmp::ObjectiveFunctor::NLoptConstraintWrapper, &ieq_ctx.back(), 1e-8);
    }

    std::vector<cmp::ObjectiveFunctor::ConstraintContext> eq_ctx;
    for(size_t i = 0; i < f.getEqualityConstraints().size(); ++i) {
        eq_ctx.push_back({&f, i, false});
        opt.add_equality_constraint(&cmp::ObjectiveFunctor::NLoptConstraintWrapper, &eq_ctx.back(), 1e-8);
    }

    // 3. Optimize
    double fval = 0.0;
    try {
        opt.optimize(x_std, fval);
    } catch(std::runtime_error &e) {
        throw std::runtime_error(std::string("NLopt optimization failed: ") + e.what());
    }

    // 4. Transform optimized result back to real physical space for the user
    for(size_t i = 0; i < x0.size(); ++i) {
        bool is_log = (i < log_mask.size() && log_mask[i]);
        x0[i] = is_log ? std::exp(x_std[i]) : x_std[i];
    }

    return fval;
}