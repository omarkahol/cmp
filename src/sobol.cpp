#include <sobol.h>

Eigen::MatrixXd cmp::sobol::SobolSaltelli::evaluationGrid(size_t nObs, std::shared_ptr<cmp::grid::Grid> gridType, bool secondOrder) {
    dim_ = gridType->dim();
    nObs_ = nObs;
    secondOrder_ = secondOrder;
    nTotalObs_ = (2 + dim_ + (secondOrder_ ? dim_ * (dim_ - 1) / 2 : 0)) * nObs_;

    Eigen::MatrixXd allData(nTotalObs_, dim_);

    // --- A and B matrices ---
    Eigen::MatrixXd A = gridType->construct(nObs_);
    Eigen::MatrixXd B = gridType->construct(nObs_);

    allData.topRows(nObs_) = A;         // first nObs rows: A
    allData.middleRows(nObs_, nObs_) = B; // next nObs rows: B

    // --- First-order hybrids (A with one column from B) ---
    for(size_t i = 0; i < dim_; ++i) {
        Eigen::MatrixXd Ai = A;
        Ai.col(i) = B.col(i);
        allData.middleRows((2 + i) * nObs_, nObs_) = Ai;
    }

    // --- Second-order hybrids (A with two columns from B) ---
    if(secondOrder_) {
        size_t count = 0;
        for(size_t i = 0; i < dim_; ++i) {
            for(size_t j = i + 1; j < dim_; ++j) {
                Eigen::MatrixXd Aij = A;
                Aij.col(i) = B.col(i);
                Aij.col(j) = B.col(j);
                allData.middleRows((2 + dim_ + count) * nObs_, nObs_) = Aij;
                count++;
            }
        }
    }

    return allData;
}

Eigen::VectorXd cmp::sobol::SobolSaltelli::sliceSaltelliOutput(const Eigen::VectorXd &Y, const Eigen::VectorXs &idx, size_t nObs, size_t dim, bool secondOrder) {

    size_t nSecondOrder = secondOrder ? dim * (dim - 1) / 2 : 0;
    size_t nTotalObs = Y.size();
    Eigen::VectorXd Yb(nTotalObs);

    // Copy A and B
    Yb.head(nObs) = cmp::slice(Y.head(nObs), idx);                       // Y_A
    Yb.segment(nObs, nObs) = cmp::slice(Y.segment(nObs, nObs), idx);     // Y_B

    // First-order hybrids
    for(size_t i = 0; i < dim; ++i) {
        Yb.segment((2 + i)*nObs, nObs) = cmp::slice(Y.segment((2 + i) * nObs, nObs), idx);
    }

    // Second-order hybrids
    if(secondOrder) {
        for(size_t k = 0; k < nSecondOrder; ++k) {
            Yb.segment((2 + dim + k)*nObs, nObs) = cmp::slice(Y.segment((2 + dim + k) * nObs, nObs), idx);
        }
    }

    return Yb;
}

cmp::sobol::SobolResults cmp::sobol::SobolSaltelli::compute(const Eigen::VectorXd &Y) {
    if(Y.size() != nTotalObs_) {
        throw std::runtime_error("Output vector size does not match the expected total number of observations.");
    }

    SobolResults results;
    results.firstOrder = Eigen::VectorXd::Zero(dim_);
    results.totalOrder = Eigen::VectorXd::Zero(dim_);

    Eigen::Map<const Eigen::VectorXd> YA(Y.data(), nObs_);
    Eigen::Map<const Eigen::VectorXd> YB(Y.data() + nObs_, nObs_);

    // Better - uses all available data
    Eigen::Map<const Eigen::VectorXd> Yall(Y.data(), 2 * nObs_);
    double meanY = Yall.mean();
    double varY = (Yall.array() - meanY).square().sum() / (Yall.size() - 1); // Sample variance


// --- First-order indices (Saltelli 2010) ---
    for(size_t i = 0; i < dim_; ++i) {
        Eigen::Map<const Eigen::VectorXd> YAB(Y.data() + (2 + i) * nObs_, nObs_);

        // Correct first-order formula: E[Y_B * (Y_AB^i - Y_A)] / Var[Y]
        results.firstOrder(i) = (YB.array() * (YAB.array() - YA.array())).mean() / varY;

        // Correct total-order formula: E[(Y_A - Y_AB^i)^2] / (2 * Var[Y])
        results.totalOrder(i) = ((YA.array() - YAB.array()).square().mean()) / (2.0 * varY);
    }

// --- Second-order indices ---
    if(secondOrder_) {
        size_t count = 0;
        size_t nSecondOrder = dim_ * (dim_ - 1) / 2;
        results.secondOrder = Eigen::VectorXd::Zero(nSecondOrder);

        for(size_t i = 0; i < dim_; ++i) {
            for(size_t j = i + 1; j < dim_; ++j) {
                Eigen::Map<const Eigen::VectorXd> YAB2(Y.data() + (2 + dim_ + count) * nObs_, nObs_);

                // Alternative approach if you don't have direct access:
                // S_ij ≈ [E[Y_B * (Y_AB^ij - Y_A)]]/Var[Y] - S_i - S_j
                results.secondOrder(count) = (YB.array() * (YAB2.array() - YA.array())).mean() / varY
                                             - results.firstOrder(i) - results.firstOrder(j);

                count++;
            }
        }
    } else {
        results.secondOrder = Eigen::VectorXd::Zero(0);
    }

    return results;
}

std::pair<cmp::sobol::SobolResults, cmp::sobol::SobolResults> cmp::sobol::SobolSaltelli::computeWithBootstrap(const Eigen::VectorXd &Y, size_t nBootstrap, size_t randomSeed) {

    cmp::statistics::Bootstrap bootstrap(nObs_, nObs_, true, randomSeed);

    SobolResults meanResults;
    meanResults.firstOrder = Eigen::VectorXd::Zero(dim_);
    meanResults.totalOrder = Eigen::VectorXd::Zero(dim_);
    meanResults.secondOrder = secondOrder_ ? Eigen::VectorXd::Zero(dim_ * (dim_ - 1) / 2) : Eigen::VectorXd::Zero(0);

    // Store all bootstrap samples temporarily
    Eigen::MatrixXd S_samples(dim_, nBootstrap);
    Eigen::MatrixXd T_samples(dim_, nBootstrap);
    Eigen::MatrixXd S2_samples;
    if(secondOrder_) S2_samples = Eigen::MatrixXd(dim_ * (dim_ - 1) / 2, nBootstrap);

    for(size_t b = 0; b < nBootstrap; ++b) {
        Eigen::VectorXs sampleIdx = bootstrap();
        Eigen::VectorXd Yb = sliceSaltelliOutput(Y, sampleIdx, nObs_, dim_, secondOrder_);
        SobolResults res = compute(Yb);

        S_samples.col(b) = res.firstOrder;
        T_samples.col(b) = res.totalOrder;
        if(secondOrder_) S2_samples.col(b) = res.secondOrder;
    }

    // Compute mean
    meanResults.firstOrder = S_samples.rowwise().mean();
    meanResults.totalOrder = T_samples.rowwise().mean();
    if(secondOrder_) meanResults.secondOrder = S2_samples.rowwise().mean();

    // Compute variance (sample variance formula)
    SobolResults varResults;
    varResults.firstOrder = ((S_samples.colwise() - meanResults.firstOrder).array().square().rowwise().sum() / (nBootstrap - 1)).matrix();
    varResults.totalOrder = ((T_samples.colwise() - meanResults.totalOrder).array().square().rowwise().sum() / (nBootstrap - 1)).matrix();
    if(secondOrder_) {
        varResults.secondOrder = ((S2_samples.colwise() - meanResults.secondOrder).array().square().rowwise().sum() / (nBootstrap - 1)).matrix();
    }

    return {meanResults, varResults};
}