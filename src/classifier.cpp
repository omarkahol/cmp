#include "classifier.h"

void cmp::classifier::KNN::compute(const std::vector<Eigen::VectorXd> &points, size_t k) {
    // Set the number of points and the number of neighbors
    nPoints_ = points.size();
    kNearestValue_ = k;

    // Initialize the neighbors
    neighbours_ = std::vector<std::vector<size_t>>(nPoints_, std::vector<size_t>(kNearestValue_, 0));

    // Iterate through the points
    for(size_t i = 0; i < nPoints_; i++) {

        // Current point
        Eigen::VectorXd point = points[i];

        // Create a linked list for easy storing of the neighbors
        std::list<std::pair<double, size_t>> neighbors;

        for(size_t j = 0; j < nPoints_; j++) {

            // Skip the current point because it is at a distance of 0 from itself
            if(i == j) {
                continue; // Go to the next point
            }

            // Compute the distance
            double dist = (point - points[j]).norm();

            // Check if the list is empty
            // If it is, just add the point
            if(neighbors.empty()) {
                neighbors.push_back(std::make_pair(dist, j));
                continue; // Go to the next point
            }

            // Now we check if we can insert the point in the list
            bool inserted = false; // Flag to check if the point was inserted
            for(auto it = neighbors.begin(); it != neighbors.end(); it++) {

                // If point j is closer than the current point, insert it
                if(dist < it->first) {
                    neighbors.insert(it, std::make_pair(dist, j)); // Insert it before the current point
                    inserted = true;
                    break; // Go to the next point (exit the iterator loop)
                }
            }

            // If the point was not inserted, check if the list is not full
            if(neighbors.size() < kNearestValue_ && !inserted) {
                neighbors.push_back(std::make_pair(dist, j)); // Add the point to the end of the list
                continue;
            }

            // Check if the list is full
            // If the list is full, remove the last element
            if(neighbors.size() > kNearestValue_) {
                neighbors.pop_back();
                continue;
            }
        }

        // Save the neighbors
        auto iterator = neighbors.begin();
        for(size_t j = 0; j < kNearestValue_; j++) {
            neighbours_[i][j] = iterator->second;
            iterator++;
        }
    }
}

const std::vector<size_t> &cmp::classifier::KNN::operator[](size_t i) const {
    return neighbours_[i];
}

size_t cmp::classifier::KNN::nPoints() const {
    return nPoints_;
}

size_t cmp::classifier::KNN::k() const {
    return kNearestValue_;
}


/**
 * Functions for the KDE classifier
 */

double cmp::classifier::KDE::density(const Eigen::Ref<const Eigen::VectorXd> &x, const size_t &classLabel) const  {
    double density = 0.0;
    size_t class_count = 0;
    for(size_t i = 0; i < nObs_; i++) {
        if(labels_[i] == classLabel) {
            Eigen::VectorXd z = bandwidth_->apply(xObs_.row(i).transpose(), x);
            density += kernel_->eval(z);
            class_count++;
        }
    }

    // Return a normalized density
    return density * bandwidth_->determinant() * kernel_->normalizationConstant(dimX_) / double(class_count);
}

std::vector<double> cmp::classifier::KDE::predictProbabilities(const Eigen::Ref<const Eigen::VectorXd>& x) const  {

    std::vector<double> probs(nClasses_, 0.0);
    double sum = 0.0;
    for(size_t i = 0; i < nObs_; i++) {
        Eigen::VectorXd z = bandwidth_->apply(xObs_.row(i).transpose(), x);
        double prod = kernel_->eval(z) * bandwidth_->determinant() / double(classCounts_(labels_(i)));
        probs[labels_[i]] += prod;
        sum += prod;
    }

    // Normalize the probabilities
    for(size_t j = 0; j < nClasses_; j++) {
        probs[j] /= sum;
    }

    return probs;
}

double cmp::classifier::KDE::objectiveFunctionCV(const method& method, const cmp::statistics::KFold& kf) const {


    double score = 0.0;
    for(auto split : kf) {

        // Split the data
        auto [xTrain, yTrain, xTest, yTest] = cmp::trainTestSplit(xObs_, labels_, split);

        // Count the number of samples per class in the training set
        std::vector<size_t> class_counts(nClasses_, 0);
        for(size_t i = 0; i < yTrain.size(); i++) {
            class_counts[yTrain(i)]++;
        }

        // For each point we compute the score
        double fold_score = 0.0;
        for(size_t i = 0; i < xTest.rows(); i++) {

            // Compute the probabilities of xTest.row(i)
            std::vector<double> probs(nClasses_, 0.0);
            double sum = 0.0;
            for(size_t j = 0; j < xTrain.rows(); j++) {
                Eigen::VectorXd z = bandwidth_->apply(xTrain.row(j).transpose(), xTest.row(i).transpose());
                double prod = kernel_->eval(z) * bandwidth_->determinant() / double(class_counts[yTrain(j)]);
                probs[yTrain(j)] += prod;
                sum += prod;
            }

            if(method == CV_SCORE) {

                // Get the best class and check if it is correct
                size_t pred = std::max_element(probs.begin(), probs.end()) - probs.begin();
                if(pred == yTest(i)) {
                    fold_score += 1.0;
                }

            } else if(method == CV_PROB_SCORE) {
                // Add the probability of the true class
                fold_score += std::log(probs[yTest(i)] / sum + cmp::TOL); // Add a small value to avoid log(0)
            }
        }

        // Average the score over the number of test samples
        fold_score /= double(xTest.rows());
        score += fold_score;
    }
    score /= double(kf.nFolds());
    return score;
}

double cmp::classifier::KDE::objectiveFunctionLOO() const {

    // LOO is undefined if any class has a single sample.
    for(size_t c = 0; c < nClasses_; ++c) {
        if(classCounts_(c) <= 1) {
            return -std::numeric_limits<double>::infinity();
        }
    }

    double score = 0.0;
    for(size_t i = 0; i < nObs_; i++) {
        std::vector<double> probs(nClasses_, 0.0);
        double sum = 0.0;

        for(size_t j = 0; j < nObs_; j++) {
            if(i == j) {
                continue;
            }

            const size_t cls = labels_(j);
            const size_t classCountLOO = classCounts_(cls) - ((labels_(i) == cls) ? 1 : 0);
            if(classCountLOO == 0) {
                continue;
            }

            Eigen::VectorXd z = bandwidth_->apply(xObs_.row(j).transpose(), xObs_.row(i).transpose());
            double prod = kernel_->eval(z) * bandwidth_->determinant() / double(classCountLOO);
            probs[cls] += prod;
            sum += prod;
        }

        score += std::log(probs[labels_(i)] / sum + cmp::TOL);
    }

    return score / double(nObs_);
}

void cmp::classifier::KDE::fit(const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& labels, cmp::statistics::KFold kf, const double& minBw, const double& maxBw, const method method, nlopt::algorithm algo, double ftol_rel) {

    // First we condition on the data
    this->condition(xObs, labels);

    // Prepare the optimization
    Eigen::VectorXd x0 = bandwidth_->getParams();
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(x0.size(), minBw);
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(x0.size(), maxBw);

    // Function to be optimized
    cmp::ObjectiveFunctor objectiveFunctor(
    [&](const Eigen::Ref<const Eigen::VectorXd>& params, Eigen::Ref<Eigen::VectorXd> grad) -> double {
        this->bandwidth_->setFromVector(params);
        return objectiveFunctionCV(method, kf);
    });

    // Call the optimizer: it will modify the bandwidth parameters in place
    cmp::nlopt_max(objectiveFunctor, x0, lb, ub, algo, ftol_rel);
}

void cmp::classifier::KDE::fitLOO(const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& labels, const double& minBw, const double& maxBw, nlopt::algorithm algo, double ftol_rel) {

    // Condition FIRST to set up the data
    this->condition(xObs, labels);

    // Prepare the optimization
    Eigen::VectorXd x0 = bandwidth_->getParams();
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(x0.size(), minBw);
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(x0.size(), maxBw);

    // Function to be optimized
    cmp::ObjectiveFunctor objectiveFunctor(
    [&](const Eigen::Ref<const Eigen::VectorXd>& params, Eigen::Ref<Eigen::VectorXd> grad) -> double {
        this->bandwidth_->setFromVector(params);
        return objectiveFunctionLOO();
    }
    );

    // Call the optimizer: it will modify the bandwidth parameters in place
    cmp::nlopt_max(objectiveFunctor, x0, lb, ub, algo, ftol_rel);
}


/**
 * Functions for the SVM classifier
 */

void cmp::classifier::SVM::condition(const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& labels) {

    // Check if we need to free an existing model
    if(model_ != nullptr) {
        svm_free_and_destroy_model(&model_);
        freeProblem(&prob_);
    }

    nObs_ = xObs.rows();
    xObs_ = xObs;

    // Create the problem
    prob_.l = nObs_;
    prob_.y = new double[prob_.l];
    prob_.x = new svm_node*[prob_.l];

    for(size_t i = 0; i < nObs_; i++) {
        prob_.y[i] = labels[i];

        // Each sample has row ID + nObs kernel entries + terminator
        prob_.x[i] = new svm_node[nObs_ + 2];

        // Row ID (mandatory, 1-based)
        prob_.x[i][0].index = 0;
        prob_.x[i][0].value = i + 1;

        // Fill kernel row on-the-fly
        for(size_t j = 0; j < nObs_; j++) {
            prob_.x[i][j + 1].index = j + 1;
            prob_.x[i][j + 1].value = covariance_->eval(xObs_.row(i), xObs_.row(j), hyperparameters_);

        }

        // End marker
        prob_.x[i][nObs_ + 1].index = -1;
    }

    // Train
    model_ = svm_train(&prob_, &modelParameters_);
}

size_t cmp::classifier::SVM::predict(const Eigen::Ref<const Eigen::VectorXd>& x) const {
    svm_node *test = new svm_node[nObs_ + 2];

    // Row ID (arbitrary, e.g. 1)
    test[0].index = 0;
    test[0].value = 1;

    // Kernel values against training samples
    for(size_t j = 0; j < nObs_; j++) {
        test[j + 1].index = j + 1;
        test[j + 1].value = covariance_->eval(x, xObs_.row(j), hyperparameters_);
    }

    test[nObs_ + 1].index = -1;
    return svm_predict(model_, test);
}

std::vector<double> cmp::classifier::SVM::predictProbabilities(const Eigen::Ref<const Eigen::VectorXd>& x) const  {

    svm_node *test = new svm_node[nObs_ + 2];

    // Row ID (arbitrary, e.g. 1)
    test[0].index = 0;
    test[0].value = 1;

    // Kernel values against training samples
    for(size_t j = 0; j < nObs_; j++) {
        test[j + 1].index = j + 1;
        test[j + 1].value = covariance_->eval(x, xObs_.row(j), hyperparameters_);
    }

    test[nObs_ + 1].index = -1;

    // Allocate memory for the probabilities
    std::vector<double> prob(model_->nr_class, 0.0);
    svm_predict_probability(model_, test, prob.data());

    // Switch the probabilities according to the class labels
    std::vector<double> probSwitched(model_->nr_class, 0.0);
    for(int i = 0; i < model_->nr_class; ++i) {
        probSwitched[model_->label[i]] = prob[i];
    }
    return probSwitched;
}

void cmp::classifier::SVM::fit(const method& method, const cmp::statistics::KFold& kf,
                               const Eigen::Ref<const Eigen::MatrixXd>& xObs,
                               const Eigen::Ref<const Eigen::VectorXs>& membershipTable,
                               Eigen::VectorXd lb, Eigen::VectorXd ub, nlopt::algorithm algo,
                               double ftol_rel) {

    // Initial guess
    Eigen::VectorXd x0(hyperparameters_.size() + 1);
    x0 << hyperparameters_, modelParameters_.C;

    cmp::ObjectiveFunctor objectiveFunctor(
    [&](const Eigen::Ref<const Eigen::VectorXd>& x) -> double {

        // Extract the hyperparameters and C
        Eigen::VectorXd hyperparameters = x.head(x.size() - 1);
        double C = x(x.size() - 1);

        // Set the parameters and condition on the data
        this->set(covariance_, hyperparameters, C, this->modelParameters_.eps);
        this->condition(xObs, membershipTable);

        // Evaluate the objective function
        return objectiveFunctionCV(method, kf);
    }
    );

    // Maximize the objective function
    cmp::nlopt_max(objectiveFunctor, x0, lb, ub, algo, ftol_rel);
}

void cmp::classifier::SVM::fit(Eigen::Ref<const Eigen::MatrixXd> xObs,
                               Eigen::Ref<const Eigen::VectorXs> membershipTable,
                               Eigen::Ref<const Eigen::VectorXd> lb,
                               Eigen::Ref<const Eigen::VectorXd> ub, nlopt::algorithm algo,
                               double ftol_rel)  {

    // Initial guess
    Eigen::VectorXd x0(hyperparameters_.size() + 1);
    x0 << hyperparameters_, modelParameters_.C;

    cmp::ObjectiveFunctor objectiveFunctor(
    [&](const Eigen::Ref<const Eigen::VectorXd>& x) -> double {

        // Extract the hyperparameters and C
        Eigen::VectorXd hyperparameters = x.head(x.size() - 1);
        double C = x(x.size() - 1);

        // Set the parameters and condition on the data
        this->set(covariance_, hyperparameters, C, this->modelParameters_.eps);
        this->condition(xObs, membershipTable);

        // Evaluate the objective function
        return objectiveFunctionSpan();
    }
    );

    // Maximize the objective function
    cmp::nlopt_max(objectiveFunctor, x0, lb, ub, algo, ftol_rel);
};

double cmp::classifier::SVM::objectiveFunctionCV(const method& method,
                                                 const cmp::statistics::KFold& kf) {
    // If we are interested in the CV score, we can use LIBSVM's built-in cross-validation function (faster).
    if(method == CV_SCORE) {

        // Create a temporary copy of the problem
        struct svm_problem probTmp;
        probTmp.l = prob_.l;
        probTmp.y = new double[probTmp.l];
        probTmp.x = new svm_node*[probTmp.l];

        for(size_t i = 0; i < probTmp.l; i++) {
            probTmp.y[i] = prob_.y[i];

            // Each sample has row ID + nObs kernel entries + terminator
            probTmp.x[i] = new svm_node[nObs_ + 2];

            // Row ID (mandatory, 1-based)
            probTmp.x[i][0].index = 0;
            probTmp.x[i][0].value = i + 1;

            // Fill kernel row on-the-fly
            for(size_t j = 0; j < nObs_; j++) {
                probTmp.x[i][j + 1].index = j + 1;
                probTmp.x[i][j + 1].value = prob_.x[i][j + 1].value;

            }

            // End marker
            probTmp.x[i][nObs_ + 1].index = -1;
        }

        // Allocate memory for the cross-validation results
        std::vector<double> target(nObs_, 0.0);
        svm_cross_validation(&probTmp, &modelParameters_, kf.nFolds(), target.data());

        // Free the memory
        freeProblem(&probTmp);

        size_t correct = 0;
        for(size_t i = 0; i < nObs_; ++i) {
            if(target[i] == prob_.y[i]) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / nObs_;
    } else {

        double score = 0.0;
        for(auto [trainIdx, testIdx] : kf) {

            // Create a temporary copy of the problem for the training set
            svm_problem probTmp;
            probTmp.l = trainIdx.size();
            probTmp.y = new double[probTmp.l];
            probTmp.x = new svm_node*[probTmp.l];

            // Fill the training problem
            for(size_t i = 0; i < trainIdx.size(); ++i) {

                // I am getting the necessary rows from the full kernel matrix using the trainIdx mapping
                probTmp.y[i] = prob_.y[trainIdx[i]];
                probTmp.x[i] = new svm_node[trainIdx.size() + 2];
                probTmp.x[i][0].index = 0;
                probTmp.x[i][0].value = i + 1;
                for(size_t j = 0; j < trainIdx.size(); ++j) {
                    probTmp.x[i][j + 1].index = j + 1;
                    probTmp.x[i][j + 1].value = prob_.x[trainIdx[i]][trainIdx[j] + 1].value;
                }
                probTmp.x[i][trainIdx.size() + 1].index = -1;
            }

            // Create and train a temporary model
            svm_model* modelTmp = svm_train(&probTmp, &modelParameters_);

            for(size_t i = 0; i < testIdx.size(); ++i) {
                svm_node* x = new svm_node[trainIdx.size() + 2];
                x[0].index = 0;
                x[0].value = testIdx[i] + 1;
                for(size_t j = 0; j < trainIdx.size(); ++j) {
                    x[j + 1].index = j + 1;
                    x[j + 1].value = prob_.x[testIdx[i]][trainIdx[j] + 1].value;
                }
                x[trainIdx.size() + 1].index = -1;

                std::vector<double> probEstimates(modelTmp->nr_class);
                svm_predict_probability(modelTmp, x, probEstimates.data());

                int trueLabel = static_cast<int>(prob_.y[testIdx[i]]);
                int labelIdx = -1;
                for(int k = 0; k < modelTmp->nr_class; ++k)
                    if(modelTmp->label[k] == trueLabel)
                        labelIdx = k;

                if(labelIdx != -1)
                    score += std::log(std::max(probEstimates[labelIdx], cmp::TOL));

                // Free the memory
                delete[] x;
            }

            // Free the temporary model and problem
            svm_free_and_destroy_model(&modelTmp);
            freeProblem(&probTmp);
        }
        return score / double(nObs_);
    }
};

double cmp::classifier::SVM::objectiveFunctionSpan() {
    if(model_ == nullptr || model_->l <= 1 || covariance_ == nullptr) {
        return -std::numeric_limits<double>::infinity();
    }

    const double C = modelParameters_.C;
    const double alphaTol = std::max(1e-8, modelParameters_.eps);

    const int nClass = model_->nr_class;
    const int nSVTot = model_->l;
    if(nClass < 2 || nSVTot <= 1) {
        return -std::numeric_limits<double>::infinity();
    }

    // libsvm keeps support vectors grouped by class; build group offsets.
    std::vector<int> svStart(nClass + 1, 0);
    for(int c = 0; c < nClass; ++c) {
        svStart[c + 1] = svStart[c] + model_->nSV[c];
    }

    auto pairContribution = [&](int ci, int cj) -> std::pair<double, int> {
        std::vector<int> pairSV;
        std::vector<double> pairAlpha;
        std::vector<double> pairSign;
        pairSV.reserve(model_->nSV[ci] + model_->nSV[cj]);
        pairAlpha.reserve(model_->nSV[ci] + model_->nSV[cj]);
        pairSign.reserve(model_->nSV[ci] + model_->nSV[cj]);

        // For one-vs-one(ci,cj), libsvm stores coefficients:
        // class ci SVs in row (cj-1), class cj SVs in row ci.
        for(int s = svStart[ci]; s < svStart[ci + 1]; ++s) {
            const double a = std::abs(model_->sv_coef[cj - 1][s]);
            if(a > alphaTol && a < (C - alphaTol)) {
                pairSV.push_back(s);
                pairAlpha.push_back(a);
                pairSign.push_back(+1.0);
            }
        }
        for(int s = svStart[cj]; s < svStart[cj + 1]; ++s) {
            const double a = std::abs(model_->sv_coef[ci][s]);
            if(a > alphaTol && a < (C - alphaTol)) {
                pairSV.push_back(s);
                pairAlpha.push_back(a);
                pairSign.push_back(-1.0);
            }
        }

        // Fallback: if no free pair-SV, use all active pair-SV.
        if(pairSV.empty()) {
            for(int s = svStart[ci]; s < svStart[ci + 1]; ++s) {
                const double a = std::abs(model_->sv_coef[cj - 1][s]);
                if(a > alphaTol) {
                    pairSV.push_back(s);
                    pairAlpha.push_back(a);
                    pairSign.push_back(+1.0);
                }
            }
            for(int s = svStart[cj]; s < svStart[cj + 1]; ++s) {
                const double a = std::abs(model_->sv_coef[ci][s]);
                if(a > alphaTol) {
                    pairSV.push_back(s);
                    pairAlpha.push_back(a);
                    pairSign.push_back(-1.0);
                }
            }
        }

        const int m = static_cast<int>(pairSV.size());
        if(m <= 1) {
            return {0.0, 0};
        }

        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(m, m);
        for(int a = 0; a < m; ++a) {
            const int svA = pairSV[a];
            const int idxA = model_->sv_indices[svA] - 1;
            for(int b = a; b < m; ++b) {
                const int svB = pairSV[b];
                const int idxB = model_->sv_indices[svB] - 1;
                const double kij = covariance_->eval(xObs_.row(idxA).transpose(), xObs_.row(idxB).transpose(), hyperparameters_);
                const double qij = pairSign[a] * pairSign[b] * kij;
                Q(a, b) = qij;
                Q(b, a) = qij;
            }
        }

        const double avgDiag = std::max(Q.diagonal().mean(), cmp::TOL);
        Q.diagonal().array() += 1e-8 * avgDiag;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(Q);
        if(ldlt.info() != Eigen::Success) {
            return {0.0, 0};
        }

        const Eigen::MatrixXd invQ = ldlt.solve(Eigen::MatrixXd::Identity(m, m));
        if(ldlt.info() != Eigen::Success) {
            return {0.0, 0};
        }

        double weightedSpan = 0.0;
        for(int i = 0; i < m; ++i) {
            const double invDiag = std::max(invQ(i, i), cmp::TOL);
            const double spanSq = 1.0 / invDiag;
            weightedSpan += pairAlpha[i] * std::max(spanSq, 0.0);
        }
        weightedSpan /= static_cast<double>(m);
        return {weightedSpan, m};
    };

    double aggregate = 0.0;
    int aggregateWeight = 0;
    for(int ci = 0; ci < nClass; ++ci) {
        for(int cj = ci + 1; cj < nClass; ++cj) {
            auto [pairSpan, weight] = pairContribution(ci, cj);
            if(weight > 0) {
                aggregate += pairSpan * static_cast<double>(weight);
                aggregateWeight += weight;
            }
        }
    }

    if(aggregateWeight == 0) {
        return -std::numeric_limits<double>::infinity();
    }

    const double meanPairwiseSpan = aggregate / static_cast<double>(aggregateWeight);
    return -meanPairwiseSpan;
};

void cmp::classifier::Dummy::condition(const Eigen::Ref<const Eigen::MatrixXd>& xObs, const Eigen::Ref<const Eigen::VectorXs>& labels) {

    // Count the number of classes
    nClasses_ = *std::max_element(labels.begin(), labels.end()) + 1;

}

size_t cmp::classifier::Dummy::predict(const Eigen::Ref<const Eigen::VectorXd>& x) const {
    return 0;
}

std::vector<double> cmp::classifier::Dummy::predictProbabilities(
    const Eigen::Ref<const Eigen::VectorXd>& x) const {
    return std::vector<double>(nClasses_, 1.0 / double(nClasses_));
}
