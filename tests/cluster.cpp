#include <cmp_defines.h>
#include <cluster.h>
#include <matplotlibcpp.h>
#include <distribution.h>

namespace plt = matplotlibcpp;

// Function to generate "make_moons" dataset
std::pair<Eigen::MatrixXd, Eigen::VectorXs> make_moons(int n_samples, double noise, unsigned int seed = 42) {

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n_samples, 2);
    Eigen::VectorXs y = Eigen::VectorXs::Zero(n_samples);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, M_PI);
    std::normal_distribution<double> noise_dist(0.0, noise);

    int half = n_samples / 2;

    // First moon
    for(int i = 0; i < half; ++i) {
        double t = dist(gen);
        X(i, 0) = std::cos(t) + noise_dist(gen);
        X(i, 1) = std::sin(t) + noise_dist(gen);
        y(i) = 0;
    }

    // Second moon
    for(int i = 0; i < n_samples - half; ++i) {
        double t = dist(gen);
        X(half + i, 0) = 1.0 - std::cos(t) + noise_dist(gen);
        X(half + i, 1) = -std::sin(t) - 0.5 + noise_dist(gen);
        y(half + i) = 1;
    }
    return {X, y};
}

int main() {

    // Prepare colors (tab20)
    std::vector<std::string> colors = {
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
        "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "tab:lime", "tab:teal",
        "tab:indigo", "tab:gold", "tab:darkred", "tab:darkgreen", "tab:darkblue",
        "tab:darkorange", "tab:magenta", "tab:yellow"
    };

    std::default_random_engine rng(42);

    // Create mixture of 2D gaussians
    int n_samples = 300;
    auto dist_1 = cmp::distribution::MultivariateNormalDistribution(Eigen::Vector2d::Zero(), 0.01 * Eigen::Matrix2d::Identity());
    auto dist_2 = cmp::distribution::MultivariateNormalDistribution(Eigen::Vector2d::Ones(), 0.01 * Eigen::Matrix2d::Identity());
    auto dist_3 = cmp::distribution::MultivariateNormalDistribution(-Eigen::Vector2d::Ones(), 0.01 * Eigen::Matrix2d::Identity());
    cmp::distribution::MultivariateMixtureDistribution dist({
        std::make_shared<cmp::distribution::MultivariateNormalDistribution>(dist_1),
        std::make_shared<cmp::distribution::MultivariateNormalDistribution>(dist_2),
        std::make_shared<cmp::distribution::MultivariateNormalDistribution>(dist_3)

    }, {1, 1, 1});

    //Eigen::MatrixXd X = Eigen::MatrixXd(n_samples, 2);

    //for(size_t i = 0; i < n_samples; i++) {
    //    X.row(i) = dist.sample(rng);
    //}

    auto [X, _] = make_moons(n_samples / 2, 0.1, 42);


    // Simple KMeans init with 2 clusters
    size_t n_init_clusters = 10;
    cmp::cluster::GeometricCluster kmeans;
    kmeans.fit(X, n_init_clusters, rng);

    Eigen::VectorXs init_labels = kmeans.getLabels();
    size_t n_clusters = kmeans.nClusters();

    // Plot the initial labels
    for(size_t lab = 0; lab < n_clusters; lab++) {
        std::vector<double> xs, ys;
        for(int i = 0; i < X.rows(); ++i) {
            if(init_labels(i) == lab) {
                xs.push_back(X(i, 0));
                ys.push_back(X(i, 1));
            }
        }
        plt::scatter(xs, ys, 20.0, {{"color", colors[lab % colors.size()]}, {"label", std::to_string(lab)}});
    }

    plt::title("Initial k means clustering on moon dataset");
    plt::legend();
    plt::show();

    // Set DPMM hyperparameters
    cmp::distribution::NormalInverseWishartDistribution hyper(X.colwise().mean().transpose(), 0.1, 4.0, Eigen::Matrix2d::Identity());

    double alpha = 1.0;

    cmp::cluster::DirichletProcessMixtureModel dpmm(alpha, hyper, 42);

    dpmm.condition(X, init_labels);

    size_t n_iters = 100;
    for(size_t iter = 0; iter < n_iters; iter++) {
        dpmm.step();
    }

    // After running dpmm.run_gibbs(...):
    Eigen::VectorXs labels = dpmm.getLabels();

    std::cout << "DPMM found " << dpmm.nClusters() << " clusters." << std::endl;
    std::cout << "Final labels:\n" << labels.transpose() << std::endl;

// Find unique clusters
    std::vector<size_t> unique_labels;
    std::map<size_t, int> label_to_color;
    for(int i = 0; i < labels.size(); ++i) {
        size_t lab = labels(i);
        if(label_to_color.find(lab) == label_to_color.end()) {
            label_to_color[lab] = static_cast<int>(label_to_color.size());
            unique_labels.push_back(lab);
        }
    }

// Scatter by cluster
    for(size_t lab : unique_labels) {
        std::vector<double> xs, ys;
        for(int i = 0; i < X.rows(); ++i) {
            if(labels(i) == lab) {
                xs.push_back(X(i, 0));
                ys.push_back(X(i, 1));
            }
        }
        plt::scatter(xs, ys, 20.0, {{"color", colors[label_to_color[lab] % colors.size()]}, {"label", std::to_string(lab)}});
    }

    plt::title("DPMM clustering on moon dataset");
    plt::legend();
    plt::show();

    return 0;

}
