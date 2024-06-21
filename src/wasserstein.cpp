#include <wasserstein.h>

using namespace cmp;


double cmp::wasserstein_1d(std::vector<double> &samples_1, std::vector<double> &samples_2, const double &p) {
        
        // Size of the data
        int N = samples_1.size();

        // Sort the arrays
        std::sort(samples_1.begin(), samples_1.end());
        std::sort(samples_2.begin(), samples_2.end());

        // Integrate the quantile function
        double dist = 0.0;
        for(int i=0; i<N; i++) {
            dist += std::pow(std::abs(samples_1[i]-samples_2[i]),p);
        }

        return dist/double(N);

    };

    vector_t cmp::hypersphere_sample(int dim, std::normal_distribution<double> &dist_n, std::default_random_engine &rng) {
        
        vector_t sample(dim);
        for (int i=0; i<dim; i++) {
            sample(i) = dist_n(rng);
        }
        
        sample.normalize();

        return sample;
    }

    double cmp::sliced_wasserstein(const std::vector<vector_t> &samples_1, const std::vector<vector_t> &samples_2, const double &p, const int &slices, std::normal_distribution<double> &dist_n, std::default_random_engine &rng) {
        
        int dim = samples_1[0].size();
        double V_n = 2*std::pow(M_PI,dim/2.0)/std::tgamma(dim/2.0);
        int n_samples = samples_1.size();

        double dist = 0.0;
        std::vector<double> samples_1_p(n_samples);
        std::vector<double> samples_2_p(n_samples);

        // Start performing slices 
        for(int i=0; i<slices; i++) {

            // Pick a sample in the hypersphere
            vector_t theta = hypersphere_sample(dim,dist_n,rng);

            // Perform the Radon transform of the KDE
            for(int j=0; j<n_samples; j++) {
                samples_1_p[j] = samples_1[j].dot(theta);
                samples_2_p[j] = samples_2[j].dot(theta);
            }

            // Compute the 1D-Wasserstein distance
            dist += wasserstein_1d(samples_1_p,samples_2_p,p);
            
        }

        return dist/(V_n*slices);
    }