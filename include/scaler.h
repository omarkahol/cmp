#ifndef SCALER_H
#define SCALER_H

#include <cmp_defines.h>


namespace cmp::scaler {

    constexpr double TOL = 1e-10;
    
    /**
     * Template virtual class for a Scaler object which works with scaler valued data
     */
    class ScalarScaler {
    public:
        
        virtual double transform(const double &x) const = 0;
        virtual double inverseTransfrom(const double &x) const = 0;

        virtual double getIntercept() const = 0;
        virtual double getScale() const = 0;

        virtual void fit(const std::vector<double> &data) = 0;

        virtual std::vector<double> fit_transform(const std::vector<double> &data) = 0;
    };

    class StandardScaler : public ScalarScaler {

        private:
            double mean_{0.0};
            double std_{1.0};
        
        public:

        // Constructors
            StandardScaler() = default;
            StandardScaler(double mean, double std) : mean_(mean), std_(std) {};
            StandardScaler(const StandardScaler &other) = default;
            StandardScaler (StandardScaler &&other) = default;
        
        // Destructor
            ~StandardScaler() = default;
        
        // Assignment operators
            StandardScaler &operator=(const StandardScaler &other) = default;
            StandardScaler &operator=(StandardScaler &&other) = default;

        // Methods
            double transform(const double &x) const override;
            double inverseTransfrom(const double &x) const override;
            
            double getIntercept() const override {return mean_;};
            double getScale() const override {return std_;};

        // Fit
            void fit(const std::vector<double> &data) override;
            std::vector<double> fit_transform(const std::vector<double> &data) override;
    };

    /**
     * This scaler does not apply any transformation to the data.
     */
    class DummyScaler : public ScalarScaler {
        public:

        // Constructors
            DummyScaler() = default;
            DummyScaler(const DummyScaler &other) = default;
            DummyScaler(DummyScaler &&other) = default;

        // Destructor
            ~DummyScaler() = default;

        // Assignment operators
            DummyScaler &operator=(const DummyScaler &other) = default;
            DummyScaler &operator=(DummyScaler &&other) = default;

        // Methods
            double transform(const double &x) const override {return x;};
            double inverseTransfrom(const double &x) const override {return x;};
            
            double getIntercept() const override {return 0;};
            double getScale() const override {return 1;};

        // Fit
            void fit(const std::vector<double> &data) override {};
            std::vector<double> fit_transform(const std::vector<double> &data) override {return data;};
    };

    /*
    VECTOR SCALERS
    */

    /**
     * Template virtual class for a Scaler object which works with vector valued data.
     */
    class VectorScaler {
        public:

            virtual Eigen::VectorXd transform(const Eigen::VectorXd &data) const = 0;
            virtual Eigen::VectorXd inverseTransfrom(const Eigen::VectorXd &data) const = 0;
            
            virtual Eigen::VectorXd getIntercept() const = 0;
            virtual Eigen::MatrixXd getScale() const = 0;
            
            virtual void fit(const std::vector<Eigen::VectorXd> &data) = 0;
            virtual std::vector<Eigen::VectorXd> fit_transform(const std::vector<Eigen::VectorXd> &data) = 0;
    };

    //  Standard vector scaler
    class StandardVectorScaler : public VectorScaler {
        private:
            Eigen::VectorXd mean_;
            Eigen::LLT<Eigen::MatrixXd> lltDecomposition_;

        public:
            StandardVectorScaler()=default;
            StandardVectorScaler(const Eigen::VectorXd &mean, const Eigen::MatrixXd &scale) : mean_(mean), lltDecomposition_(scale) {};
            StandardVectorScaler(const StandardVectorScaler &other) = default;
            StandardVectorScaler(StandardVectorScaler &&other) = default;
            
            ~StandardVectorScaler() = default;
            
            StandardVectorScaler &operator=(const StandardVectorScaler &other) = default;
            StandardVectorScaler &operator=(StandardVectorScaler &&other) = default;

            Eigen::VectorXd transform(const Eigen::VectorXd &data) const override;
            Eigen::VectorXd inverseTransfrom(const Eigen::VectorXd &data) const override;
            
            Eigen::VectorXd getIntercept() const override {return mean_;};
            Eigen::MatrixXd getScale() const override {return lltDecomposition_.matrixL();};

            void fit(const std::vector<Eigen::VectorXd> &data) override;
            std::vector<Eigen::VectorXd> fit_transform(const std::vector<Eigen::VectorXd> &data) override;
    };

    // PCA vector scaler
    class PCA : public VectorScaler {
        private:
            Eigen::VectorXd mean_;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver_;
            Eigen::MatrixXd sqrtCov_;
            Eigen::MatrixXd sqrtCovInv_;

            size_t nComponents_;

            void eigenDecomposition();


        public:
            PCA(size_t nComponents) : nComponents_(nComponents) {};
            PCA(const PCA &other) = default;
            PCA(PCA &&other) = default;
            
            ~PCA() = default;

            PCA &operator=(const PCA &other) = default;
            PCA &operator=(PCA &&other) = default;

            Eigen::VectorXd transform(const Eigen::VectorXd &data) const override;
            Eigen::VectorXd inverseTransfrom(const Eigen::VectorXd &data) const override;
            
            Eigen::VectorXd getIntercept() const override {return mean_;};
            Eigen::MatrixXd getScale() const override {return sqrtCov_;};

            void fit(const std::vector<Eigen::VectorXd> &data) override;
            std::vector<Eigen::VectorXd> fit_transform(const std::vector<Eigen::VectorXd> &data) override;

            void resize(size_t nComponents);

            Eigen::VectorXd getEigenvalues() const {return eigenSolver_.eigenvalues();};
            Eigen::MatrixXd getEigenvectors() const {return eigenSolver_.eigenvectors();};
    };

    // Dummy vector scaler
    class DummyVectorScaler : public VectorScaler {
        private:
            size_t dim_;
        public:
            DummyVectorScaler() = default;

            DummyVectorScaler(const DummyVectorScaler &other) = default;
            DummyVectorScaler(DummyVectorScaler &&other) = default;

            ~DummyVectorScaler() = default;

            DummyVectorScaler &operator=(const DummyVectorScaler &other) = default;
            DummyVectorScaler &operator=(DummyVectorScaler &&other) = default;

            Eigen::VectorXd transform(const Eigen::VectorXd &data) const override {return data;};
            Eigen::VectorXd inverseTransfrom(const Eigen::VectorXd &data) const override {return data;};
            
            Eigen::VectorXd getIntercept() const override {return Eigen::VectorXd::Zero(dim_);};
            Eigen::MatrixXd getScale() const override {return Eigen::MatrixXd::Identity(dim_,dim_);};

            void fit(const std::vector<Eigen::VectorXd> &data) override {
                dim_ = data[0].size();
            };

            std::vector<Eigen::VectorXd> fit_transform(const std::vector<Eigen::VectorXd> &data) override {
                dim_ = data[0].size();
                return data;
            };

            void setDim(size_t dim) {dim_ = dim;};
    
    };

    // Elliptic vector scaler
    class EllipticScaler : public VectorScaler {
        private:
            Eigen::VectorXd mean_;
            std::vector<double> std_;

        public:
            EllipticScaler()=default;
            EllipticScaler(const Eigen::VectorXd &mean, const std::vector<double> &std) : mean_(mean), std_(std) {};
            EllipticScaler(const EllipticScaler &other) = default;
            EllipticScaler(EllipticScaler &&other) = default;

            ~EllipticScaler() = default;

            EllipticScaler &operator=(const EllipticScaler &other) = default;
            EllipticScaler &operator=(EllipticScaler &&other) = default;
            Eigen::VectorXd transform(const Eigen::VectorXd &data) const override;
            Eigen::VectorXd inverseTransfrom(const Eigen::VectorXd &data) const override;
            Eigen::VectorXd getIntercept() const override {return mean_;};
            Eigen::MatrixXd getScale() const override {
                Eigen::MatrixXd scale = Eigen::MatrixXd::Identity(mean_.size(), mean_.size());
                for (int i = 0; i < mean_.size(); i++) {
                    scale(i,i) = std_[i];
                }
                return scale;
            };
            void fit(const std::vector<Eigen::VectorXd> &data) override;
            std::vector<Eigen::VectorXd> fit_transform(const std::vector<Eigen::VectorXd> &data) override;
            void setMean(const Eigen::VectorXd &mean) {mean_ = mean;};
            void setStd(const std::vector<double> &std) {std_ = std;};

    };
}


#endif // SCALER_H