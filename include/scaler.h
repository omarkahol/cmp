#ifndef SCALER_H
#define SCALER_H

#include <cmp_defines.h>


namespace cmp {
    
    /*
    SCALAR SCALERS

    These classes are used to scale the data. 
    The class scalar_scaler is the base class for all the scalers. 
    */

    class scalar_scaler {
    public:
        virtual void transform(double &data) = 0;
        virtual void inverse_transform(double &data) = 0;
        virtual double get_mean() = 0;
        virtual double get_scale() = 0;
        virtual size_t get_size() = 0;
        virtual double operator[](const int &i) const = 0;
        virtual const double &at(const int &i) const = 0;
        virtual const std::vector<double> &get_data() const = 0;

        void inverse_transform(std::vector<double> &data) {
            for (auto &d : data)
                inverse_transform(d);
        }
        void transform(std::vector<double> &data) {
            for (auto &d : data) {
                transform(d);
            }
        }

    };

    class standard_scaler : public scalar_scaler {

        private:
            double m_mean{0.0};
            double m_scale{1.0};
            std::vector<double> m_data;
            size_t m_size;
        
        public:
            standard_scaler(const std::vector<double> &x);
            void transform(double &x) override;
            void inverse_transform(double &x) override;
            double operator[](const int &i) const override;
            const double &at(const int &i) const override;
            size_t get_size() override;
            const std::vector<double> &get_data() const;

            double get_mean() override;
            double get_scale() override;
        
    };

    class dummy_scaler : public scalar_scaler {
        private:
            std::vector<double> m_data;
            size_t m_size;
        public:
            dummy_scaler(const std::vector<double> &x);
            void transform(double &x) override;
            void inverse_transform(double &x) override;
            double operator[](const int &i) const override;
            const double &at(const int &i) const override;
            size_t get_size() override;
            const std::vector<double> &get_data() const;

            double get_mean() override;
            double get_scale() override;
    };

    /*
    VECTOR SCALERS
    */

    // Template class for the vector scalers
    class vector_scaler {
        public:

            virtual void transform(Eigen::VectorXd &data) = 0;
            virtual void inverse_transform(Eigen::VectorXd &data) = 0;
            virtual Eigen::VectorXd get_mean() = 0;
            virtual Eigen::MatrixXd get_scale() = 0;
            virtual Eigen::VectorXd operator[](const int &i) const = 0;
            virtual const Eigen::VectorXd &at(const int &i) const = 0;

            virtual size_t get_size() = 0;
            virtual const std::vector<Eigen::VectorXd> &get_data() const = 0;

            void transform(std::vector<Eigen::VectorXd> &data) {
                for (auto &d : data) {
                    transform(d);
                }
            }
            void inverse_transform(std::vector<Eigen::VectorXd> &data) {
                for (auto &d : data) {
                    inverse_transform(d);
                }
            }
    };

    //  Standard vector scaler
    class standard_vector_scaler : public vector_scaler {
        private:
            Eigen::VectorXd m_mean;
            Eigen::LLT<Eigen::MatrixXd> m_scale;
            std::vector<Eigen::VectorXd> m_data;
            size_t m_size;

        public:
            standard_vector_scaler(std::vector<Eigen::VectorXd> data);
            void transform(Eigen::VectorXd &data) override;
            void inverse_transform(Eigen::VectorXd &data) override;
            Eigen::VectorXd operator[](const int &i) const override;
            const Eigen::VectorXd &at(const int &i) const override;
            size_t get_size() override;
            const std::vector<Eigen::VectorXd> &get_data() const override;

            Eigen::VectorXd get_mean() override;
            Eigen::MatrixXd get_scale() override;
    };

    // PCA vector scaler
    class pca_scaler : public vector_scaler {
        private:
            Eigen::VectorXd m_mean;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> m_eigen_solver;
            Eigen::MatrixXd m_L;
            Eigen::MatrixXd m_L_inv;
            std::vector<Eigen::VectorXd> m_data;
            size_t m_size;


        public:
            pca_scaler(std::vector<Eigen::VectorXd> data, size_t n_components);

            void transform(Eigen::VectorXd &data) override;
            void inverse_transform(Eigen::VectorXd &data) override;
            Eigen::VectorXd operator[](const int &i) const override;
            const Eigen::VectorXd &at(const int &i) const override;
            size_t get_size() override;
            const std::vector<Eigen::VectorXd> &get_data() const override;

            Eigen::VectorXd get_mean() override;
            Eigen::MatrixXd get_scale() override;
    };

    // Dummy vector scaler
    class dummy_vector_scaler : public vector_scaler {
        private:
            std::vector<Eigen::VectorXd> m_data;
            size_t m_size;
        public:
            dummy_vector_scaler(std::vector<Eigen::VectorXd> data);
            void transform(Eigen::VectorXd &data) override;
            void inverse_transform(Eigen::VectorXd &data) override;
            Eigen::VectorXd operator[](const int &i) const override;
            const Eigen::VectorXd &at(const int &i) const override;
            size_t get_size() override;
            const std::vector<Eigen::VectorXd> &get_data() const override;

            Eigen::VectorXd get_mean() override;
            Eigen::MatrixXd get_scale() override;
    };

    class component_scaler : public scalar_scaler {
        private:
            cmp::vector_scaler *m_base;
            size_t m_component;

        public:
            component_scaler()=default;
            component_scaler(cmp::vector_scaler* base, size_t component);

            void fit(cmp::vector_scaler* base, size_t component);
            void transform(double &data) override;
            void inverse_transform(double &data) override;
            double get_mean() override;
            double get_scale() override;
            size_t get_size() override;
            double operator[](const int &i) const override;
            const double &at(const int &i) const override;
            const std::vector<double> &get_data() const override;
    };
    
}


#endif // SCALER_H