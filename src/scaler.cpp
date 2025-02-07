#include "scaler.h"

cmp::standard_scaler::standard_scaler(const std::vector<double> &data)
{   
    double mean = 0.0;
    double std = 0.0;
    for (int i=0; i<data.size(); i++){
        mean += data[i];
        std += data[i]*data[i];
    }
    m_mean = mean/double(data.size());
    m_scale = std::sqrt((std - mean*mean/double(data.size()))/double(data.size()-1.0));


    // Transform the data and save it
    m_size = data.size();
    m_data = data;
    for (int i=0; i<m_size; i++){
        transform(m_data[i]);
    }
}

void cmp::standard_scaler::transform(double &x)
{
    x = (x - m_mean)/m_scale;
}

void cmp::standard_scaler::inverse_transform(double &x)
{
    x = x*m_scale + m_mean;
}

double cmp::standard_scaler::operator[](const int &i) const
{
    return m_data[i];
}

const double &cmp::standard_scaler::at(const int &i) const
{
    return m_data.at(i);
}

size_t cmp::standard_scaler::get_size()
{
    return m_size;
}

const std::vector<double> &cmp::standard_scaler::get_data() const
{
    return m_data;
}

double cmp::standard_scaler::get_mean()
{
    return m_mean;
}

double cmp::standard_scaler::get_scale()
{
    return m_scale;
}

cmp::standard_vector_scaler::standard_vector_scaler(std::vector<Eigen::VectorXd> data)
{
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(data[0].size());
    Eigen::MatrixXd scale = Eigen::MatrixXd::Zero(data[0].size(),data[0].size());

    for (int i=0; i<data.size(); i++){
        mean += data[i];
        scale += data[i]*data[i].transpose();
    }

    mean /= data.size();
    scale -= data.size()*mean*mean.transpose();
    scale /= double(data.size()-1);

    // Check if the matrix is self adjoint
    for (size_t i=0; i<scale.rows(); i++){
        for (size_t j=i+1; j<scale.cols(); j++){
            if (scale(i,j) != scale(j,i)){
                scale(j,i) = scale(i,j);
            }
        }
    }


    m_scale = scale.llt();
    m_mean = mean;

    // Transform the data
    m_data = data;
    m_size = data.size();
    for (int i=0; i<m_size; i++){
        transform(m_data[i]);
    }
}

void cmp::standard_vector_scaler::transform(Eigen::VectorXd &x)
{
    x = m_scale.matrixL().solve(x - m_mean);
}

void cmp::standard_vector_scaler::inverse_transform(Eigen::VectorXd &x)
{
    x = m_scale.matrixL()*x + m_mean;
}

Eigen::VectorXd cmp::standard_vector_scaler::operator[](const int &i) const
{
    return m_data[i];
}

const Eigen::VectorXd &cmp::standard_vector_scaler::at(const int &i) const
{
    return m_data.at(i);
}

size_t cmp::standard_vector_scaler::get_size()
{
    return m_size;
}

const std::vector<Eigen::VectorXd> &cmp::standard_vector_scaler::get_data() const
{
    return m_data;
}

Eigen::VectorXd cmp::standard_vector_scaler::get_mean()
{
    return m_mean;
}

Eigen::MatrixXd cmp::standard_vector_scaler::get_scale()
{
    return m_scale.matrixL();
}

cmp::pca_scaler::pca_scaler(std::vector<Eigen::VectorXd> data, size_t n_components)
{

    m_size = data.size();
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(data[0].size());
    Eigen::MatrixXd scale = Eigen::MatrixXd::Zero(data[0].size(),data[0].size());

    for (int i=0; i<m_size; i++){
        mean += data[i];
        scale += data[i]*data[i].transpose();
    }

    mean /= data.size();
    scale -= data.size()*mean*mean.transpose();
    scale /= double(data.size()-1);
    m_mean = mean;

    // Check if the matrix is self adjoint
    for (size_t i=0; i<m_size; i++){
        for (size_t j=i+1; j<scale.cols(); j++){
            if (scale(i,j) != scale(j,i)){
                scale(j,i) = scale(i,j);
            }
        }
    }

    // Perform the PCA
    m_eigen_solver.compute(scale);
    
    // Check the number of components
    if (n_components > m_eigen_solver.eigenvalues().size() || n_components < 0){
        
        Eigen::VectorXd eigenvalues = m_eigen_solver.eigenvalues();
        eigenvalues.array() = eigenvalues.array().abs().sqrt();

        Eigen::MatrixXd eigenvectors = m_eigen_solver.eigenvectors();
        m_L = eigenvectors*eigenvalues.asDiagonal();

        // Compute the inverse
        eigenvalues.array() = 1.0/eigenvalues.array();
        m_L_inv = eigenvalues.asDiagonal()*eigenvectors.transpose();
    } else {
        
        // Take the first n_components
        Eigen::VectorXd eigenvalues = m_eigen_solver.eigenvalues().tail(n_components);
        eigenvalues.array() = eigenvalues.array().abs().sqrt();

        // Take the first n_components eigenvectors
        Eigen::MatrixXd eigenvectors = m_eigen_solver.eigenvectors().rightCols(n_components);
        m_L = eigenvectors*eigenvalues.asDiagonal();

        // Compute the inverse
        eigenvalues.array() = 1.0/eigenvalues.array();
        m_L_inv = eigenvalues.asDiagonal()*eigenvectors.transpose();

    }

    // Transform the data
    m_data = data;
    for (int i=0; i<m_size; i++){
        transform(m_data[i]);
    }
}

void cmp::pca_scaler::transform(Eigen::VectorXd &data)
{
    data = m_L_inv*(data - m_mean);
}

void cmp::pca_scaler::inverse_transform(Eigen::VectorXd &data)
{
    data = m_L*data + m_mean;
}

Eigen::VectorXd cmp::pca_scaler::operator[](const int &i) const
{
    return m_data[i];
}

const Eigen::VectorXd &cmp::pca_scaler::at(const int &i) const
{
    return m_data.at(i);
}

size_t cmp::pca_scaler::get_size()
{
    return m_size;
}

const std::vector<Eigen::VectorXd> &cmp::pca_scaler::get_data() const
{
    return m_data;
}

Eigen::VectorXd cmp::pca_scaler::get_mean()
{
    return m_mean;
}

Eigen::MatrixXd cmp::pca_scaler::get_scale()
{
    return m_L;
}

cmp::dummy_scaler::dummy_scaler(const std::vector<double> &x)
{
    m_data = x;
    m_size = x.size();
}

void cmp::dummy_scaler::transform(double &x)
{
    // Do nothing
}

void cmp::dummy_scaler::inverse_transform(double &x)
{
    // Do nothing
}

double cmp::dummy_scaler::operator[](const int &i) const
{
    return m_data[i];
}

const double &cmp::dummy_scaler::at(const int &i) const
{
    return m_data.at(i);
}

size_t cmp::dummy_scaler::get_size()
{
    return m_size;
}

double cmp::dummy_scaler::get_mean()
{
    return 0.0;
}

double cmp::dummy_scaler::get_scale()
{
    return 1.0;
}

const std::vector<double> &cmp::dummy_scaler::get_data() const
{
    return m_data;
}

cmp::dummy_vector_scaler::dummy_vector_scaler(std::vector<Eigen::VectorXd> data)
{
    m_data = data;
    m_size = data.size();
}

void cmp::dummy_vector_scaler::transform(Eigen::VectorXd &data)
{
    // Do nothing
}

void cmp::dummy_vector_scaler::inverse_transform(Eigen::VectorXd &data)
{
    // Do nothing
}

Eigen::VectorXd cmp::dummy_vector_scaler::operator[](const int &i) const
{
    return m_data[i];
}

const Eigen::VectorXd &cmp::dummy_vector_scaler::at(const int &i) const
{
    return m_data.at(i);
}

size_t cmp::dummy_vector_scaler::get_size()
{
    return m_size;
}

const std::vector<Eigen::VectorXd> &cmp::dummy_vector_scaler::get_data() const
{
    return m_data;
}

Eigen::VectorXd cmp::dummy_vector_scaler::get_mean()
{
    return Eigen::VectorXd::Ones(m_data[0].size());
}

Eigen::MatrixXd cmp::dummy_vector_scaler::get_scale()
{
    return Eigen::MatrixXd::Identity(m_data[0].size(),m_data[0].size());
}

cmp::component_scaler::component_scaler(cmp::vector_scaler *base, size_t component)
{
    m_base = base;
    m_component = component;
}

void cmp::component_scaler::fit(cmp::vector_scaler *base, size_t component)
{
    m_base = base;
    m_component = component;
}

void cmp::component_scaler::transform(double &data)
{
    // Do nothing
}

void cmp::component_scaler::inverse_transform(double &data)
{
    // Do nothing
}

double cmp::component_scaler::get_mean()
{
    return 0;
}

double cmp::component_scaler::get_scale()
{
    return 1;
}

size_t cmp::component_scaler::get_size()
{
    return m_base->get_size();
}

double cmp::component_scaler::operator[](const int &i) const
{
    return m_base->operator[](i)(m_component);
}

const double &cmp::component_scaler::at(const int &i) const
{
    return m_base->at(i)(m_component);
}

const std::vector<double> &cmp::component_scaler::get_data() const
{
    std::vector<double> data(m_base->get_size());
    for (int i=0; i<m_base->get_size(); i++){
        data[i] = m_base->at(i)(m_component);
    }
    return data;
}




