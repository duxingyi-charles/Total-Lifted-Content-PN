#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>


#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/CholmodSupport>

#include <chrono>


using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat;

typedef Eigen::Triplet<double> eigenT;

typedef Eigen::CholmodSupernodalLLT<SpMat> CholmodSolver;

// Eigen Utils

// convert c++ matrix to Eigen matrix
template<typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> toEigenMatrix(const std::vector<std::vector<Scalar> > &mat) {
    int ncol = mat.size();
    int nrow = mat[0].size();
    Matrix<Scalar, Dynamic, Dynamic> m(nrow, ncol);
    for (int i = 0; i < ncol; ++i) {
        for (int j = 0; j < nrow; ++j) {
            m(j, i) = mat[i][j];
        }
    }
    return m;
}

// convert c++ vector to Eigen vector
template<typename Scalar>
Matrix<Scalar, Dynamic, 1> toEigenVector(const std::vector<Scalar> &vec) {
    int size = vec.size();
    Matrix<Scalar, Dynamic, 1> v(size);
    for (int i = 0; i < size; ++i) {
        v(i) = vec[i];
    }
    return v;
}

// export Eigen Matrix
bool exportMatrix(std::string filename, const MatrixXd &mat) {
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        std::cerr << "Failed to open " << filename << "!" << std::endl;
        return false;
    }

    //precision of output
    typedef std::numeric_limits<double> dbl;
    out_file.precision(dbl::max_digits10);
    //
    for (auto i = 0; i < mat.cols(); ++i) {
        for (auto j = 0; j < mat.rows(); ++j) {
            out_file << mat(j, i) << " ";
        }
        out_file << std::endl;
    }
    //
    out_file.close();
    return true;
}

bool importData(const char *filename,
                std::vector<std::vector<double> > &restV,
                std::vector<std::vector<double> > &initV,
                std::vector<std::vector<int> > &F,
                std::vector<int> &handles) {
    std::ifstream in_file(filename);

    if (!in_file.is_open()) {
        std::cerr << "Failed to open " << filename << "!" << std::endl;
        return false;
    }

    //read the file
    size_t n, ndim;
    // restV
    in_file >> n >> ndim;
    restV.resize(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> v(ndim);
        for (size_t j = 0; j < ndim; ++j) {
            in_file >> v[j];
        }
        restV[i] = v;
    }

    //initV
    in_file >> n >> ndim;
    initV.resize(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> v(ndim);
        for (size_t j = 0; j < ndim; ++j) {
            in_file >> v[j];
        }
        initV[i] = v;
    }

    //F
    size_t simplexSize;
    in_file >> n >> simplexSize;
    F.resize(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<int> v(simplexSize);
        for (size_t j = 0; j < simplexSize; ++j) {
            in_file >> v[j];
        }
        F[i] = v;
    }

    //handles
    in_file >> n;
    handles.resize(n);
    for (size_t i = 0; i < n; ++i) {
        int v;
        in_file >> v;
        handles[i] = v;
    }

    in_file.close();

    return true;
}

// solver options
class SolverOptionManager {
public:
    //default options
    SolverOptionManager() :
            form("Tutte"), alphaRatio(1e-6), alpha(1e-6),
            ftol_abs(1e-8), ftol_rel(1e-8), xtol_abs(1e-8), xtol_rel(1e-8), gtol_abs(1e-8),
            maxeval(1000), algorithm("ProjectedNewton"), stopCode("all_good"),
            /*record()*/ record_vert(false), record_energy(false), record_minArea(false),
            record_gradient(false), record_gradientNorm(false), record_searchDirection(false),
            record_searchNorm(false), record_stepSize(false), record_stepNorm(false),
            save_vert(false), resFile("") {};

    //import options from file
    SolverOptionManager(const char *option_filename, const char *result_filename) :
            form("Tutte"), alphaRatio(1e-6), alpha(1e-6),
            ftol_abs(1e-8), ftol_rel(1e-8), xtol_abs(1e-8), xtol_rel(1e-8), gtol_abs(1e-8),
            maxeval(1000), algorithm("ProjectedNewton"), stopCode("all_good"),
            /*record()*/ record_vert(false), record_energy(false), record_minArea(false),
            record_gradient(false), record_gradientNorm(false), record_searchDirection(false),
            record_searchNorm(false), record_stepSize(false), record_stepNorm(false),
            save_vert(false), resFile(result_filename) {
        if (!importOptions(option_filename)) {
            std::cout << "SolverOptionManager Warn: default options are used." << std::endl;
        }
    };

    ~SolverOptionManager() = default;

    // energy formulation options
    std::string form;
    double alphaRatio;
    double alpha;

    // optimization options
    double ftol_abs;
    double ftol_rel;
    double xtol_abs;
    double xtol_rel;
    double gtol_abs;  //todo: add gtol_abs. also need to change code in Mathematica
    int maxeval;
    std::string algorithm;
    std::string stopCode;

    //record values
    bool record_vert;
    bool record_energy;
    bool record_minArea;
    bool record_gradient;
    bool record_gradientNorm;
    bool record_searchDirection;
    bool record_searchNorm;
    bool record_stepSize;
    bool record_stepNorm;  // ||x_next - x||
    std::vector<MatrixXd> vertRecord;
    std::vector<double> energyRecord;
    std::vector<double> minAreaRecord;
    std::vector<VectorXd> gradientRecord;
    std::vector<double> gradientNormRecord;
    std::vector<VectorXd> searchDirectionRecord;
    std::vector<double> searchNormRecord;
    std::vector<double> stepSizeRecord;
    std::vector<double> stepNormRecord;

    //save values
    std::string resFile;
    bool save_vert;


    void printOptions() {
        std::cout << "form:\t" << form << "\n";
        std::cout << "alphaRatio:\t" << alphaRatio << "\n";
        std::cout << "alpha:\t" << alpha << "\n";
        std::cout << "ftol_abs:\t" << ftol_abs << "\n";
        std::cout << "ftol_rel:\t" << ftol_rel << "\n";
        std::cout << "xtol_abs:\t" << xtol_abs << "\n";
        std::cout << "xtol_rel:\t" << xtol_rel << "\n";
        std::cout << "gtol_abs:\t" << gtol_abs << "\n";
        std::cout << "maxeval:\t" << maxeval << "\n";
        std::cout << "algorithm:\t" << algorithm << "\n";
        std::cout << "stopCode:\t" << stopCode << "\n";
        std::cout << "record:  \t" << "{ ";
        if (record_vert) std::cout << "vert ";
        if (record_energy) std::cout << "energy ";
        if (record_minArea) std::cout << "minArea ";
        if (record_gradient) std::cout << "gradient ";
        if (record_gradientNorm) std::cout << "gNorm ";
        if (record_searchDirection) std::cout << "searchDirection ";
        if (record_searchNorm) std::cout << "searchNorm ";
        if (record_stepSize) std::cout << "stepSize ";
        if (record_stepNorm) std::cout << "stepNorm ";
        std::cout << "}" << std::endl;
        //
        std::cout << "result file: \t" << resFile << std::endl;
        std::cout << "save:  \t" << "{ ";
        if (save_vert) std::cout << "vert ";
        std::cout << "}" << std::endl;
    }

    bool importOptions(const char *filename) {
        //open the data file
        std::ifstream in_file(filename);

        if (!in_file.is_open()) {
            std::cerr << "Failed to open " << filename << "!" << std::endl;
            return false;
        }

        //read the file
        unsigned normal = 0;
        std::string optName;
        while (true) {
            in_file >> optName;
            if (optName != "form") {
                normal = 1;
                break;
            }
            in_file >> form;

            in_file >> optName;
            if (optName != "alphaRatio") {
                normal = 2;
                break;
            }
            in_file >> alphaRatio;

            in_file >> optName;
            if (optName != "alpha") {
                normal = 3;
                break;
            }
            in_file >> alpha;

            in_file >> optName;
            if (optName != "ftol_abs") {
                normal = 4;
                break;
            }
            in_file >> ftol_abs;

            in_file >> optName;
            if (optName != "ftol_rel") {
                normal = 5;
                break;
            }
            in_file >> ftol_rel;

            in_file >> optName;
            if (optName != "xtol_abs") {
                normal = 6;
                break;
            }
            in_file >> xtol_abs;

            in_file >> optName;
            if (optName != "xtol_rel") {
                normal = 7;
                break;
            }
            in_file >> xtol_rel;

            in_file >> optName;
            if (optName != "gtol_abs") {
                normal = 8;
                break;
            }
            in_file >> gtol_abs;

            in_file >> optName;
            if (optName != "algorithm") {
                normal = 9;
                break;
            }
            in_file >> algorithm;

            in_file >> optName;
            if (optName != "maxeval") {
                normal = 10;
                break;
            }
            in_file >> maxeval;

            in_file >> optName;
            if (optName != "stopCode") {
                normal = 11;
                break;
            }
            in_file >> stopCode;

            // record values
            in_file >> optName;
            if (optName != "record") {
                normal = 12;
                break;
            }

            in_file >> optName;
            if (optName != "vert") {
                normal = 13;
                break;
            }
            int selected = 0;
            in_file >> selected;
            if (selected > 0) {
                record_vert = true;
            }

            in_file >> optName;
            if (optName != "energy") {
                normal = 14;
                break;
            }
            selected = 0;
            in_file >> selected;
            if (selected > 0) {
                record_energy = true;
            }

            in_file >> optName;
            if (optName != "minArea") {
                normal = 15;
                break;
            }
            selected = 0;
            in_file >> selected;
            if (selected > 0) {
                record_minArea = true;
            }

            in_file >> optName;
            if (optName != "gradient") {
                normal = 16;
                break;
            }
            selected = 0;
            in_file >> selected;
            if (selected > 0) {
                record_gradient = true;
            }

            in_file >> optName;
            if (optName != "gNorm") {
                normal = 17;
                break;
            }
            selected = 0;
            in_file >> selected;
            if (selected > 0) {
                record_gradientNorm = true;
            }

            in_file >> optName;
            if (optName != "searchDirection") {
                normal = 18;
                break;
            }
            selected = 0;
            in_file >> selected;
            if (selected > 0) {
                record_searchDirection = true;
            }

            in_file >> optName;
            if (optName != "searchNorm") {
                normal = 19;
                break;
            }
            selected = 0;
            in_file >> selected;
            if (selected > 0) {
                record_searchNorm = true;
            }

            in_file >> optName;
            if (optName != "stepSize") {
                normal = 20;
                break;
            }
            selected = 0;
            in_file >> selected;
            if (selected > 0) {
                record_stepSize = true;
            }

            in_file >> optName;
            if (optName != "stepNorm") {
                normal = 21;
                break;
            }
            selected = 0;
            in_file >> selected;
            if (selected > 0) {
                record_stepNorm = true;
            }

            // save values
            in_file >> optName;
            if (optName != "save") {
                normal = 22;
                break;
            }

            in_file >> optName;
            if (optName != "vert") {
                normal = 23;
                break;
            }
            selected = 0;
            in_file >> selected;
            if (selected > 0) {
                save_vert = true;
            }

            break;
        }

        in_file.close();

        if (normal != 0) {
            std::cout << "Err:(" << normal << ") fail to import options from file. Check file format." << std::endl;
            return false;
        }

        return true;
    }

};


// helper functions
void computeSquaredEdgeLength(const MatrixXd &V,
                              const MatrixXi &F,
                              MatrixXd &D) {
    int nf = F.cols();
    // int simplexSize = F.rows();
    // int n_edge = simplexSize * (simplexSize-1) / 2;
    int n_edge = 3;

    D.resize(n_edge, nf);
    for (int i = 0; i < nf; ++i) {
        auto v1 = V.col(F(0, i));
        auto v2 = V.col(F(1, i));
        auto v3 = V.col(F(2, i));
        auto e1 = v2 - v3;
        auto e2 = v3 - v1;
        auto e3 = v1 - v2;
        D(0, i) = e1.squaredNorm();
        D(1, i) = e2.squaredNorm();
        D(2, i) = e3.squaredNorm();
    }
}

inline double tri_signed_area(const Vector2d &p1, const Vector2d &p2, const Vector2d &p3) {
    // input: 2D coordinates of 3 points of triangle
    // return: signed area of the triangle
    return 0.5 * (p3(0) * (p1(1) - p2(1)) + p1(0) * (p2(1) - p3(1)) + p2(0) * (p3(1) - p1(1)));
}

void computeSignedArea(const MatrixXd &V, const MatrixXi &F, VectorXd &areaList) {
    // note: V must be a (2 * nv) matrix
    int nf = F.cols();
    areaList.resize(nf);
    for (int i = 0; i < nf; ++i) {
        const Vector2d &p1 = V.col(F(0, i));
        const Vector2d &p2 = V.col(F(1, i));
        const Vector2d &p3 = V.col(F(2, i));
        areaList(i) = tri_signed_area(p1, p2, p3);
    }
}

double computeMinSignedArea(const MatrixXd &V, const MatrixXi &F) {
    VectorXd areaList;
    computeSignedArea(V, F, areaList);
    return areaList.minCoeff();
}

double computeTotalSignedArea(const MatrixXd &V, const MatrixXi &F) {
    VectorXd areaList;
    computeSignedArea(V, F, areaList);
    return areaList.sum();
}

// Heron's formula and its derivatives
double squaredHeronTriArea(double d1, double d2, double d3) {
    // sort d1,d2,d3 as a >= b >= c
    double a, b, c;
    if (d1 > d2) {
        a = d1;
        b = d2;
    }
    else {
        a = d2;
        b = d1;
    }
    c = d3;
    if (d3 > b) {
        c = b;
        b = d3;
        if (d3 > a) {
            b = a;
            a = d3;
        }
    }

    a = sqrt(a);
    b = sqrt(b);
    c = sqrt(c);

    return 0.0625 * (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c));
}


double HeronTriArea(double d1, double d2, double d3) {
    // sort d1,d2,d3 as a >= b >= c
    double a, b, c;
    if (d1 > d2) {
        a = d1;
        b = d2;
    }
    else {
        a = d2;
        b = d1;
    }
    c = d3;
    if (d3 > b) {
        c = b;
        b = d3;
        if (d3 > a) {
            b = a;
            a = d3;
        }
    }

    a = sqrt(a);
    b = sqrt(b);
    c = sqrt(c);

    return 0.25 * sqrt(abs((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))));
}

double totalHeronTriArea(const MatrixXd &V, const MatrixXi &F) {
    int nF = F.cols();
    MatrixXd D(3, nF);
    computeSquaredEdgeLength(V, F, D);

    // compute triangle areas
    double area = 0;
    for (int i = 0; i < nF; ++i) {
        area += HeronTriArea(D(0, i), D(1, i), D(2, i));
    }
    return area;
}


void HeronTriAreaGrad(double d1, double d2, double d3,
                      double &area, Vector3d &grad) {
    area = HeronTriArea(d1, d2, d3);
    double s = 1 / (16 * area);
    //ToDo: more robust (sort d1,d2,d3)
    grad << d2 + d3 - d1, d1 + d3 - d2, d1 + d2 - d3;
    grad *= s;
}

void HeronTriAreaGradHessian(double d1, double d2, double d3,
                             double &area, Vector3d &grad, Matrix3d &Hess) {
    area = HeronTriArea(d1, d2, d3);
    double s = 1.0 / (16.0 * area);
    grad << d2 + d3 - d1, d1 + d3 - d2, d1 + d2 - d3;
    grad *= s;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Hess(j, i) = grad(i) * grad(j);
        }
    }

    Hess /= (-area);

    Matrix3d t;
    t << -1.0, 1.0, 1.0,
            1.0, -1.0, 1.0,
            1.0, 1.0, -1.0;
    t *= s;

    Hess += t;
}

// Lifted triangle area and its derivatives

double liftedTriArea(const MatrixXd &vert, const Vector3d &r) {
    auto v1 = vert.col(0);
    auto v2 = vert.col(1);
    auto v3 = vert.col(2);
    auto e1 = v2 - v3;
    auto e2 = v3 - v1;
    auto e3 = v1 - v2;
    double d1 = e1.squaredNorm() + r(0);
    double d2 = e2.squaredNorm() + r(1);
    double d3 = e3.squaredNorm() + r(2);
    return HeronTriArea(d1, d2, d3);
}

void liftedTriAreaGrad(const MatrixXd &vert, const Vector3d &r,
                       double &area, MatrixXd &grad) {
    auto v1 = vert.col(0);
    auto v2 = vert.col(1);
    auto v3 = vert.col(2);
    auto e1 = v2 - v3;
    auto e2 = v3 - v1;
    auto e3 = v1 - v2;
    double d1 = e1.squaredNorm() + r(0);
    double d2 = e2.squaredNorm() + r(1);
    double d3 = e3.squaredNorm() + r(2);

    //
    area = HeronTriArea(d1, d2, d3);

    //
    double g1 = d2 + d3 - d1;
    double g2 = d3 + d1 - d2;
    double g3 = d1 + d2 - d3;

    //
    auto ge1 = g1 * e1;
    auto ge2 = g2 * e2;
    auto ge3 = g3 * e3;

    //note: grad has the same dimension as vert
    grad.resize(vert.rows(), vert.cols());
    grad.col(0) = ge3 - ge2;
    grad.col(1) = ge1 - ge3;
    grad.col(2) = ge2 - ge1;

    double s = 1 / (8 * area);
    grad *= s;
}

void liftedTriAreaGradLaplacian(const MatrixXd &vert, const Vector3d &r,
                                double &area, MatrixXd &grad, Matrix3d &Lap) {
    auto v1 = vert.col(0);
    auto v2 = vert.col(1);
    auto v3 = vert.col(2);
    auto e1 = v2 - v3;
    auto e2 = v3 - v1;
    auto e3 = v1 - v2;
    double d1 = e1.squaredNorm() + r(0);
    double d2 = e2.squaredNorm() + r(1);
    double d3 = e3.squaredNorm() + r(2);

    //
    area = HeronTriArea(d1, d2, d3);

    //
    double g1 = d2 + d3 - d1;
    double g2 = d3 + d1 - d2;
    double g3 = d1 + d2 - d3;

    //
    auto ge1 = g1 * e1;
    auto ge2 = g2 * e2;
    auto ge3 = g3 * e3;

    //note: grad has the same dimension as vert
    grad.resize(vert.rows(), vert.cols());
    grad.col(0) = ge3 - ge2;
    grad.col(1) = ge1 - ge3;
    grad.col(2) = ge2 - ge1;
    double s = 1 / (8 * area);
    grad *= s;

    //
    Lap << g2 + g3, -g3, -g2,
            -g3, g1 + g3, -g1,
            -g2, -g1, g1 + g2;
    Lap *= s;

}

void liftedTriAreaGradHessian(const MatrixXd &vert, const Vector3d &r,
                              double &area, MatrixXd &grad, MatrixXd &Hess) {
    auto v1 = vert.col(0);
    auto v2 = vert.col(1);
    auto v3 = vert.col(2);
    auto e1 = v2 - v3;
    auto e2 = v3 - v1;
    auto e3 = v1 - v2;
    double d1 = e1.squaredNorm() + r(0);
    double d2 = e2.squaredNorm() + r(1);
    double d3 = e3.squaredNorm() + r(2);

    int vDim = v1.size();

    //
    area = HeronTriArea(d1, d2, d3);

    //
    double g1 = d2 + d3 - d1;
    double g2 = d3 + d1 - d2;
    double g3 = d1 + d2 - d3;

    //
    auto ge1 = g1 * e1;
    auto ge2 = g2 * e2;
    auto ge3 = g3 * e3;

    //
    auto av1 = ge3 - ge2;
    auto av2 = ge1 - ge3;
    auto av3 = ge2 - ge1;

    //note: grad has the same dimension as vert
    grad.resize(vert.rows(), vert.cols());
    grad.col(0) = av1;
    grad.col(1) = av2;
    grad.col(2) = av3;
    double s = 1 / (8 * area);
    grad *= s;

    // Hess 1: Laplacian
    Matrix3d Lap;
    Lap << g2 + g3, -g3, -g2,
            -g3, g1 + g3, -g1,
            -g2, -g1, g1 + g2;
    Lap *= s;

    // Kronecker product
    MatrixXd Hess1(3 * vDim, 3 * vDim);
    MatrixXd I = MatrixXd::Identity(vDim, vDim);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Hess1.block(i * vDim, j * vDim, vDim, vDim) = Lap(i, j) * I;
        }
    }

    // Hess 2
    MatrixXd E11(vDim, vDim);
    MatrixXd E22(vDim, vDim);
    MatrixXd E33(vDim, vDim);
    MatrixXd E13(vDim, vDim);
    MatrixXd E12(vDim, vDim);
    MatrixXd E231(vDim, vDim);
    MatrixXd E312(vDim, vDim);

    E11 = e1 * e1.transpose();
    E12 = e1 * e2.transpose();
    E13 = e1 * e3.transpose();
    E22 = e2 * e2.transpose();
    E33 = e3 * e3.transpose();
    E231 = (e2 - e3) * e1.transpose();
    E312 = (e3 - e1) * e2.transpose();

    MatrixXd Hess2(3 * vDim, 3 * vDim);
    Hess2.block(0, 0, vDim, vDim) = E11;
    Hess2.block(0, vDim, vDim, vDim) = E13 + E231;
    Hess2.block(0, 2 * vDim, vDim, vDim) = E12 - E231;
    Hess2.block(vDim, vDim, vDim, vDim) = E22;
    Hess2.block(vDim, 2 * vDim, vDim, vDim) = E12.transpose() + E312;
    Hess2.block(2 * vDim, 2 * vDim, vDim, vDim) = E33;

    Hess2.block(vDim, 0, vDim, vDim) = Hess2.block(0, vDim, vDim, vDim).transpose();
    Hess2.block(2 * vDim, 0, vDim, vDim) = Hess2.block(0, 2 * vDim, vDim, vDim).transpose();
    Hess2.block(2 * vDim, vDim, vDim, vDim) = Hess2.block(vDim, 2 * vDim, vDim, vDim).transpose();

    s *= 2; // 1/4area
    Hess2 *= s;

    // Hess 3
    MatrixXd Hess3(3 * vDim, 3 * vDim);
    Hess3.block(0, 0, vDim, vDim) = av1 * av1.transpose();
    Hess3.block(0, vDim, vDim, vDim) = av1 * av2.transpose();
    Hess3.block(0, 2 * vDim, vDim, vDim) = av1 * av3.transpose();
    Hess3.block(vDim, vDim, vDim, vDim) = av2 * av2.transpose();
    Hess3.block(vDim, 2 * vDim, vDim, vDim) = av2 * av3.transpose();
    Hess3.block(2 * vDim, 2 * vDim, vDim, vDim) = av3 * av3.transpose();

    Hess3.block(vDim, 0, vDim, vDim) = Hess3.block(0, vDim, vDim, vDim).transpose();
    Hess3.block(2 * vDim, 0, vDim, vDim) = Hess3.block(0, 2 * vDim, vDim, vDim).transpose();
    Hess3.block(2 * vDim, vDim, vDim, vDim) = Hess3.block(vDim, 2 * vDim, vDim, vDim).transpose();

    s = s * s / (4 * area);  // 1/64area^3
    Hess3 *= s;

    // Hessian
    Hess.resize(3 * vDim, 3 * vDim);
    Hess = Hess1 - Hess2 - Hess3;

}

// polynomial coefficicents of squared lifted triangle area
VectorXd squaredLiftedTriAreaPolyCoeff(const MatrixXd &vert, const Vector3d &r, const MatrixXd &pvec) {
    auto v1 = vert.col(0);
    auto v2 = vert.col(1);
    auto v3 = vert.col(2);
    auto e1 = v2 - v3;
    auto e2 = v3 - v1;
    auto e3 = v1 - v2;
    double d1 = e1.squaredNorm() + r(0);
    double d2 = e2.squaredNorm() + r(1);
    double d3 = e3.squaredNorm() + r(2);

    double a0 = squaredHeronTriArea(d1, d2, d3);

    auto p1 = pvec.col(0);
    auto p2 = pvec.col(1);
    auto p3 = pvec.col(2);
    auto q1 = p2 - p3;
    auto q2 = p3 - p1;
    auto q3 = p1 - p2;
    double eq1 = e1.transpose() * q1;
    double eq2 = e2.transpose() * q2;
    double eq3 = e3.transpose() * q3;
    double q1s = q1.squaredNorm();
    double q2s = q2.squaredNorm();
    double q3s = q3.squaredNorm();

    double a1 = (d3 * (eq1 + eq2 - eq3) + d2 * (eq1 - eq2 + eq3) + d1 * (eq2 + eq3 - eq1)) / 4;
    double a2 = (4 * eq1 * (eq2 + eq3) - 2 * eq1 * eq1 - 2 * (eq2 - eq3) * (eq2 - eq3) +
                 d3 * (q1s + q2s - q3s) + d2 * (q1s - q2s + q3s) +
                 d1 * (-q1s + q2s + q3s)) / 8;
    double a3 = (eq3 * (q1s + q2s - q3s) + eq2 * (q1s - q2s + q3s) + eq1 * (q2s + q3s - q1s)) / 4;
    double a4 = (2 * q1s * (q2s + q3s) - q1s * q1s - (q2s - q3s) * (q2s - q3s)) / 16;

    VectorXd coeff(5);
    coeff << a0, a1, a2, a3, a4;

    return coeff;
}

class LiftedFormulation {
public:
    LiftedFormulation(MatrixXd &restV, MatrixXd &initV, MatrixXi &restF,
                      VectorXi &handles, const std::string &form, double alphaRatio, double alpha) :
            V(initV), F(restF) {
        // compute freeI
        int nV = V.cols();
        vDim = V.rows();

        std::vector<bool> freeQ(nV, true);
        for (auto i = 0; i < handles.size(); ++i) {
            freeQ[handles(i)] = false;
        }
        freeI.resize(nV - handles.size());
        int ii = 0;
        for (int i = 0; i < nV; ++i) {
            if (freeQ[i]) {
                freeI[ii] = i;
                ++ii;
            }
        }
        std::sort(freeI.data(), freeI.data() + freeI.size());


        // compute indexDict and F_free
        indexDict = VectorXi::Constant(nV, -1);
        for (auto i = 0; i < freeI.size(); ++i) {
            indexDict(freeI(i)) = i;
        }

        F_free.resize(F.rows(), F.cols());
        for (auto i = 0; i < F.cols(); ++i) {
            for (auto j = 0; j < F.rows(); ++j) {
                F_free(j, i) = indexDict(F(j, i));
            }
        }

        // compute alpha if it's not explicitly given
        double alphaFinal;
        if (alpha >= 0) alphaFinal = alpha;
        else {  //alpha < 0. Deduce alpha from alphaRatio
            alphaFinal = computeAlpha(restV, initV, F, form, alphaRatio);
        }
        std::cout << "alphaRatio: " << alphaRatio << std::endl;
        std::cout << "alpha: " << alphaFinal << std::endl;

        // compute restD
        //for triangle mesh
        double a = alphaFinal;
        if (form == "harmonic") {
            computeSquaredEdgeLength(restV, F, restD);
            restD *= a;
        } else // Tutte form
        {
            restD = MatrixXd::Constant(3, F.cols(), a);
        }

        // compute x0 from initV
        x0.resize(vDim * freeI.size());
        for (auto i = 0; i < freeI.size(); ++i) {
            int vi = freeI(i);
            for (int j = 0; j < vDim; ++j) {
                x0(i * vDim + j) = V(j, vi);
            }
        }
    };

    ~LiftedFormulation() = default;

    VectorXi freeI;   // indices of free vertices
    //int nV;         // number of vertices
    size_t vDim;         // dimension of target vertices
    MatrixXi F;       // V indices of triangles
    MatrixXd restD;   // squared edge lengths of rest/auxiliary mesh

    VectorXi indexDict; // map: V index --> freeV index. If V(i) is not free, indexDict(i) = -1.
    MatrixXi F_free;    // freeV indices of triangles.

    VectorXd x0;  // initial variable vector
    MatrixXd V;   // current V of target mesh

    // compute alpha from alphaRatio
    static double computeAlpha(const MatrixXd &restV,
                               const MatrixXd &initV,
                               const MatrixXi &Faces,
                               std::string form, double alphaRatio) {
        unsigned nF = Faces.cols();
        double rest_measure = 0;
        double init_measure = 0;
        // tri
        init_measure = computeTotalSignedArea(initV, Faces);
        if (form == "harmonic") {
            rest_measure = totalHeronTriArea(restV, Faces);
        } else { // Tutte form
            rest_measure = nF * sqrt(3) / 4;
        }

        // alpha
        return alphaRatio * init_measure / rest_measure;
    }


    // x = Flatten(freeV)
    void update_V(const VectorXd &x) {
        for (auto i = 0; i < freeI.size(); ++i) {
            for (auto j = 0; j < vDim; ++j) {
                V(j, freeI(i)) = x[i * vDim + j];
            }
        }
    }

    double getLiftedEnergy(const VectorXd &x, VectorXd &energyList) {
        update_V(x);

        // compute lifted energy
        energyList.resize(F.cols());

        for (auto i = 0; i < F.cols(); ++i) {
            MatrixXd vert(vDim, 3);
            vert.col(0) = V.col(F(0, i));
            vert.col(1) = V.col(F(1, i));
            vert.col(2) = V.col(F(2, i));

            Vector3d r = restD.col(i);

            energyList(i) = liftedTriArea(vert, r);
        }

        return energyList.sum();
    }

    void getSquaredLiftedAreaPolyCoeff(const VectorXd &x, const VectorXd &p, MatrixXd &mat) {
        //compute mat such that the squared lifted area of triangle i in mesh (x+s*p) equals
        // mat(0,i) + mat(1,i)*s + mat(2,i)*s^2 + mat(3,i)*s^3 + mat(4,i)*s^4

        // init V and P
        update_V(x);

        MatrixXd P = MatrixXd::Zero(vDim, V.cols());
        for (auto i = 0; i < freeI.size(); ++i) {
            for (auto j = 0; j < vDim; ++j) {
                P(j, freeI(i)) = p[i * vDim + j];
            }
        }

        // compute coefficient matrix
        mat.resize(5, F.cols());

        for (auto i = 0; i < F.cols(); ++i) {
            MatrixXd vert(vDim, 3);
            vert.col(0) = V.col(F(0, i));
            vert.col(1) = V.col(F(1, i));
            vert.col(2) = V.col(F(2, i));

            MatrixXd searchVec(vDim, 3);
            searchVec.col(0) = P.col(F(0, i));
            searchVec.col(1) = P.col(F(1, i));
            searchVec.col(2) = P.col(F(2, i));

            Vector3d r = restD.col(i);

            mat.col(i) = squaredLiftedTriAreaPolyCoeff(vert, r, searchVec);
        }
    }

    void getLiftedEnergyGrad(const VectorXd &x, double &energy, VectorXd &grad) {
        update_V(x);

        // compute lifted energy and gradient
        energy = 0.0;
        MatrixXd fullGrad = MatrixXd::Zero(V.rows(), V.cols());

        for (auto i = 0; i < F.cols(); ++i) {
            int i1, i2, i3;
            i1 = F(0, i);
            i2 = F(1, i);
            i3 = F(2, i);

            MatrixXd vert(vDim, 3);
            vert.col(0) = V.col(i1);
            vert.col(1) = V.col(i2);
            vert.col(2) = V.col(i3);
            Vector3d r = restD.col(i);

            double f;
            MatrixXd g;
            liftedTriAreaGrad(vert, r, f, g);
            energy += f;

            fullGrad.col(i1) += g.col(0);
            fullGrad.col(i2) += g.col(1);
            fullGrad.col(i3) += g.col(2);
        }

        // get free gradient
        grad.resize(x.size());
        for (auto i = 0; i < freeI.size(); ++i) {
            for (int j = 0; j < vDim; ++j) {
                grad(i * vDim + j) = fullGrad(j, freeI(i));
            }
        }

    }

    void getLiftedEnergyGradLaplacian(const VectorXd &x, double &energy, VectorXd &grad, SpMat &Lap) {
        update_V(x);

        // compute energy, gradient and Laplacian
        energy = 0.0;
        MatrixXd fullGrad = MatrixXd::Zero(V.rows(), V.cols());

        std::vector<eigenT> tripletList;
        tripletList.reserve(8 * vDim * freeI.size());

        for (auto i = 0; i < F.cols(); ++i) {
            int i1, i2, i3;
            i1 = F(0, i);
            i2 = F(1, i);
            i3 = F(2, i);

            MatrixXd vert(vDim, 3);
            vert.col(0) = V.col(i1);
            vert.col(1) = V.col(i2);
            vert.col(2) = V.col(i3);
            Vector3d r = restD.col(i);

            double f;
            MatrixXd g;
            Matrix3d lap;
            liftedTriAreaGradLaplacian(vert, r, f, g, lap);
            energy += f;

            fullGrad.col(i1) += g.col(0);
            fullGrad.col(i2) += g.col(1);
            fullGrad.col(i3) += g.col(2);


            Vector3i indices = F_free.col(i);
            for (int j = 0; j < 3; ++j) {
                int idx_j = indices(j);
                for (int k = 0; k < 3; ++k) {
                    int idx_k = indices(k);
                    if (idx_j != -1 && idx_k != -1) {
                        double lap_jk = lap(j, k);
                        for (int l = 0; l < vDim; ++l) {
                            tripletList.emplace_back(idx_j * vDim + l, idx_k * vDim + l, lap_jk);
                        }
                    }
                }
            }

        }

        // get free gradient
        grad.resize(x.size());
        for (auto i = 0; i < freeI.size(); ++i) {
            for (int j = 0; j < vDim; ++j) {
                grad(i * vDim + j) = fullGrad(j, freeI(i));
            }
        }

        // get free Laplacian
        Lap.resize(vDim * freeI.size(), vDim * freeI.size());
        Lap.setFromTriplets(tripletList.begin(), tripletList.end());

    }

    void
    getLiftedEnergyGradHessian(const VectorXd &x, double &energy, VectorXd &energyList, VectorXd &grad, SpMat &Hess) {
        update_V(x);

        // compute energy, gradient and Hessian
        energy = 0.0;
        energyList.resize(F.cols());

        MatrixXd fullGrad = MatrixXd::Zero(V.rows(), V.cols());

        std::vector<eigenT> tripletList(3 * 3 * vDim * vDim * F.cols());


#pragma omp parallel
#pragma omp for
        for (auto i = 0; i < F.cols(); ++i) {
            int i1, i2, i3;
            i1 = F(0, i);
            i2 = F(1, i);
            i3 = F(2, i);

            MatrixXd vert(vDim, 3);
            vert.col(0) = V.col(i1);
            vert.col(1) = V.col(i2);
            vert.col(2) = V.col(i3);
            Vector3d r = restD.col(i);

            double f;
            MatrixXd g;
            MatrixXd hess;
            liftedTriAreaGradHessian(vert, r, f, g, hess);
            energyList(i) = f;

#pragma omp critical
            {
                fullGrad.col(i1) += g.col(0);
                fullGrad.col(i2) += g.col(1);
                fullGrad.col(i3) += g.col(2);
            }

            int current_index = i * 3 * 3 * vDim * vDim;
            Vector3i indices = F_free.col(i);
            for (int j = 0; j < 3; ++j) {
                int idx_j = indices(j);
                for (int k = 0; k < 3; ++k) {
                    int idx_k = indices(k);
                    if (idx_j != -1 && idx_k != -1) {
                        for (int l = 0; l < vDim; ++l) {
                            for (int n = 0; n < vDim; ++n) {
                                tripletList[current_index] = eigenT(idx_j * vDim + l, idx_k * vDim + n,
                                                                    hess(j * vDim + l, k * vDim + n));
                                ++current_index;
                            }
                        }
                    }
                }
            }
        }

        // get total energy
        energy = energyList.sum();


        // get free gradient
        grad.resize(x.size());
        for (auto i = 0; i < freeI.size(); ++i) {
            for (int j = 0; j < vDim; ++j) {
                grad(i * vDim + j) = fullGrad(j, freeI(i));
            }
        }

        // get free Hessian
        Hess.resize(vDim * freeI.size(), vDim * freeI.size());
        Hess.setFromTriplets(tripletList.begin(), tripletList.end());
    }


    void getLiftedEnergyGradProjectedHessian(const VectorXd &x, double &energy, VectorXd &energyList, VectorXd &grad,
                                             SpMat &Hess) {
        update_V(x);

        // compute energy, gradient and Hessian
        energyList.resize(F.cols());

        MatrixXd fullGrad = MatrixXd::Zero(V.rows(), V.cols());

        std::vector<eigenT> tripletList(3 * 3 * vDim * vDim * F.cols());

        // triangle-wise Hessian of signed area
        // this is used later in the PSD projection step
        MatrixXd signedHess(3 * 2, 3 * 2);
        signedHess << 0.0, 0.0, 0.0, 0.5, 0.0, -0.5,
                0.0, 0.0, -0.5, 0.0, 0.5, 0.0,
                0.0, -0.5, 0.0, 0.0, 0.0, 0.5,
                0.5, 0.0, 0.0, 0.0, -0.5, 0.0,
                0.0, 0.5, 0.0, -0.5, 0.0, 0.0,
                -0.5, 0.0, 0.5, 0.0, 0.0, 0.0;

        //

#pragma omp parallel
#pragma omp for
        for (auto i = 0; i < F.cols(); ++i) {
            int i1, i2, i3;
            i1 = F(0, i);
            i2 = F(1, i);
            i3 = F(2, i);

            MatrixXd vert(vDim, 3);
            vert.col(0) = V.col(i1);
            vert.col(1) = V.col(i2);
            vert.col(2) = V.col(i3);
            Vector3d r = restD.col(i);

            double f;
            MatrixXd g;
            MatrixXd hess;
            liftedTriAreaGradHessian(vert, r, f, g, hess);
            energyList(i) = f;

#pragma omp critical
            {
                fullGrad.col(i1) += g.col(0);
                fullGrad.col(i2) += g.col(1);
                fullGrad.col(i3) += g.col(2);
            }

            //project hess to PSD

            // modify Hessian before PSD projection


            // policy 1
//			 double signed_area = tri_signed_area(vert.col(0),vert.col(1),vert.col(2));
//			 if (signed_area > 0.0) hess -= signedHess;

            // policy 3
            // double signed_area = tri_signed_area(vert.col(0),vert.col(1),vert.col(2));
            // if (signed_area < 0.0) hess += signedHess;
            // else if (signed_area > 0.0) hess -= signedHess;

            // policy 5
//			hess -= signedHess;


            Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(hess);
            VectorXd eigenVals = eigenSolver.eigenvalues();
            for (auto j = 0; j < eigenVals.size(); ++j) {
                if (eigenVals(j) < 0.0) {
                    eigenVals(j) = 0.0;
                }
            }
            MatrixXd eigenVecs = eigenSolver.eigenvectors();
            hess = eigenVecs * (eigenVals.asDiagonal()) * eigenVecs.transpose();
            //end project hess to PSD

            int current_index = i * 3 * 3 * vDim * vDim;
            Vector3i indices = F_free.col(i);
            for (int j = 0; j < 3; ++j) {
                int idx_j = indices(j);
                for (int k = 0; k < 3; ++k) {
                    int idx_k = indices(k);
                    if (idx_j != -1 && idx_k != -1) {
                        for (int l = 0; l < vDim; ++l) {
                            for (int n = 0; n < vDim; ++n) {
                                tripletList[current_index] = eigenT(idx_j * vDim + l, idx_k * vDim + n,
                                                                    hess(j * vDim + l, k * vDim + n));
                                ++current_index;
                            }
                        }
                    }
                }
            }


        }

        // get total energy
        energy = energyList.sum();


        // get free gradient
        grad.resize(x.size());
        for (auto i = 0; i < freeI.size(); ++i) {
            for (int j = 0; j < vDim; ++j) {
                grad(i * vDim + j) = fullGrad(j, freeI(i));
            }
        }

        // add small positive values to the diagonal of Hessian
        for (auto i = 0; i < vDim * freeI.size(); ++i) {
            //			tripletList.push_back(eigenT(i,i,1e-8));
            tripletList.emplace_back(i, i, 1e-8);
        }

        // get free Hessian
        Hess.resize(vDim * freeI.size(), vDim * freeI.size());
        Hess.setFromTriplets(tripletList.begin(), tripletList.end());

    }


    void lineSearch(VectorXd &x, const VectorXd &p, double &step_size, const VectorXd &grad,
                    double energy, const VectorXd &energyList,
                    double &energy_next,
                    double shrink, double gamma) {
        double gp = gamma * grad.transpose() * p;
        VectorXd x_next = x + step_size * p;
        VectorXd energyList_next;
        energy_next = getLiftedEnergy(x_next, energyList_next);

        VectorXd energy_diff_list = energyList_next - energyList;
        double energy_diff = energy_diff_list.sum();

        while (energy_diff > step_size * gp) {
            step_size *= shrink;
            x_next = x + step_size * p;
            energy_next = getLiftedEnergy(x_next, energyList_next);

            energy_diff_list = energyList_next - energyList;
            energy_diff = energy_diff_list.sum();

            std::cout << energy_diff << std::endl;
        }
        x = x_next;
    }

    void lineSearch2(VectorXd &x, const VectorXd &p, double &step_size, const VectorXd &grad,
                     double energy, const VectorXd &energyList,
                     double &energy_next,
                     double shrink, double gamma) {
        double gp = gamma * grad.transpose() * p;

        //first iter
        VectorXd x_next = x + step_size * p;
        VectorXd energyList_next;
        energy_next = getLiftedEnergy(x_next, energyList_next);
        VectorXd energy_diff_list = energyList_next - energyList;
        double energy_diff = energy_diff_list.sum();

        if (energy_diff <= step_size * gp) {
            x = x_next;
            return;
        }

        //need more than one iter
        MatrixXd coeff;
        getSquaredLiftedAreaPolyCoeff(x, p, coeff);
        coeff.transposeInPlace();
        VectorXd c0 = coeff.col(0);
        VectorXd c1 = coeff.col(1);
        VectorXd c2 = coeff.col(2);
        VectorXd c3 = coeff.col(3);
        VectorXd c4 = coeff.col(4);

        do {
            step_size *= shrink;

            energyList_next = c0 +
                              step_size * (c1 +
                                           step_size * (c2 +
                                                        step_size * (c3 +
                                                                     step_size * c4)));
            //energyList_next = energyList_next.array().sqrt();
            energyList_next = energyList_next.array().abs().sqrt();

            energy_diff_list = energyList_next - energyList;
            energy_diff = energy_diff_list.sum();
        } while (energy_diff > step_size * gp);

        x = x + step_size * p;
        energy_next = energyList_next.sum();
    }

};


void Laplacian_precondition_gradient_descent(LiftedFormulation &formulation, VectorXd &x, int maxIter) {
    double energy;
    VectorXd grad(x.size());
    SpMat mat(x.size(), x.size());

    //first iter: initialize solver
    formulation.getLiftedEnergyGradLaplacian(x, energy, grad, mat);

    CholmodSolver solver;
    solver.analyzePattern(mat);

    solver.factorize(mat);
    if (solver.info() != Success) {
        cout << "iter 0: decomposition failed" << endl;
        return;
    }
    VectorXd p = solver.solve(-grad);
    if (solver.info() != Success) {
        cout << "iter 0: solving failed" << endl;
        return;
    }
    x += p;


    for (int i = 1; i < maxIter; ++i) {
        formulation.getLiftedEnergyGradLaplacian(x, energy, grad, mat);

        solver.factorize(mat);
        if (solver.info() != Success) {
            cout << "iter " << i << ": decomposition failed" << endl;
            return;
        }
        VectorXd p = solver.solve(-grad);
        if (solver.info() != Success) {
            cout << "iter " << i << ": solving failed" << endl;
            return;
        }
        x += p;

    }
}


void projected_Newton(LiftedFormulation &formulation, VectorXd &x, SolverOptionManager &options, double shrink = 0.7) {
    //handle options
    //todo: xtol
    double ftol_rel = options.ftol_rel;
    double ftol_abs = options.ftol_abs;
    double gtol_abs = options.gtol_abs;
    int maxIter = options.maxeval;
    //
    std::string stopCode = options.stopCode;
    //
    bool record_vert = false;
    std::vector<MatrixXd> &vertRecord = options.vertRecord;
    if (options.record_vert) {
        record_vert = true;
        vertRecord.resize(0);
    }
    //
    bool record_energy = false;
    std::vector<double> &energyRecord = options.energyRecord;
    if (options.record_energy) {
        record_energy = true;
        energyRecord.resize(0);
    }
    //
    bool record_minArea = false;
    std::vector<double> &minAreaRecord = options.minAreaRecord;
    if (options.record_minArea) {
        record_minArea = true;
        minAreaRecord.resize(0);
    }
    //
    bool record_gradient = false;
    std::vector<VectorXd> &gradientRecord = options.gradientRecord;
    if (options.record_gradient) {
        record_gradient = true;
        gradientRecord.resize(0);
    }
    //
    bool record_gradientNorm = false;
    std::vector<double> &gradientNormRecord = options.gradientNormRecord;
    if (options.record_gradientNorm) {
        record_gradientNorm = true;
        gradientNormRecord.resize(0);
    }
    //
    bool record_searchDirection = false;
    std::vector<VectorXd> &searchDirectionRecord = options.searchDirectionRecord;
    if (options.record_searchDirection) {
        record_searchDirection = true;
        searchDirectionRecord.resize(0);
    }
    //
    bool record_searchNorm = false;
    std::vector<double> &searchNormRecord = options.searchNormRecord;
    if (options.record_searchNorm) {
        record_searchNorm = true;
        searchNormRecord.resize(0);
    }
    //
    bool record_stepSize = false;
    std::vector<double> &stepSizeRecord = options.stepSizeRecord;
    if (options.record_stepSize) {
        record_stepSize = true;
        stepSizeRecord.resize(0);
    }
    //
    bool record_stepNorm = false;
    std::vector<double> &stepNormRecord = options.stepNormRecord;
    if (options.record_stepNorm) {
        record_stepNorm = true;
        stepNormRecord.resize(0);
    }
    //
    bool save_vert = options.save_vert;
    //handle options end

    //
    double energy;
    VectorXd energyList;
    VectorXd grad(x.size());
    SpMat mat(x.size(), x.size());

    double energy_next;

    //first iter: initialize solver
    formulation.getLiftedEnergyGradProjectedHessian(x, energy, energyList, grad, mat);

    // solver step monitor
    if (record_vert) vertRecord.push_back(formulation.V);
    if (save_vert) exportMatrix(options.resFile + "_vert_iter0", formulation.V);
    if (record_energy) energyRecord.push_back(energy);
    if (record_gradient) gradientRecord.push_back(grad);
    if (record_gradientNorm) gradientNormRecord.push_back(grad.norm());
    if (record_minArea || stopCode == "all_good") {
        double minA = computeMinSignedArea(formulation.V, formulation.F);
        if (record_minArea) minAreaRecord.push_back(minA);
        if ((stopCode == "all_good") && (minA > 0.0)) return;
    }
    // solver step monitor end

    //check gtol
    if (grad.norm() < gtol_abs) return;

    // initialize solver
    CholmodSolver solver;
    solver.analyzePattern(mat);
    // initialize solver end

    solver.factorize(mat);
    if (solver.info() != Success) {
        cout << "iter 0: decomposition failed" << endl;
        return;
    }
    VectorXd p = solver.solve(-grad);
    if (solver.info() != Success) {
        cout << "iter 0: solving failed" << endl;
        return;
    }
    if (record_searchDirection) searchDirectionRecord.push_back(p);
    if (record_searchNorm) searchNormRecord.push_back(p.norm());

    // backtracking line search
    double step_size = 1.0;
    formulation.lineSearch2(x, p, step_size, grad, energy, energyList, energy_next, shrink, 1e-4);
    //
    if (record_stepSize) stepSizeRecord.push_back(step_size);
    if (record_stepNorm) stepNormRecord.push_back(p.norm() * step_size);
    //check ftol
    if (fabs(energy_next - energy) < ftol_abs) return;
    if (fabs((energy_next - energy) / energy) < ftol_rel) return;


    for (int i = 1; i < maxIter; ++i) {
        formulation.getLiftedEnergyGradProjectedHessian(x, energy, energyList, grad, mat);

        // solver step monitor
        if (record_vert) vertRecord.push_back(formulation.V);
        if (save_vert) exportMatrix(options.resFile + "_vert_iter" + std::to_string(i), formulation.V);
        if (record_energy) energyRecord.push_back(energy);
        if (record_gradient) gradientRecord.push_back(grad);
        if (record_gradientNorm) gradientNormRecord.push_back(grad.norm());
        if (record_minArea || stopCode == "all_good") {
            double minA = computeMinSignedArea(formulation.V, formulation.F);
            if (record_minArea) minAreaRecord.push_back(minA);
            if ((stopCode == "all_good") && (minA > 0.0)) return;
        }
        // solver step monitor end

        //check gtol
        if (grad.norm() < gtol_abs) return;

        solver.factorize(mat);
        if (solver.info() != Success) {
            cout << "iter " << i << ": decomposition failed" << endl;
            return;
        }
        VectorXd p = solver.solve(-grad);
        if (solver.info() != Success) {
            cout << "iter " << i << ": solving failed" << endl;
            return;
        }
        if (record_searchDirection) searchDirectionRecord.push_back(p);
        if (record_searchNorm) searchNormRecord.push_back(p.norm());

        // backtracking line search
        double step_size = 1.0;
        formulation.lineSearch2(x, p, step_size, grad, energy, energyList, energy_next, shrink, 1e-4);

        //
        if (record_stepSize) stepSizeRecord.push_back(step_size);
        if (record_stepNorm) stepNormRecord.push_back(p.norm() * step_size);
        //check ftol
        if (fabs(energy_next - energy) < ftol_abs) return;
        if (fabs((energy_next - energy) / energy) < ftol_rel) return;
    }
}


//export result
bool exportResult(const char *filename, LiftedFormulation &formulation, const VectorXd &x,
                  const SolverOptionManager &options) {
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        std::cerr << "Failed to open " << filename << "!" << std::endl;
        return false;
    }

    //precision of output
    typedef std::numeric_limits<double> dbl;
    out_file.precision(dbl::max_digits10);

    //update V
    formulation.update_V(x);
    const MatrixXd &V = formulation.V;

    //write the file
    unsigned nv = V.cols();
    unsigned ndim = V.rows();
    out_file << "resV " << nv << " " << ndim << "\n";
    for (auto i = 0; i < nv; ++i) {
        for (auto j = 0; j < ndim; ++j) {
            out_file << V(j, i) << " ";
        }
    }
    out_file << std::endl;


    if (options.record_vert) {
        const std::vector<MatrixXd> &vertRecord = options.vertRecord;
        unsigned n_record = vertRecord.size();
        out_file << "vert " << n_record << " " << nv << " " << ndim << std::endl;
        for (auto i = 0; i < n_record; ++i) {
            for (auto j = 0; j < nv; ++j) {
                for (auto k = 0; k < ndim; ++k) {
                    out_file << vertRecord[i](k, j) << " ";
                }
            }
        }
        out_file << std::endl;
    }


    if (options.record_energy) {
        const std::vector<double> &energyRecord = options.energyRecord;
        unsigned n_record = energyRecord.size();
        out_file << "energy " << n_record << std::endl;
        for (auto i = 0; i < n_record; ++i) {
            out_file << energyRecord[i] << " ";
        }
        out_file << std::endl;
    }

    if (options.record_minArea) {
        const std::vector<double> &minAreaRecord = options.minAreaRecord;
        unsigned n_record = minAreaRecord.size();
        out_file << "minArea " << n_record << std::endl;
        for (auto i = 0; i < n_record; ++i) {
            out_file << minAreaRecord[i] << " ";
        }
        out_file << std::endl;
    }

    if (options.record_gradient) {
        const std::vector<VectorXd> &gradientRecord = options.gradientRecord;
        unsigned n_record = gradientRecord.size();
        unsigned n_free = formulation.freeI.size();
        out_file << "gradient " << n_record << " " << n_free << " " << ndim << std::endl;
        for (auto i = 0; i < n_record; ++i) {
            for (auto j = 0; j < n_free; ++j) {
                for (auto k = 0; k < ndim; ++k) {
                    out_file << gradientRecord[i](j * ndim + k) << " ";
                }
            }
        }
        out_file << std::endl;
    }

    if (options.record_gradientNorm) {
        const std::vector<double> &gradientNormRecord = options.gradientNormRecord;
        unsigned n_record = gradientNormRecord.size();
        out_file << "gNorm " << n_record << std::endl;
        for (auto i = 0; i < n_record; ++i) {
            out_file << gradientNormRecord[i] << " ";
        }
        out_file << std::endl;
    }

    if (options.record_searchDirection) {
        const std::vector<VectorXd> &searchDirectionRecord = options.searchDirectionRecord;
        unsigned n_record = searchDirectionRecord.size();
        unsigned n_free = formulation.freeI.size();
        out_file << "searchDirection " << n_record << " " << n_free << " " << ndim << std::endl;
        for (auto i = 0; i < n_record; ++i) {
            for (auto j = 0; j < n_free; ++j) {
                for (auto k = 0; k < ndim; ++k) {
                    out_file << searchDirectionRecord[i](j * ndim + k) << " ";
                }
            }
        }
        out_file << std::endl;
    }

    if (options.record_searchNorm) {
        const std::vector<double> &searchNormRecord = options.searchNormRecord;
        unsigned n_record = searchNormRecord.size();
        out_file << "searchNorm " << n_record << std::endl;
        for (auto i = 0; i < n_record; ++i) {
            out_file << searchNormRecord[i] << " ";
        }
        out_file << std::endl;
    }

    if (options.record_stepSize) {
        const std::vector<double> &stepSizeRecord = options.stepSizeRecord;
        unsigned n_record = stepSizeRecord.size();
        out_file << "stepSize " << n_record << std::endl;
        for (auto i = 0; i < n_record; ++i) {
            out_file << stepSizeRecord[i] << " ";
        }
        out_file << std::endl;
    }

    if (options.record_stepNorm) {
        const std::vector<double> &stepNormRecord = options.stepNormRecord;
        unsigned n_record = stepNormRecord.size();
        out_file << "stepNorm " << n_record << std::endl;
        for (auto i = 0; i < n_record; ++i) {
            out_file << stepNormRecord[i] << " ";
        }
        out_file << std::endl;
    }

    //
    out_file.close();
    return true;
}


int main(int argc, char const *argv[]) {
    const char *dataFile = (argc > 1) ? argv[1] : "./input";
    const char *optFile = (argc > 2) ? argv[2] : "./solver_options";
    const char *resFile = (argc > 3) ? argv[3] : "./result";

    //import data
    std::vector<std::vector<double> > raw_restV;
    std::vector<std::vector<double> > raw_initV;
    std::vector<std::vector<int> > raw_F;
    std::vector<int> raw_handles;

    bool succeed = importData(dataFile, raw_restV, raw_initV, raw_F, raw_handles);
    if (!succeed) {
        std::cout << "usage: findInjective_PN [inputFile] [solverOptionsFile] [resultFile]" << std::endl;
        return -1;
    }

    //convert raw data to Eigen data
    MatrixXd restV = toEigenMatrix(raw_restV);
    MatrixXd initV = toEigenMatrix(raw_initV);
    MatrixXi F = toEigenMatrix(raw_F);
    VectorXi handles = toEigenVector(raw_handles);

    //import options
    SolverOptionManager options(optFile, resFile);

    //
    LiftedFormulation myLifted(restV, initV, F, handles, options.form, options.alphaRatio, options.alpha);
    VectorXd x = myLifted.x0;

    //projected newton
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    projected_Newton(myLifted, x, options);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << " [microseconds]" << std::endl;

    exportResult(resFile, myLifted, x, options);


    return 0;
}
