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

bool importData(const char* filename,
	std::vector<std::vector<double> > &restV,
	std::vector<std::vector<double> > &initV,
	std::vector<std::vector<unsigned> > &F,
	std::vector<unsigned> &handles,
	std::string &form,
	double &alpha)
{
	std::ifstream in_file(filename);

	if (! in_file.is_open()) {
        std::cerr << "Failed to open " << filename << "!" << std::endl;
        return false;
    }

    //read the file
    size_t n, ndim;
    // restV
    in_file >> n >> ndim;
    restV.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
    	std::vector<double> v(ndim);
    	for (size_t j = 0; j < ndim; ++j)
    	{
    		in_file >> v[j];
    	}
    	restV[i] = v;
    }

    //initV
    in_file >> n >> ndim;
    initV.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
    	std::vector<double> v(ndim);
    	for (size_t j = 0; j < ndim; ++j)
    	{
    		in_file >> v[j];
    	}
    	initV[i] = v;
    }

    //F
    size_t simplexSize;
    in_file >> n >> simplexSize;
    F.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
    	std::vector<unsigned> v(simplexSize);
    	for (size_t j = 0; j < simplexSize; ++j)
    	{
    		in_file >> v[j];
    	}
    	F[i] = v;
    }

    //handles
    in_file >> n;
    handles.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
    	unsigned v;
    	in_file >> v;
    	handles[i] = v;
    }

    //form
    in_file >> form;

    //alpha
    in_file >> alpha;

    in_file.close();

    return true;
}

// solver options
class SolverOptionManager
{
public:
	//default options
	SolverOptionManager():
	ftol_abs(1e-8), ftol_rel(1e-8), xtol_abs(1e-8), xtol_rel(1e-8), gtol_abs(1e-8),
	maxeval(1000), algorithm("ProjectedNewton"), stopCode("none"),
	/*record()*/ record_vert(false), record_energy(false), record_minArea(false),
	record_gradient(false), record_searchDirection(false), record_stepSize(false)
	{};
	//import options from file
	SolverOptionManager(const char* filename):
	ftol_abs(1e-8), ftol_rel(1e-8), xtol_abs(1e-8), xtol_rel(1e-8), gtol_abs(1e-8),
	maxeval(1000), algorithm("ProjectedNewton"), stopCode("none"),
	/*record()*/ record_vert(false), record_energy(false), record_minArea(false),
	record_gradient(false), record_searchDirection(false), record_stepSize(false)
	{
		if (!importOptions(filename))
		{
			std::cout << "SolverOptionManager Warn: default options are used." << std::endl;
		}
	};

	~SolverOptionManager() = default;

	double ftol_abs;
	double ftol_rel;
	double xtol_abs;
	double xtol_rel;
	double gtol_abs;  //todo: add gtol_abs. also need to change code in Mathematica
	int maxeval;
	std::string algorithm;
	std::string stopCode;


	//std::vector<std::string> record;
	bool record_vert;
	bool record_energy;
	bool record_minArea;
	bool record_gradient;
	bool record_searchDirection;
	bool record_stepSize;
	std::vector<MatrixXd> vertRecord;
	std::vector<double> energyRecord;
	std::vector<double> minAreaRecord;
	std::vector<VectorXd> gradientRecord;
	std::vector<VectorXd> searchDirectionRecord;
	std::vector<double> stepSizeRecord;



	void printOptions()
	{
		std::cout << "ftol_abs:\t"  <<  ftol_abs  << "\n";
		std::cout << "ftol_rel:\t"  <<  ftol_rel  << "\n";
		std::cout << "xtol_abs:\t"  <<  xtol_abs  << "\n";
		std::cout << "xtol_rel:\t"  <<  xtol_rel  << "\n";
		std::cout << "gtol_abs:\t"  <<  gtol_abs  << "\n";
		std::cout << "maxeval:\t"   <<  maxeval   << "\n";
		std::cout << "algorithm:\t" <<  algorithm << "\n";
		std::cout << "stopCode:\t"  <<  stopCode  << "\n";
		std::cout << "record:  \t"  <<  "{ ";
		if (record_vert)    std::cout << "vert ";
		if (record_energy)  std::cout << "energy ";
		if (record_minArea) std::cout << "minArea ";
		if (record_gradient) std::cout << "gradient ";
		if (record_searchDirection) std::cout << "searchDirection "; 
		if (record_stepSize)	std::cout << "stepSize ";
		std::cout << "}" << std::endl;


	}

	bool importOptions(const char* filename)
	{
		//open the data file
		std::ifstream in_file(filename);

		if (! in_file.is_open())
		{
			std::cerr << "Failed to open " << filename << "!" << std::endl;
			return false;
		}

		//read the file
		unsigned normal = 0;
		std::string optName;
		while(true)
		{
			in_file >> optName;
			if (optName != "ftol_abs")
			{
				normal = 1;
				break;
			}
			in_file >> ftol_abs;

			in_file >> optName;
			if (optName != "ftol_rel")
			{
				normal = 2;
				break;
			}
			in_file >> ftol_rel;

			in_file >> optName;
			if (optName != "xtol_abs")
			{
				normal = 3;
				break;
			}
			in_file >> xtol_abs;

			in_file >> optName;
			if (optName != "xtol_rel")
			{
				normal = 4;
				break;
			}
			in_file >> xtol_rel;

			in_file >> optName;
			if (optName != "gtol_abs")
			{
				normal = 100;
				break;
			}
			in_file >> gtol_abs;

			in_file >> optName;
			if (optName != "algorithm")
			{
				normal = 5;
				break;
			}
			in_file >> algorithm;

			in_file >> optName;
			if (optName != "maxeval")
			{
				normal = 6;
				break;
			}
			in_file >> maxeval;

			in_file >> optName;
			if (optName != "stopCode")
			{
				normal = 7;
				break;
			}
			in_file >> stopCode;

			in_file >> optName;
			if (optName != "record")
			{
				normal = 8;
				break;
			}
			size_t n;
			in_file >> n;
			std::string cur_record;
			//record.resize(n);
			for (size_t i = 0; i < n; ++i)
			{
				//in_file >> record[i];
				in_file >> cur_record;
				if (cur_record == "vert")
				{
					record_vert = true;
				}
				if (cur_record == "energy")
				{
					record_energy = true;
				}
				if (cur_record == "minArea")
				{
					record_minArea = true;
				}
				if (cur_record == "gradient")
				{
					record_gradient = true;
				}
				if (cur_record == "searchDirection")
				{
					record_searchDirection = true;
				}
				if (cur_record == "stepSize")
				{
					record_stepSize = true;
				}
			}

			break;
		}

		in_file.close();

		if (normal!=0)
		{
			std::cout << "Err:(" << normal << ") fail to import options from file. Check file format." << std::endl;
			return false;
		}

		return true;
	}

};


// helper functions
void computeSquaredEdgeLength(const MatrixXd& V,
	const MatrixXi& F,
	MatrixXd& D)
{
	int nf = F.cols();
	// int simplexSize = F.rows();
	// int n_edge = simplexSize * (simplexSize-1) / 2;
	int n_edge = 3;

	D.resize(n_edge,nf);
	for (int i = 0; i < nf; ++i)
	{
		auto v1 = V.col(F(0,i));
		auto v2 = V.col(F(1,i));
		auto v3 = V.col(F(2,i));
		auto e1 = v2 - v3;
		auto e2 = v3 - v1;
		auto e3 = v1 - v2;
		D(0,i) = e1.squaredNorm();
		D(1,i) = e2.squaredNorm();
		D(2,i) = e3.squaredNorm();
	}
}

inline double tri_signed_area(const Vector2d& p1, const Vector2d& p2, const Vector2d& p3)
{
	// input: 2D coordinates of 3 points of triangle
	// return: signed area of the triangle
	return 0.5 * (p3(0)*(p1(1)-p2(1)) + p1(0)*(p2(1)-p3(1)) + p2(0)*(p3(1)-p1(1)));
}

void computeSignedArea(const MatrixXd& V, const MatrixXi& F, VectorXd& areaList)
{
	// note: V must be a (2 * nv) matrix
	int nf = F.cols();
	areaList.resize(nf);
	for (int i = 0; i < nf; ++i)
	{
		const Vector2d& p1 = V.col(F(0,i));
		const Vector2d& p2 = V.col(F(1,i));
		const Vector2d& p3 = V.col(F(2,i));
		areaList(i) = tri_signed_area(p1,p2,p3);
	}
}

double computeMinSignedArea(const MatrixXd& V, const MatrixXi& F)
{
	VectorXd areaList;
	computeSignedArea(V,F,areaList);
	return areaList.minCoeff();
}

// Heron's formula and its derivatives
double HeronTriArea(double d1, double d2, double d3)
{
	return 0.25 * sqrt((d1+d2+d3)*(d1+d2+d3)-2*(d1*d1+d2*d2+d3*d3));
}

void HeronTriAreaGrad(double d1, double d2, double d3,
	double& area, Vector3d& grad)
{
	area = HeronTriArea(d1,d2,d3);
	double s = 1 / (16 * area);
	// grad(0) = d2 + d3 - d1;
	// grad(1) = d1 + d3 - d2;
	// grad(2) = d1 + d2 - d3;
	grad << d2+d3-d1, d1+d3-d2, d1+d2-d3;
	grad *= s;
}

void HeronTriAreaGradHessian(double d1, double d2, double d3,
	double& area, Vector3d& grad, Matrix3d& Hess)
{
	area = HeronTriArea(d1,d2,d3);
	double s = 1.0 / (16.0 * area);
	grad << d2+d3-d1, d1+d3-d2, d1+d2-d3;
	grad *= s;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			Hess(j,i) = grad(i) * grad(j);
		}
	}

	Hess /= (-area);

	Matrix3d t;
	t << -1.0,  1.0,  1.0,
		  1.0, -1.0,  1.0,
		  1.0,  1.0, -1.0;
	t *= s;

	Hess += t;
}

// Lifted triangle area and its derivatives

double liftedTriArea(const MatrixXd& vert, const Vector3d& r)
{
	auto v1 = vert.col(0);
	auto v2 = vert.col(1);
	auto v3 = vert.col(2);
	auto e1 = v2 - v3;
	auto e2 = v3 - v1;
	auto e3 = v1 - v2;
	double d1 = e1.squaredNorm() + r(0);
	double d2 = e2.squaredNorm() + r(1);
	double d3 = e3.squaredNorm() + r(2);
	return HeronTriArea(d1,d2,d3);
}

void liftedTriAreaGrad(const MatrixXd& vert, const Vector3d& r,
	double& area, MatrixXd& grad)
{
	auto v1 = vert.col(0);
	auto v2 = vert.col(1);
	auto v3 = vert.col(2);
	auto e1 = v2 - v3;
	auto e2 = v3 - v1;
	auto e3 = v1 - v2;
	double d1 = e1.squaredNorm() + r(0);
	double d2 = e2.squaredNorm() + r(1);
	double d3 = e3.squaredNorm() + r(2);

	Vector3d dAdD;
	HeronTriAreaGrad(d1,d2,d3,area,dAdD);

	double g1,g2,g3;
	g1 = dAdD(0);
	g2 = dAdD(1);
	g3 = dAdD(2);

	Matrix3d Lap;
	Lap <<  g2+g3, -g3, -g2,
			-g3, g1+g3, -g1,
			-g2, -g1, g1+g2;
	Lap *= 2.0;

	//note: grad has the same dimension as vert
	grad.resize(vert.rows(),vert.cols());
	grad = vert * Lap;
}

void liftedTriAreaGradLaplacian(const MatrixXd& vert, const Vector3d& r,
	double& area, MatrixXd& grad, Matrix3d& Lap)
{
	auto v1 = vert.col(0);
	auto v2 = vert.col(1);
	auto v3 = vert.col(2);
	auto e1 = v2 - v3;
	auto e2 = v3 - v1;
	auto e3 = v1 - v2;
	double d1 = e1.squaredNorm() + r(0);
	double d2 = e2.squaredNorm() + r(1);
	double d3 = e3.squaredNorm() + r(2);

	Vector3d dAdD;
	HeronTriAreaGrad(d1,d2,d3,area,dAdD);

	double g1,g2,g3;
	g1 = dAdD(0);
	g2 = dAdD(1);
	g3 = dAdD(2);

	Lap <<  g2+g3, -g3, -g2,
			-g3, g1+g3, -g1,
			-g2, -g1, g1+g2;
	Lap *= 2.0;

	//note: grad has the same dimension as vert
	grad.resize(vert.rows(),vert.cols());
	grad = vert * Lap;
}

void liftedTriAreaGradHessian(const MatrixXd& vert, const Vector3d& r,
	double& area, MatrixXd& grad, MatrixXd& Hess)
{
	auto v1 = vert.col(0);
	auto v2 = vert.col(1);
	auto v3 = vert.col(2);
	auto e1 = v2 - v3;
	auto e2 = v3 - v1;
	auto e3 = v1 - v2;
	double d1 = e1.squaredNorm() + r(0);
	double d2 = e2.squaredNorm() + r(1);
	double d3 = e3.squaredNorm() + r(2);

	Vector3d dAdD;
	Matrix3d hAhD;
	HeronTriAreaGradHessian(d1,d2,d3,area,dAdD,hAhD);

	int vDim = v1.size();

	// dDdV
	std::vector<std::vector<VectorXd> > dDdV(3);
	for (int i = 0; i < 3; ++i)
	{
		dDdV[i].resize(3);
		dDdV[i][i] = VectorXd::Zero(vDim);
	}
	dDdV[0][1] =  2.0 * e1; dDdV[0][2] = -2.0 * e1;
	dDdV[1][0] = -2.0 * e2; dDdV[1][2] =  2.0 * e2;
	dDdV[2][0] =  2.0 * e3; dDdV[2][1] = -2.0 * e3;

	// Hess 1
	MatrixXd Hess1(3*vDim, 3*vDim);
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			MatrixXd Hij = MatrixXd::Zero(vDim,vDim);
			for (int k = 0; k < 3; ++k)
			{
				VectorXd gkj = VectorXd::Zero(vDim);
				for (int l = 0; l < 3; ++l)
				{
					gkj += (hAhD(k,l) * dDdV[l][j]);
				}
				Hij += dDdV[k][i] * gkj.transpose();
			}
			Hess1.block(i*vDim,j*vDim,vDim,vDim) = Hij;
		}
	}

	// Laplacian
	double g1,g2,g3;
	g1 = dAdD(0);
	g2 = dAdD(1);
	g3 = dAdD(2);

	Matrix3d Lap;
	Lap <<  g2+g3, -g3, -g2,
			-g3, g1+g3, -g1,
			-g2, -g1, g1+g2;
	Lap *= 2.0;

	//note: grad has the same dimension as vert
	grad.resize(vert.rows(),vert.cols());
	grad = vert * Lap;

	// Hess2
	MatrixXd Hess2(3*vDim,3*vDim);
	MatrixXd I = MatrixXd::Identity(vDim,vDim);
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			Hess2.block(i*vDim,j*vDim,vDim,vDim) = Lap(i,j) * I;
		}
	}

	//
	Hess.resize(3*vDim,3*vDim);
	Hess = Hess1 + Hess2;

}


class LiftedFormulation
{
public:
	LiftedFormulation(MatrixXd& restV, MatrixXd& initV, MatrixXi& restF,
		VectorXi& handles, const std::string& form, double alpha) :
	V(initV), F(restF)
	{
		// compute freeI
		int nV = V.cols();
		vDim = V.rows();

		std::vector<bool> freeQ(nV, true);
		for (auto i = 0; i < handles.size(); ++i)
		{
			freeQ[handles(i)] = false;
		}
		freeI.resize(nV - handles.size());
		int ii = 0;
		for (int i = 0; i < nV; ++i)
		{
			if (freeQ[i])
			{
				freeI[ii] = i;
				++ii;
			}
		}
		std::sort(freeI.data(),freeI.data()+freeI.size());


		// compute indexDict and F_free
		indexDict = VectorXi::Constant(nV,-1);
		for (auto i = 0; i < freeI.size(); ++i)
		{
			indexDict(freeI(i)) = i;
		}

		F_free.resize(F.rows(),F.cols());
		for (auto i = 0; i < F.cols(); ++i)
		{
			for (auto j = 0; j < F.rows(); ++j)
			{
				F_free(j,i) = indexDict(F(j,i));
			}
		}

		// compute restD
		double a = alpha; //for triangle mesh
		if (form == "harmonic")
		{
			computeSquaredEdgeLength(restV,F,restD);
			restD *= a;
		}
		else // tutte-uniform form
		{
			restD = MatrixXd::Constant(3,F.cols(),a);
		}

		// compute x0 from initV
		x0.resize(vDim * freeI.size());
		for (auto i = 0; i < freeI.size(); ++i)
		{
			int vi = freeI(i);
			for (int j = 0; j < vDim; ++j)
			{
				x0(i*vDim+j) = V(j,vi);
			}
		}



	}
	;

	~LiftedFormulation() = default;

	VectorXi freeI;   // indices of free vertices
	//int nV;         // number of vertices
	int vDim;         // dimension of target vertices
	MatrixXi F;       // V indices of triangles
	MatrixXd restD;   // squared edge lengths of rest/auxiliary mesh

	VectorXi indexDict; // map: V index --> freeV index. If V(i) is not free, indexDict(i) = -1.
	MatrixXi F_free;    // freeV indices of triangles.

	VectorXd x0;  // initial variable vector
	MatrixXd V;   // current V of target mesh


	// x = Flatten(freeV)
	void update_V(const VectorXd& x)
	{
		for (auto i = 0; i < freeI.size(); ++i)
		{
			for (int j = 0; j < vDim; ++j)
			{
				V(j,freeI(i)) = x[i*vDim + j];
			}
		}
	}


//	double getLiftedEnergy(const VectorXd& x)
    long double getLiftedEnergy(const VectorXd& x)
	{
		// update V
		for (auto i = 0; i < freeI.size(); ++i)
		{
			for (int j = 0; j < vDim; ++j)
			{
				V(j,freeI(i)) = x[i*vDim + j];
			}
		}

		// compute lifted energy

		//debug
		long double a = 0.0;
//		double minA = 100.0;
//		double maxA = 0.0;
		//

//		double energy = 0.0;
        long double energy = 0.0;
		for (auto i = 0; i < F.cols(); ++i)
		{
			MatrixXd vert(vDim,3);
			vert.col(0) = V.col(F(0,i));
			vert.col(1) = V.col(F(1,i));
			vert.col(2) = V.col(F(2,i));

			Vector3d r = restD.col(i);

			//debug
			a = liftedTriArea(vert, r);
//			std::cout << a << " ";
//			minA = (a < minA) ? a : minA;
//			maxA = (a > maxA) ? a : maxA;

			energy += a;
			//
//			energy += liftedTriArea(vert, r);

		}
//		std::cout << std::endl;
//        std::cout << energy << std::endl;
		//debug
//		std::cout << "---------" << std::endl;
//		std::cout << "total lifted area:" << std::endl << energy << std::endl;
//		std::cout << "min lifted area: " << std::endl << minA << std::endl;
//        std::cout << "max lifted area: " << std::endl << maxA << std::endl;
//        std::cout << "---------" << std::endl;
		//

		return energy;
	}

	void getLiftedEnergyGrad(const VectorXd& x, double& energy, VectorXd& grad)
	{
		// update V
		for (auto i = 0; i < freeI.size(); ++i)
		{
			for (int j = 0; j < vDim; ++j)
			{
				V(j,freeI(i)) = x[i*vDim + j];
			}
		}

		// compute lifted energy and gradient
		energy = 0.0;
		MatrixXd fullGrad = MatrixXd::Zero(V.rows(),V.cols());

		for (auto i = 0; i < F.cols(); ++i)
		{
			int i1,i2,i3;
			i1 = F(0,i);
			i2 = F(1,i);
			i3 = F(2,i);

			MatrixXd vert(vDim,3);
			vert.col(0) = V.col(i1);
			vert.col(1) = V.col(i2);
			vert.col(2) = V.col(i3);
			Vector3d r = restD.col(i);

			double f;
			MatrixXd g;
			liftedTriAreaGrad(vert,r,f,g);
			energy += f;

			fullGrad.col(i1) += g.col(0);
			fullGrad.col(i2) += g.col(1);
			fullGrad.col(i3) += g.col(2);
		}

		// get free gradient
		grad.resize(x.size());
		for (auto i = 0; i < freeI.size(); ++i)
		{
			for (int j = 0; j < vDim; ++j)
			{
				grad(i*vDim + j) = fullGrad(j,freeI(i));
			}
		}

	}

	void getLiftedEnergyGradLaplacian(const VectorXd& x, double& energy, VectorXd& grad, SpMat& Lap)
	{
		// update V
		for (auto i = 0; i < freeI.size(); ++i)
		{
			for (int j = 0; j < vDim; ++j)
			{
				V(j,freeI(i)) = x[i*vDim + j];
			}
		}

		// compute energy, gradient and Laplacian
		energy = 0.0;
		MatrixXd fullGrad = MatrixXd::Zero(V.rows(),V.cols());

		std::vector<eigenT> tripletList;
		tripletList.reserve(8*vDim*freeI.size());

		for (auto i = 0; i < F.cols(); ++i)
		{
			int i1,i2,i3;
			i1 = F(0,i);
			i2 = F(1,i);
			i3 = F(2,i);

			MatrixXd vert(vDim,3);
			vert.col(0) = V.col(i1);
			vert.col(1) = V.col(i2);
			vert.col(2) = V.col(i3);
			Vector3d r = restD.col(i);

			double f;
			MatrixXd g;
			Matrix3d lap;
			liftedTriAreaGradLaplacian(vert,r,f,g,lap);
			energy += f;

			fullGrad.col(i1) += g.col(0);
			fullGrad.col(i2) += g.col(1);
			fullGrad.col(i3) += g.col(2);


			Vector3i indices = F_free.col(i);
			for (int j = 0; j < 3; ++j)
			{
				int idx_j = indices(j);
				for (int k = 0; k < 3; ++k)
				{
					int idx_k = indices(k);
					if (idx_j!=-1 && idx_k!=-1) {
						double lap_jk = lap(j,k);
						for (int l = 0; l < vDim; ++l)
						{
//							tripletList.push_back(eigenT(idx_j*vDim+l,idx_k*vDim+l,lap_jk));
							tripletList.emplace_back(idx_j*vDim+l,idx_k*vDim+l,lap_jk);
						}
					}
				}
			}

		}

		// get free gradient
		grad.resize(x.size());
		for (auto i = 0; i < freeI.size(); ++i)
		{
			for (int j = 0; j < vDim; ++j)
			{
				grad(i*vDim + j) = fullGrad(j,freeI(i));
			}
		}

		// get free Laplacian
		Lap.resize(vDim * freeI.size(), vDim * freeI.size());
		Lap.setFromTriplets(tripletList.begin(), tripletList.end());

	}

//	void getLiftedEnergyGradHessian(const VectorXd& x, double& energy, VectorXd& grad, SpMat& Hess)
    void getLiftedEnergyGradHessian(const VectorXd& x, long double& energy, VectorXd& grad, SpMat& Hess)
    {
		// update V
		for (auto i = 0; i < freeI.size(); ++i)
		{
			for (int j = 0; j < vDim; ++j)
			{
				V(j,freeI(i)) = x[i*vDim + j];
			}
		}

		// compute energy, gradient and Hessian
		energy = 0.0;
//		std::vector<double> energyList(F.cols());
        std::vector<long double> energyList(F.cols());

        MatrixXd fullGrad = MatrixXd::Zero(V.rows(),V.cols());

		std::vector<eigenT> tripletList(3*3*vDim*vDim*F.cols());

		// triangle-wise Hessian of signed area
		// this is used later in the PSD projection step
		MatrixXd signedHess(3*2,3*2);
		signedHess << 	0.0, 0.0, 0.0, 0.5, 0.0, -0.5,
						0.0, 0.0, -0.5, 0.0, 0.5, 0.0,
						0.0, -0.5, 0.0, 0.0, 0.0, 0.5,
						0.5, 0.0, 0.0, 0.0, -0.5, 0.0,
						0.0, 0.5, 0.0, -0.5, 0.0, 0.0,
						-0.5, 0.0, 0.5, 0.0, 0.0, 0.0;

		//

		#pragma omp parallel
		#pragma omp for
		for (auto i = 0; i < F.cols(); ++i)
		{
			// cout << "face " << i << ": " << std::endl;
			int i1,i2,i3;
			i1 = F(0,i);
			i2 = F(1,i);
			i3 = F(2,i);

			MatrixXd vert(vDim,3);
			vert.col(0) = V.col(i1);
			vert.col(1) = V.col(i2);
			vert.col(2) = V.col(i3);
			Vector3d r = restD.col(i);

			double f;
			MatrixXd g;
			MatrixXd hess;
			liftedTriAreaGradHessian(vert,r,f,g,hess);
			energyList[i] = f;

			#pragma omp critical
			{
				fullGrad.col(i1) += g.col(0);
				fullGrad.col(i2) += g.col(1);
				fullGrad.col(i3) += g.col(2);
			}

			//project hess to PSD

			// modify Hessian before PSD projection
			

			// policy 1
			// double signed_area = tri_signed_area(vert.col(0),vert.col(1),vert.col(2));
			// if (signed_area > 0.0) hess -= signedHess;

			// policy 3
			// double signed_area = tri_signed_area(vert.col(0),vert.col(1),vert.col(2));
			// if (signed_area < 0.0) hess += signedHess;
			// else if (signed_area > 0.0) hess -= signedHess;

			// policy 5
			hess -= signedHess;


			Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(hess);
			VectorXd eigenVals = eigenSolver.eigenvalues();
			for (auto j = 0; j < eigenVals.size(); ++j)
			{
				if (eigenVals(j) < 0.0) {
					eigenVals(j) = 0.0;
				}
			}
			MatrixXd eigenVecs = eigenSolver.eigenvectors();
			hess = eigenVecs * (eigenVals.asDiagonal()) * eigenVecs.transpose();
			//end project hess to PSD

			int current_index = i*3*3*vDim*vDim;
			Vector3i indices = F_free.col(i);
			for (int j = 0; j < 3; ++j)
			{
				int idx_j = indices(j);
				for (int k = 0; k < 3; ++k)
				{
					int idx_k = indices(k);
					if (idx_j!=-1 && idx_k!=-1) {
						for (int l = 0; l < vDim; ++l)
						{
							for (int n = 0; n < vDim; ++n)
							{
								tripletList[current_index] = eigenT(idx_j*vDim+l,idx_k*vDim+n,hess(j*vDim+l,k*vDim+n));
								++current_index;
							}
						}
					}
				}
			}
			

		}

		// get total energy

		// debug
//		double minA = 100.0;
//		double maxA = 0.0;

		for (long double i : energyList)
		{
//		    minA = (i < minA) ? i : minA;
//		    maxA = (i > maxA) ? i : maxA;
//            std::cout << i <<  " ";

			energy += i;
		}
//		std::cout << std::endl;
//        std::cout << energy << std::endl;
//        std::cout << "---------" << std::endl;
//        std::cout << "total lifted area:" << std::endl << energy << std::endl;
//        std::cout << "min lifted area: " << std::endl << minA << std::endl;
//        std::cout << "max lifted area: " << std::endl << maxA << std::endl;
//        std::cout << "---------" << std::endl;
		//

		// get free gradient
		grad.resize(x.size());
		for (auto i = 0; i < freeI.size(); ++i)
		{
			for (int j = 0; j < vDim; ++j)
			{
				grad(i*vDim + j) = fullGrad(j,freeI(i));
			}
		}

		// add small positive values to the diagonal of Hessian
		for (auto i = 0; i < vDim * freeI.size(); ++i)
		{
//			tripletList.push_back(eigenT(i,i,1e-8));
			tripletList.emplace_back(i,i,1e-8);
		}

		// get free Hessian
		Hess.resize(vDim * freeI.size(), vDim * freeI.size());
		Hess.setFromTriplets(tripletList.begin(), tripletList.end());

	}

};





void Laplacian_precondition_gradient_descent(LiftedFormulation& formulation, VectorXd& x, int maxIter)
{
	double energy;
	VectorXd grad(x.size());
	SpMat mat(x.size(),x.size());

	//first iter: initialize solver
	formulation.getLiftedEnergyGradLaplacian(x,energy,grad,mat);

	CholmodSolver solver;
	solver.analyzePattern(mat);

	solver.factorize(mat);
	if(solver.info()!=Success) {
  		cout <<  "iter 0: decomposition failed" << endl;
  		return;
	}
	VectorXd p = solver.solve(-grad);
	if(solver.info()!=Success) {
  		cout << "iter 0: solving failed" << endl;
  		return;
	}
	x += p;


	for (int i = 1; i < maxIter; ++i)
	{
		formulation.getLiftedEnergyGradLaplacian(x,energy,grad,mat);

		solver.factorize(mat);
		if(solver.info()!=Success) {
	  		cout <<  "iter " << i << ": decomposition failed" << endl;
	  		return;
		}
		VectorXd p = solver.solve(-grad);
		if(solver.info()!=Success) {
	  		cout << "iter " << i << ": solving failed" << endl;
	  		return;
		}
		x += p;

	}
}


void projected_Newton(LiftedFormulation& formulation, VectorXd& x, SolverOptionManager& options, long double shrink = 0.7)
{
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
	std::vector<MatrixXd>& vertRecord = options.vertRecord;
	if (options.record_vert)
	{
		record_vert = true;
		vertRecord.resize(0);
	}
	//
	bool record_energy = false;
	std::vector<double>& energyRecord = options.energyRecord;
	if (options.record_energy)
	{
		record_energy = true;
		energyRecord.resize(0);
	}
	//
	bool record_minArea = false;
	std::vector<double>& minAreaRecord = options.minAreaRecord;
	if (options.record_minArea)
	{
		record_minArea = true;
		minAreaRecord.resize(0);
	}
	//
	bool record_gradient = false;
	std::vector<VectorXd>& gradientRecord = options.gradientRecord;
	if (options.record_gradient)
	{
		record_gradient = true;
		gradientRecord.resize(0);
	}
	//
	bool record_searchDirection = false;
	std::vector<VectorXd>& searchDirectionRecord = options.searchDirectionRecord;
	if (options.record_searchDirection)
	{
		record_searchDirection = true;
		searchDirectionRecord.resize(0);
	}
	//
	bool record_stepSize = false;
	std::vector<double>& stepSizeRecord = options.stepSizeRecord;
	if (options.record_stepSize)
	{
		record_stepSize = true;
		stepSizeRecord.resize(0);
	}
	//handle options end

	//
	long double energy;
	VectorXd grad(x.size());
	SpMat mat(x.size(),x.size());

	VectorXd x_next(x.size());
	long double energy_next;


	//first iter: initialize solver
	formulation.getLiftedEnergyGradHessian(x,energy,grad,mat);

	// solver step monitor
	if (record_vert) vertRecord.push_back(formulation.V);
	if (record_energy) energyRecord.push_back(energy);
	if (record_gradient) gradientRecord.push_back(grad);
	if (record_minArea || stopCode == "all_good") {
		double minA = computeMinSignedArea(formulation.V, formulation.F);
		if (record_minArea) minAreaRecord.push_back(minA);
		if ((stopCode == "all_good") && (minA > 0.0)) return;
	}

	// solver step monitor end

	// initialize solver
	CholmodSolver solver;
	solver.analyzePattern(mat);
	// initialize solver end

	solver.factorize(mat);
	if(solver.info()!=Success) {
  		cout <<  "iter 0: decomposition failed" << endl;
  		return;
	}
	VectorXd p = solver.solve(-grad);
	if(solver.info()!=Success) {
  		cout << "iter 0: solving failed" << endl;
  		return;
	}
	if (record_searchDirection) searchDirectionRecord.push_back(p);

	// backtracking line search
	long double gp = 0.5 * grad.transpose() * p;
	long double step_size = 1.0;
	x_next = x + step_size * p;
	energy_next = formulation.getLiftedEnergy(x_next);
	while (energy_next > energy + step_size * gp) {
        // while (energy_next >= energy)
        step_size *= shrink;
        x_next = x + step_size * p;
        energy_next = formulation.getLiftedEnergy(x_next);
    }
	x = x_next;
	//
	if (record_stepSize) stepSizeRecord.push_back(step_size);
	//check ftol
	if (fabs(energy_next-energy) < ftol_abs) return;
	if (fabs((energy_next-energy)/energy) < ftol_rel) return;
	//check gtol
	if (grad.norm() < gtol_abs) return;



	for (int i = 1; i < maxIter; ++i)
	{
		formulation.getLiftedEnergyGradHessian(x,energy,grad,mat);

		// solver step monitor
		if (record_vert) vertRecord.push_back(formulation.V);
		if (record_energy) energyRecord.push_back(energy);
		if (record_gradient) gradientRecord.push_back(grad);
		if (record_minArea || stopCode == "all_good") {
			double minA = computeMinSignedArea(formulation.V, formulation.F);
			if (record_minArea) minAreaRecord.push_back(minA);
			if ((stopCode == "all_good") && (minA > 0.0)) return;
		}
		// solver step monitor end


		solver.factorize(mat);
		if(solver.info()!=Success) {
	  		cout <<  "iter " << i << ": decomposition failed" << endl;
	  		return;
		}
		VectorXd p = solver.solve(-grad);
		if(solver.info()!=Success) {
	  		cout << "iter " << i << ": solving failed" << endl;
	  		return;
		}
		if (record_searchDirection) searchDirectionRecord.push_back(p);

		// backtracking line search
		long double gp = 0.5 * grad.transpose() * p;
		long double step_size = 1.0;
		x_next = x + step_size * p;
		energy_next = formulation.getLiftedEnergy(x_next);
		while (energy_next > energy + step_size * gp) {
		// while (energy_next >= energy) {
			step_size *= shrink;
			x_next = x + step_size * p;
			energy_next = formulation.getLiftedEnergy(x_next);
		}
		//
		x = x_next;
		//
		if (record_stepSize) stepSizeRecord.push_back(step_size);
		//check ftol
		if (fabs(energy_next-energy) < ftol_abs) return;
		if (fabs((energy_next-energy)/energy) < ftol_rel) return;
		//check gtol
		if (grad.norm() < gtol_abs) return;
	}
}


//export result
bool exportResult(const char* filename, LiftedFormulation& formulation, const VectorXd& x, const SolverOptionManager& options)
{
	std::ofstream out_file(filename);
	if (! out_file.is_open()) {
		std::cerr << "Failed to open " << filename << "!" << std::endl;
		return false;
	}

	//precision of output
	typedef std::numeric_limits< double > dbl;
	out_file.precision(dbl::max_digits10);

	//update V
	formulation.update_V(x);
	const MatrixXd& V = formulation.V;

	//write the file
	unsigned nv = V.cols();
	unsigned ndim = V.rows();
	out_file << "resV " << nv << " " << ndim << "\n";
	for (auto i = 0; i < nv; ++i)
	{
		for (auto j = 0; j < ndim; ++j)
		{
			out_file << V(j,i) << " ";
		}
	}
	out_file << std::endl;


	if (options.record_vert)
	{
		const std::vector<MatrixXd>& vertRecord = options.vertRecord;
		unsigned n_record = vertRecord.size();
		out_file << "vert " << n_record << " " << nv << " " << ndim << std::endl;
		for (auto i = 0; i < n_record; ++i)
		{
			for (auto j = 0; j < nv; ++j)
			{
				for (auto k = 0; k < ndim; ++k)
				{
					out_file << vertRecord[i](k,j) << " ";
				}
			}
		}
		out_file << std::endl;
	}


	if (options.record_energy)
	{
		const std::vector<double>& energyRecord = options.energyRecord;
		unsigned n_record = energyRecord.size();
		out_file << "energy " << n_record << std::endl;
		for (auto i = 0; i < n_record; ++i)
		{
			out_file << energyRecord[i] << " ";
		}
		out_file << std::endl;
	}

	if (options.record_minArea)
	{
		const std::vector<double>& minAreaRecord = options.minAreaRecord;
		unsigned n_record = minAreaRecord.size();
		out_file << "minArea " << n_record << std::endl;
		for (auto i = 0; i < n_record; ++i)
		{
			out_file << minAreaRecord[i] << " ";
		}
		out_file << std::endl;
	}

	if (options.record_gradient)
	{
		const std::vector<VectorXd>& gradientRecord = options.gradientRecord;
		unsigned n_record = gradientRecord.size();
		unsigned n_free = formulation.freeI.size();
		out_file << "gradient " << n_record << " " << n_free << " " << ndim << std::endl;
		for (auto i = 0; i < n_record; ++i)
		{
			for (auto j = 0; j < n_free; ++j)
			{
				for (auto k = 0; k < ndim; ++k)
				{
					out_file << gradientRecord[i](j*ndim + k) << " ";
				}
			}
		}
		out_file << std::endl;
	}

	if (options.record_searchDirection)
	{
		const std::vector<VectorXd>& searchDirectionRecord = options.searchDirectionRecord;
		unsigned n_record = searchDirectionRecord.size();
		unsigned n_free = formulation.freeI.size();
		out_file << "searchDirection " << n_record << " " << n_free << " " << ndim << std::endl;
		for (auto i = 0; i < n_record; ++i)
		{
			for (auto j = 0; j < n_free; ++j)
			{
				for (auto k = 0; k < ndim; ++k)
				{
					out_file << searchDirectionRecord[i](j*ndim + k) << " ";
				}
			}
		}
		out_file << std::endl;
	}

	if (options.record_stepSize)
	{
		const std::vector<double>& stepSizeRecord = options.stepSizeRecord;
		unsigned n_record = stepSizeRecord.size();
		out_file << "stepSize " << n_record << std::endl;
		for (auto i = 0; i < n_record; ++i)
		{
			out_file << stepSizeRecord[i] << " ";
		}
		out_file << std::endl;
	}

	//
	out_file.close();
	return true;
}


int main(int argc, char const *argv[])
{
	const char* dataFile = (argc > 1) ? argv[1] : "./test/lifted";
	const char* optFile  = (argc > 2) ? argv[2] : "./test/lifted_solver_options";
	const char* resFile  = (argc > 3) ? argv[3] : "./test/lifted_res";

	//import data
	std::vector<std::vector<double> > raw_restV;
	std::vector<std::vector<double> > raw_initV;
	std::vector<std::vector<unsigned> > raw_F;
	std::vector<unsigned> raw_handles;
	std::string form;
	double alpha;

	importData(dataFile,raw_restV,raw_initV,raw_F,raw_handles,form,alpha);

	std::cout << "alpha: " << alpha << std::endl;

	//convert raw data to Eigen data
	const int nv = raw_restV.size();
	const int restDim = raw_restV[0].size();
	const int nf = raw_F.size();
	const int simplexSize = raw_F[1].size();
	const int initDim = raw_initV[0].size();

	MatrixXd restV(restDim, nv);
	for (int i = 0; i < nv; ++i)
	{
		for (int j = 0; j < restDim; ++j)
		{
			restV(j,i) = raw_restV[i][j];
		}
	}

	MatrixXd initV(initDim, nv);
	for (int i = 0; i < nv; ++i)
	{
		for (int j = 0; j < initDim; ++j)
		{
			initV(j,i) = raw_initV[i][j];
		}
	}

	MatrixXi F(simplexSize, nf);
	for (int i = 0; i < nf; ++i)
	{
		for (int j = 0; j < simplexSize; ++j)
		{
			F(j,i) = raw_F[i][j];
		}
	}

	VectorXi handles(raw_handles.size());
	for (auto i = 0; i < raw_handles.size(); ++i)
	{
		handles(i) = raw_handles[i];
	}

	//import options
	SolverOptionManager options(optFile);

	//
	LiftedFormulation myLifted(restV,initV,F,handles,form,alpha);
	VectorXd x = myLifted.x0;

    // dubug
    std::cout.precision(std::numeric_limits< double >::max_digits10);

	//projected newton
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	projected_Newton(myLifted,x,options);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [microseconds]" << std::endl;

	exportResult(resFile,myLifted,x,options);


	return 0;
}
