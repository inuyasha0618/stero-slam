#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.h>
#include <sophus/so3.h>
#include <opencv2/core/core.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

#include <vector>
#include <list>
#include <memory>

#include <string>
#include <iostream>
#include <set>
#include <unordered_map>
#include <map>
using namespace std;
#endif