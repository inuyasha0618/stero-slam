#include "myslam/common_include.h"
#include "myslam/camera.h"

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

//#include <g2o/core/block_solver.h>
//#include <g2o/core/optimization_algorithm_levenberg.h>
//#include <g2o/solvers/eigen/linear_solver_eigen.h>
//#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
//#include <g2o/types/sba/types_six_dof_expmap.h>
//#include <g2o/core/robust_kernel_impl.h>
//#include <g2o/solvers/dense/linear_solver_dense.h>
//#include <g2o/types/sim3/types_seven_dof_expmap.h>
//#include <g2o/types/slam3d/vertex_se3.h>
//#include <g2o/types/slam3d/edge_se3.h>

namespace myslam
{
    // 模板参数１：测量值维度 2: 测量值数据类型 3: 连接的定点类型
    class EdgeProjXYZ2UVPoseOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        virtual void computeError();
        virtual void linearizeOplus();

        virtual bool read(std::istream& in) {}
        virtual bool write(std::ostream& os) const {}

        Eigen::Vector3d point_;
        Camera* camera_;
    };

    class EdgeProjXYZ2SteroUVPoseOnly: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        virtual void computeError() {
            const g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
            Eigen::Vector3d P_c_l = pose->estimate().map(point_);

            Eigen::Vector2d p_p_l = camera2pixel(P_c_l);

            // 这里的误差和重投影都是三维的
            Eigen::Vector3d obs(_measurement);
            _error = obs - Eigen::Vector3d(p_p_l(0), p_p_l(1), p_p_l(0) - fx * bw / P_c_l(2));

        }
        virtual void linearizeOplus();

        virtual bool read(std::istream& in) {}
        virtual bool write(std::ostream& os) const {}

        Eigen::Vector3d point_;
//        Camera* camera_;

        Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c) const{
            Eigen::Vector2d p_p;
            p_p(0) = fx * p_c(0) / p_c(2) + cx;
            p_p(1) = fy * p_c(1) / p_c(2) + cy;

            return p_p;
        }

//        Eigen::Vector2d EdgeSE3ProjectXYZOnlyPose::cam_project(const Eigen::Vector3d & trans_xyz) const{
//            Eigen::Vector2d proj;
//            proj(0) = trans_xyz(0)/trans_xyz(2);
//            proj(1) = trans_xyz(1)/trans_xyz(2);
//
//            Eigen::Vector2d res;
//            res[0] = proj[0]*fx + cx;
//            res[1] = proj[1]*fy + cy;
//            return res;
//        }

        double fx, fy, cx, cy, bw;
    };

    class EdgeProjXYZ2SteroUVRotOnly: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        virtual void computeError();
        virtual void linearizeOplus();

        virtual bool read(std::istream& in) {}
        virtual bool write(std::ostream& os) const {}

        Eigen::Vector3d point_;

        Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c) const{
            Eigen::Vector2d p_p;
            p_p(0) = fx * p_c(0) / p_c(2) + cx;
            p_p(1) = fy * p_c(1) / p_c(2) + cy;

            return p_p;
        }

        double fx, fy, cx, cy, bw;
    };

}