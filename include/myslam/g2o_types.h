#include "myslam/common_include.h"
#include "myslam/camera.h"

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

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
        virtual void computeError();
        virtual void linearizeOplus();

        virtual bool read(std::istream& in) {}
        virtual bool write(std::ostream& os) const {}

        Eigen::Vector3d point_;
        Camera* camera_;
    };

    class EdgeStereoSE3ProjectXYZOnlyPose : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeStereoSE3ProjectXYZOnlyPose() {}

        virtual bool read(std::istream& in) {}
        virtual bool write(std::ostream& os) const {}

        void computeError() {
            const g2o::VertexSE3Expmap *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
            Eigen::Vector3d obs(_measurement);
            _error = obs - cam_project(v1->estimate().map(Xw));
        }

        bool isDepthPositive() {
            const g2o::VertexSE3Expmap *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
            return (v1->estimate().map(Xw))(2) > 0;
        }

        virtual void linearizeOplus();

        Eigen::Vector3d cam_project(const Eigen::Vector3d &trans_xyz) const;

        Eigen::Vector3d Xw;
        double fx, fy, cx, cy, bf;
    };

}