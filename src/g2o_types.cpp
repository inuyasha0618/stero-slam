#include "myslam/g2o_types.h"

namespace myslam
{
    void EdgeProjXYZ2UVPoseOnly::computeError() {
        const g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        _error = _measurement - camera_->camera2pixel(pose->estimate().map(point_));
    }

    void EdgeProjXYZ2UVPoseOnly::linearizeOplus() {
        double fx = camera_->fx_;
        double fy = camera_->fy_;

        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d xyz_trans = pose->estimate().map(point_);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        double z_2 = z*z;

        _jacobianOplusXi ( 0,0 ) =  x*y/z_2 *fx;
        _jacobianOplusXi ( 0,1 ) = - ( 1+ ( x*x/z_2 ) ) *fx;
        _jacobianOplusXi ( 0,2 ) = y/z * fx;
        _jacobianOplusXi ( 0,3 ) = -1./z * fx;
        _jacobianOplusXi ( 0,4 ) = 0;
        _jacobianOplusXi ( 0,5 ) = x/z_2 * fx;

        _jacobianOplusXi ( 1,0 ) = ( 1+y*y/z_2 ) *fy;
        _jacobianOplusXi ( 1,1 ) = -x*y/z_2 *fy;
        _jacobianOplusXi ( 1,2 ) = -x/z *fy;
        _jacobianOplusXi ( 1,3 ) = 0;
        _jacobianOplusXi ( 1,4 ) = -1./z *fy;
        _jacobianOplusXi ( 1,5 ) = y/z_2 *fy;
    }
}