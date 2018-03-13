#include "myslam/g2o_types.h"

namespace myslam
{
    //***************************EdgeProjXYZ2UVPoseOnly相关*****************************************
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

    //*************************EdgeProjXYZ2UV相关******************************************

    void EdgeProjXYZ2UV::computeError() {
        const g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* point = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
        _error = _measurement - camera_->camera2pixel(pose->estimate().map(point->estimate()));
    }

    void EdgeProjXYZ2UV::linearizeOplus() {
        double fx = camera_->fx_;
        double fy = camera_->fy_;

        const g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* point = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);

        Eigen::Vector3d xyz_trans = pose->estimate().map(point->estimate());
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        double z_2 = z*z;

        _jacobianOplusXi ( 0,0 ) = -fx / z;
        _jacobianOplusXi ( 0,1 ) = 0;
        _jacobianOplusXi ( 0,2 ) = fx * x / z_2;

        _jacobianOplusXi ( 1,0 ) = 0;
        _jacobianOplusXi ( 1,1 ) = -fy / z;
        _jacobianOplusXi ( 1,2 ) = fy * y / z_2;

        _jacobianOplusXj ( 0,0 ) =  x*y/z_2 *fx;
        _jacobianOplusXj ( 0,1 ) = - ( 1+ ( x*x/z_2 ) ) *fx;
        _jacobianOplusXj ( 0,2 ) = y/z * fx;
        _jacobianOplusXj ( 0,3 ) = -1./z * fx;
        _jacobianOplusXj ( 0,4 ) = 0;
        _jacobianOplusXj ( 0,5 ) = x/z_2 * fx;

        _jacobianOplusXj ( 1,0 ) = ( 1+y*y/z_2 ) *fy;
        _jacobianOplusXj ( 1,1 ) = -x*y/z_2 *fy;
        _jacobianOplusXj ( 1,2 ) = -x/z *fy;
        _jacobianOplusXj ( 1,3 ) = 0;
        _jacobianOplusXj ( 1,4 ) = -1./z *fy;
        _jacobianOplusXj ( 1,5 ) = y/z_2 *fy;
    }

    //*************************************************************************************

    void EdgeProjXYZ2SteroUVPoseOnly::linearizeOplus() {

        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d xyz_trans = pose->estimate().map(point_);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        double z_2 = z*z;
        double inv_z_2 = 1.0/z_2;
        double bf = fx * bw;

        _jacobianOplusXi ( 0,0 ) =  x*y * inv_z_2 *fx;
        _jacobianOplusXi ( 0,1 ) = - ( 1+ ( x*x * inv_z_2 ) ) *fx;
        _jacobianOplusXi ( 0,2 ) = y/z * fx;
        _jacobianOplusXi ( 0,3 ) = -1./z * fx;
        _jacobianOplusXi ( 0,4 ) = 0;
        _jacobianOplusXi ( 0,5 ) = x * inv_z_2 * fx;

        _jacobianOplusXi ( 1,0 ) = ( 1+y*y * inv_z_2 ) *fy;
        _jacobianOplusXi ( 1,1 ) = -x*y * inv_z_2 *fy;
        _jacobianOplusXi ( 1,2 ) = -x/z *fy;
        _jacobianOplusXi ( 1,3 ) = 0;
        _jacobianOplusXi ( 1,4 ) = -1./z *fy;
        _jacobianOplusXi ( 1,5 ) = y * inv_z_2 *fy;

//        _jacobianOplusXi ( 2,0 ) =  (x - bw)*y * inv_z_2 *fx;
//        _jacobianOplusXi ( 2,1 ) = - ( 1+ ( (x - bw) * x * inv_z_2 ) ) *fx;
//        _jacobianOplusXi ( 2,2 ) = y/z * fx;
//        _jacobianOplusXi ( 2,3 ) = -1./z * fx;
//        _jacobianOplusXi ( 2,4 ) = 0;
//        _jacobianOplusXi ( 2,5 ) = (x - bw) * inv_z_2 * fx;

        _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - bf * y * inv_z_2;
        _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) + bf * x * inv_z_2;
        _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2);
        _jacobianOplusXi(2, 3) = _jacobianOplusXi(0, 3);
        _jacobianOplusXi(2, 4) = 0;
        _jacobianOplusXi(2, 5) = _jacobianOplusXi(0, 5) - bf * inv_z_2;
    }

    void EdgeProjXYZ2SteroUVRotOnly::computeError() {
        const g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d P_c_l = pose->estimate().map(point_);
        Sophus::SE3 T_r_l = Sophus::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(-bw, 0, 0));
        Eigen::Vector3d P_c_r = T_r_l * P_c_l;

        Eigen::Vector2d p_p_l = camera2pixel(P_c_l);
        Eigen::Vector2d p_p_r = camera2pixel(P_c_r);

        // 这里的误差和重投影都是三维的
        Eigen::Vector3d obs(_measurement);
        _error = obs - Eigen::Vector3d(p_p_l(0), p_p_l(1), p_p_r(0));
    }

    void EdgeProjXYZ2SteroUVRotOnly::linearizeOplus() {

        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d xyz_trans = pose->estimate().map(point_);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        double z_2 = z*z;
        double inv_z_2 = 1.0/z_2;

        _jacobianOplusXi ( 0,0 ) =  x*y * inv_z_2 *fx;
        _jacobianOplusXi ( 0,1 ) = - ( 1+ ( x*x * inv_z_2 ) ) *fx;
        _jacobianOplusXi ( 0,2 ) = y/z * fx;
        _jacobianOplusXi ( 0,3 ) = 0;
        _jacobianOplusXi ( 0,4 ) = 0;
        _jacobianOplusXi ( 0,5 ) = 0;

        _jacobianOplusXi ( 1,0 ) = ( 1+y*y * inv_z_2 ) *fy;
        _jacobianOplusXi ( 1,1 ) = -x*y * inv_z_2 *fy;
        _jacobianOplusXi ( 1,2 ) = -x/z *fy;
        _jacobianOplusXi ( 1,3 ) = 0;
        _jacobianOplusXi ( 1,4 ) = 0;
        _jacobianOplusXi ( 1,5 ) = 0;

        _jacobianOplusXi ( 2,0 ) =  (x - bw)*y * inv_z_2 *fx;
        _jacobianOplusXi ( 2,1 ) = - ( 1+ ( (x - bw) * x * inv_z_2 ) ) *fx;
        _jacobianOplusXi ( 2,2 ) = y/z * fx;
        _jacobianOplusXi ( 2,3 ) = 0;
        _jacobianOplusXi ( 2,4 ) = 0;
        _jacobianOplusXi ( 2,5 ) = 0;
    }

}

