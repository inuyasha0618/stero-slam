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

    void EdgeProjXYZ2SteroUVPoseOnly::computeError() {
        const g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d P_c_l = pose->estimate().map(point_);
        Sophus::SE3 T_r_l = Sophus::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(-camera_->base_line_, 0, 0));
        Eigen::Vector3d P_c_r = T_r_l * P_c_l;

        Eigen::Vector2d p_p_l = camera_->camera2pixel(P_c_l);
        Eigen::Vector2d p_p_r = camera_->camera2pixel(P_c_r);

        // 这里的误差和重投影都是三维的
        Eigen::Vector3d obs(_measurement);
        _error = obs - Eigen::Vector3d(p_p_l(0), p_p_l(1), p_p_r(0));
    }

    void EdgeProjXYZ2SteroUVPoseOnly::linearizeOplus() {
        double fx = camera_->fx_;
        double fy = camera_->fy_;

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
        _jacobianOplusXi ( 0,3 ) = -1./z * fx;
        _jacobianOplusXi ( 0,4 ) = 0;
        _jacobianOplusXi ( 0,5 ) = x * inv_z_2 * fx;

        _jacobianOplusXi ( 1,0 ) = ( 1+y*y * inv_z_2 ) *fy;
        _jacobianOplusXi ( 1,1 ) = -x*y * inv_z_2 *fy;
        _jacobianOplusXi ( 1,2 ) = -x/z *fy;
        _jacobianOplusXi ( 1,3 ) = 0;
        _jacobianOplusXi ( 1,4 ) = -1./z *fy;
        _jacobianOplusXi ( 1,5 ) = y * inv_z_2 *fy;

        _jacobianOplusXi ( 2,0 ) =  (x - camera_->base_line_)*y * inv_z_2 *fx;
        _jacobianOplusXi ( 2,1 ) = - ( 1+ ( (x - camera_->base_line_) * x * inv_z_2 ) ) *fx;
        _jacobianOplusXi ( 2,2 ) = y/z * fx;
        _jacobianOplusXi ( 2,3 ) = -1./z * fx;
        _jacobianOplusXi ( 2,4 ) = 0;
        _jacobianOplusXi ( 2,5 ) = (x - camera_->base_line_) * inv_z_2 * fx;
    }

    Eigen::Vector3d EdgeStereoSE3ProjectXYZOnlyPose::cam_project(const Eigen::Vector3d &trans_xyz) const {
      const float invz = 1.0f / trans_xyz[2];
      Eigen::Vector3d res;
      res[0] = trans_xyz[0] * invz * fx + cx;
      res[1] = trans_xyz[1] * invz * fy + cy;
      res[2] = res[0] - bf * invz;
      return res;
    }

    void EdgeStereoSE3ProjectXYZOnlyPose::linearizeOplus() {
      g2o::VertexSE3Expmap *vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
      Eigen::Vector3d xyz_trans = vi->estimate().map(Xw);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

      _jacobianOplusXi(0, 0) = x * y * invz_2 * fx;
      _jacobianOplusXi(0, 1) = -(1 + (x * x * invz_2)) * fx;
      _jacobianOplusXi(0, 2) = y * invz * fx;
      _jacobianOplusXi(0, 3) = -invz * fx;
      _jacobianOplusXi(0, 4) = 0;
      _jacobianOplusXi(0, 5) = x * invz_2 * fx;

      _jacobianOplusXi(1, 0) = (1 + y * y * invz_2) * fy;
      _jacobianOplusXi(1, 1) = -x * y * invz_2 * fy;
      _jacobianOplusXi(1, 2) = -x * invz * fy;
      _jacobianOplusXi(1, 3) = 0;
      _jacobianOplusXi(1, 4) = -invz * fy;
      _jacobianOplusXi(1, 5) = y * invz_2 * fy;

      _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - bf * y * invz_2;
      _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) + bf * x * invz_2;
      _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2);
      _jacobianOplusXi(2, 3) = _jacobianOplusXi(0, 3);
      _jacobianOplusXi(2, 4) = 0;
      _jacobianOplusXi(2, 5) = _jacobianOplusXi(0, 5) - bf * invz_2;
    }





























}

