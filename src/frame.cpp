#include "myslam/frame.h"
namespace myslam
{
    Frame::Frame(): id_(-1), time_stamp_(-1), camera_(nullptr) {}

    Frame::Frame(unsigned long id, double time_stamp, Sophus::SE3 T_c_w, Camera::Ptr camera, cv::Mat img_left, cv::Mat img_right):
    id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), img_left_(img_left), img_right_(img_right) {}

    Frame::~Frame() {}

    Frame::Ptr Frame::createFrame() {
        static unsigned long id = 0;
        return Frame::Ptr(new Frame(id++));
    }

    double Frame::findDepth(const cv::KeyPoint& kp_l, const cv::KeyPoint& kp_r) {
        int u_l = cvRound(kp_l.pt.x);
        int u_r = cvRound(kp_r.pt.x);

        int disparity = u_l - u_r;

        if (disparity !=0) {
            return camera_->fx_ * camera_->base_line_ / disparity;
        } else {
            return -1.0;
        }

    }

    Eigen::Vector3d Frame::getCameraCenter() const {
        return T_c_w_.inverse().translation();
    }

    bool Frame::isInFrame(const Eigen::Vector3d &p_w) {
        Eigen::Vector3d p_cam = camera_->world2camera(p_w, T_c_w_);
        if (p_cam(2) < 0) {
            return false;
        }
        Eigen::Vector2d p_pix = camera_->world2pixel(p_w, T_c_w_);

        return  p_pix(0) > 0 && p_pix(1) > 0 && p_pix(0) < img_left_.cols && p_pix(1) < img_left_.rows;
    }
}