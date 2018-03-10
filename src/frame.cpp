#include "myslam/frame.h"
namespace myslam
{
    Frame::Frame(): id_(-1), time_stamp_(-1), camera_(nullptr) {}

    Frame::Frame(unsigned long id, double time_stamp, Sophus::SE3 T_c_w, Camera::Ptr camera, cv::Mat img_left, cv::Mat img_right):
    id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), img_left_(img_left), img_right_(img_right) {
        frame_grid_.resize(FRAM_GRID_ROWS * FRAM_GRID_COLS);
    }

    Frame::~Frame() {}

    Frame::Ptr Frame::createFrame() {
        static unsigned long id = 0;
        return Frame::Ptr(new Frame(id++));
    }

    double Frame::findDepth(const cv::KeyPoint& kp_l, const cv::KeyPoint& kp_r) {

        double disparity = kp_l.pt.x - kp_r.pt.x;

        if (disparity > 0) {

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

    bool Frame::posInGrid(const shared_ptr<Feature> feature, int &posX, int &posY) {
        posX = int(feature->pixel_[0] / FRAM_GRID_SIZE);
        posY = int(feature->pixel_[1] / FRAM_GRID_SIZE);

        if(posX < 0 || posY < 0 || posX >= FRAM_GRID_COLS || posY >= FRAM_GRID_ROWS) {
            return false;
        }

        return true;
    }

    void Frame::assignFeaturesToGrid() {
        if (leftFeatures_.empty())
            return;

        for (auto grid: frame_grid_) {
            grid.clear();
        }

        for (size_t i = 0; i < leftFeatures_.size(); i++) {
            int posX, posY;
            shared_ptr<Feature> feature = leftFeatures_[i];
            if (feature == nullptr)
                continue;
            if (posInGrid(feature, posX, posY)) {
                frame_grid_[posY * FRAM_GRID_COLS + posX].push_back(i);
            }
        }
    }
}