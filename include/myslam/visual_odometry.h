#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H


#include "myslam/common_include.h"
#include "myslam/map.h"
#include "myslam/feature.h"

#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
    class VisualOdometry
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef shared_ptr<VisualOdometry> Ptr;
        enum VOstate {
            INITIALIZING=-1,
            OK=0,
            LOST
        };

        VOstate state_;
        Map::Ptr map_;
        Frame::Ptr ref_;
        Frame::Ptr curr_;

        cv::Ptr<cv::ORB> orb_; // orb detector and computer
        vector<cv::Point3f> pts_3d_ref_;
        vector<cv::KeyPoint> keypoints_curr_, keypoints_curr_right_;
        cv::Mat descriptors_curr_;
        cv::Mat descriptors_ref_;
        vector<cv::DMatch> features_matches_;

        Sophus::SE3 T_c_r_esti_;
        int num_inliers_;
        int num_lost_;

        int num_of_features_;
        double scale_factor_;
        int level_pyramid_;
        float match_ratio_;
        int max_num_lost_;
        int min_inliers_;

        double key_frame_min_rot_;
        double key_frame_min_trans_;

        int FRAME_GRID_ROWS_;
        int FRAME_GRID_COLS_;
        int FRAME_CRID_SIZE_;

    public:
        VisualOdometry();
        ~VisualOdometry();

        bool addFrame(Frame::Ptr frame);

    protected:
        void addStereoMapPoints(shared_ptr<Frame> frame); //　通过左右目的视差，创建３维地图点，并且关联到左视图的features上
        void featureDetection(shared_ptr<Frame> frame); // 提取frame的左视图的features
        void stereoMatching(shared_ptr<Frame> frame);    // 左右目匹配，同时找出视差
        void featrureTracking();
        void poseEstimationPnP();
        void setRef3DPoints();

        void addKeyFrame();
        bool checkEstimatedPose();
        bool checkKeyFrame();

    private:
        // Shi-Tomasi 分数，这个分数越高则特征越优先
        inline float ShiTomasiScore(const cv::Mat &img, const int &u, const int &v) const {
            float dXX = 0.0;
            float dYY = 0.0;
            float dXY = 0.0;
            const int halfbox_size = 4;
            const int box_size = 2 * halfbox_size;
            const int box_area = box_size * box_size;
            const int x_min = u - halfbox_size;
            const int x_max = u + halfbox_size;
            const int y_min = v - halfbox_size;
            const int y_max = v + halfbox_size;

            if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1)
                return 0.0; // patch is too close to the boundary

            const int stride = img.step.p[0];
            for (int y = y_min; y < y_max; ++y) {
                const uint8_t *ptr_left = img.data + stride * y + x_min - 1;
                const uint8_t *ptr_right = img.data + stride * y + x_min + 1;
                const uint8_t *ptr_top = img.data + stride * (y - 1) + x_min;
                const uint8_t *ptr_bottom = img.data + stride * (y + 1) + x_min;
                for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
                    float dx = *ptr_right - *ptr_left;
                    float dy = *ptr_bottom - *ptr_top;
                    dXX += dx * dx;
                    dYY += dy * dy;
                    dXY += dx * dy;
                }
            }

            // Find and return smaller eigenvalue:
            dXX = dXX / (2.0 * box_area);
            dYY = dYY / (2.0 * box_area);
            dXY = dXY / (2.0 * box_area);
            return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
        }
    };
}

#endif