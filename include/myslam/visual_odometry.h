#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H


#include "myslam/common_include.h"
#include "myslam/map.h"

#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
    class VisualOdometry
    {
    public:
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

    public:
        VisualOdometry();
        ~VisualOdometry();

        bool addFrame(Frame::Ptr frame);

    protected:
        void extractKeyPointsAndComputeDescriptors();
        void featrureMatching();
        void poseEstimationPnP();
        void setRef3DPoints();

        void addKeyFrame();
        bool checkEstimatedPose();
        bool checkKeyFrame();
    };
}

#endif