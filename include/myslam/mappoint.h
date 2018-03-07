#ifndef MAP_POINT_H
#define MAP_POINT_H

#include "common_include.h"

namespace myslam
{
    class Frame;

    class MapPoint
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_;
        Eigen::Vector3d pos_;
        Eigen::Vector3d norm_;
        cv::Mat descriptor_;
        int observed_times_;
        int correct_times_;

        MapPoint();
        MapPoint(unsigned long id, Eigen::Vector3d pos, Eigen::Vector3d norm);

        static MapPoint::Ptr createMapPoint();

        weak_ptr<Frame> refKF_; // 第一次观测到此地图点的关键帧
    };
}

#endif