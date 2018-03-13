#ifndef MAP_POINT_H
#define MAP_POINT_H

#include "common_include.h"

namespace myslam
{
    class Frame;

    // Todo 由于mappoint被前后端所共享，所以应该加些锁
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

        // 维护了一个map, key是观测到该点的帧，以及在该帧当中是第几个feature
        typedef std::map<weak_ptr<Frame>, size_t, std::owner_less<weak_ptr<Frame>>> ObsMap;
        ObsMap observations_;

        void addObservation(shared_ptr<Frame> keyFrame, size_t index);

        Eigen::Vector3d getWorldPos() {
            // Todo 看看是否需要加锁
            return pos_;
        }
    };
}

#endif