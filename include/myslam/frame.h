#ifndef FRAME_H
#define FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"
#include "myslam/feature.h"
#include "myslam/settings.h"

namespace myslam
{
    class Feature;
    // Todo 由于Frame被前后端所共享，所以应该加些锁
    class Frame
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr;
        unsigned long id_; // 帧的id
        string imgPath;
        double time_stamp_;
        Sophus::SE3 T_c_w_;
        Camera::Ptr camera_;
        cv::Mat img_left_, img_right_;
        vector<shared_ptr<Feature>> leftFeatures_;                // 左视图的features
        vector<vector<size_t>> frame_grid_;         // 把grid拉平，　每个格子里存放一个vector，里面是在这个格子里的feature的序号

    public:
        Frame();

        Frame(unsigned long id, double time_stamp = 0, Sophus::SE3 T_c_w = Sophus::SE3(), Camera::Ptr camera = nullptr, cv::Mat img_left = cv::Mat(), cv::Mat img_right = cv::Mat());

        ~Frame();

        static Frame::Ptr createFrame();

        double findDepth(const cv::KeyPoint& kp_l, const cv::KeyPoint& kp_r);

        Eigen::Vector3d getCameraCenter() const;

        bool isInFrame(const Eigen::Vector3d& p_w);

        bool posInGrid(const shared_ptr<Feature> feature, int &posX, int &posY);

        void assignFeaturesToGrid();
    };
}

#endif