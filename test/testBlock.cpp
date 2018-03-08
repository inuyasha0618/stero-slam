#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

int main(int argc, char** argv) {

    std::string leftImgPath = "/home/slam/datasets/kitti/00/image_0/000000.png";
    cv::Mat originalImg = cv::imread(leftImgPath, 0);

    cv::Mat Block = originalImg(cv::Rect(300, 100, 50, 50));

    cv::circle(Block, cv::Point(0, 0), 10, CV_RGB(0, 255, 0), 5);

    cv::imshow("original img", originalImg);
    cv::imshow("block img: ", Block);
    cv::waitKey(0);

    return 0;
}