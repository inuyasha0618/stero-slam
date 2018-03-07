// -------------- test the visual odometry -------------
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

void featureDetection(Mat img_1)	{   //uses FAST as of now, modify parameters as necessary
    vector<KeyPoint> keypoints_1, kps_orb;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    clock_t fast_begin = clock();
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
    clock_t fast_end = clock();
    cout << "fast take time: " << double(fast_end - fast_begin) / CLOCKS_PER_SEC << endl << " " << keypoints_1.size() << " points" << endl;
//    cv::Mat output;
//    cv::drawKeypoints(img_1,keypoints_1,output);
//    cv::imshow("Fast corners: ", output);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    clock_t orb_begin = clock();
    orb->detect(img_1,kps_orb);
    clock_t orb_end = clock();
    cout << "orb take time: " << double(orb_end - orb_begin) / CLOCKS_PER_SEC << endl << " " << kps_orb.size() << " points" << endl;
    cv::waitKey(0);
}

int main ( int argc, char** argv )
{
    clock_t start = clock();
//    myslam::Config::setParamFile(argv[1]);
//    string dataset_dir = myslam::Config::getParam<string> ( "dataset_dir" );

    string leftImgPath = "/home/slam/datasets/kitti/00/image_0/000000.png";

    cv::Mat leftImg = cv::imread(leftImgPath, 0);

//    cv::imshow("before", leftImg);
//    cv::waitKey(0);

    featureDetection(leftImg);

//    vector<cv::KeyPoint> kps;
//    cv::FAST(leftImg, kps, 20, true);
//    cv::Mat outputImg;
//    cv::drawKeypoints(leftImg, kps, outputImg);
//    cv::imshow("Fast corners: ", outputImg);
//    cv::waitKey(0);


    return 0;
}
