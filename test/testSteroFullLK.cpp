// -------------- test the visual odometry -------------
#include <fstream>
#include <sstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/video/tracking.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int main ( int argc, char** argv )
{
    clock_t start = clock();
    myslam::Config::setParamFile(argv[1]);
    string dataset_dir = myslam::Config::getParam<string> ( "dataset_dir" );

    string leftImgPath = dataset_dir + "image_0/000000.png";
    string rightImgPath = dataset_dir + "image_1/000000.png";

    cv::Mat leftImg = cv::imread(leftImgPath, 0);
    cv::Mat rightImg = cv::imread(rightImgPath, 0);

    vector<cv::KeyPoint> leftKps;

    cv::FAST(leftImg, leftKps, 20, true);

    vector<cv::Point2f> left_points;
    vector<cv::Point2f> right_points;
    vector<uchar> status;
    vector<float> err;

    cv::KeyPoint::convert(leftKps, left_points);

    cv::calcOpticalFlowPyrLK(leftImg, rightImg, left_points, right_points, status, err, cv::Size(11, 11));

    clock_t end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    cout << "Total time taken: " << elapsed_secs << "s" << endl << " total: " << left_points.size() << " points" << endl;

    for (int i = 0; i < right_points.size(); i++) {
        if (status[i] && fabs(left_points[i].y - right_points[i].y) <= 3) {
            cv::circle(leftImg, left_points[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::circle(rightImg, right_points[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(rightImg, left_points[i], right_points[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("key points in left image", leftImg);
    cv::imshow("tracked in right image", rightImg);

    cv::waitKey(0);

    return 0;
}
