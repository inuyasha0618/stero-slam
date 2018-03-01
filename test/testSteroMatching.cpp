// -------------- test the visual odometry -------------
#include <fstream>
#include <sstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int main ( int argc, char** argv )
{
    clock_t start = clock();
//    stringstream ss;
    myslam::Config::setParamFile(argv[1]);
    string dataset_dir = myslam::Config::getParam<string> ( "dataset_dir" );
    int num_of_features = myslam::Config::getParam<int> ( "number_of_features" );
    double scale_factor = myslam::Config::getParam<double> ( "scale_factor" );
    int level_pyramid = myslam::Config::getParam<int> ( "level_pyramid" );

    string leftImgPath = dataset_dir + "image_0/000000.png";
    string rightImgPath = dataset_dir + "image_1/000000.png";

    cv::Mat leftImg = cv::imread(leftImgPath);
    cv::Mat rightImg = cv::imread(rightImgPath);

    cv::Ptr<cv::ORB> orb = cv::ORB::create(num_of_features, scale_factor, level_pyramid);
    vector<cv::KeyPoint> leftKps;
    vector<cv::KeyPoint> rightKps;

    orb->detect(leftImg, leftKps);
    orb->detect(rightImg, rightKps);

    cv::Mat desLeft, desRight;
    cv::Mat matchImg;

    orb->compute(leftImg, leftKps, desLeft);
    orb->compute(rightImg, rightKps, desRight);

    cv::HammingLUT lut;

    vector<cv::DMatch> matches;
    int size = desLeft.cols;

    for (int i = 0; i < desLeft.rows; i++) {
        cv::KeyPoint left_kp_i = leftKps.at(i);
        float left_kp_i_y = left_kp_i.pt.y;
        float left_kp_i_x = left_kp_i.pt.x;

        int trainId = -1;
        int bestDistance = 99999999;

        for (int j = 0; j < desRight.rows; j++) {
            cv::KeyPoint right_kp_j = rightKps.at(j);
            float right_kp_j_y = right_kp_j.pt.y;
            float right_kp_j_x = right_kp_j.pt.x;

//            if (abs(left_kp_i_y - right_kp_j_y) > 2) continue;
            if (abs(left_kp_i_y - right_kp_j_y) > 2 || right_kp_j_x > left_kp_i_x) continue;

            int distance = lut(desLeft.ptr<uchar>(i), desRight.ptr<uchar>(j), size);
            if (distance < bestDistance) {
                trainId = j;
                bestDistance = distance;
            }
        }

        if (trainId != -1) {
            matches.push_back(cv::DMatch(i, trainId, bestDistance));
        }
    }

    cv::Mat desLeft_filtered;
    vector<cv::KeyPoint> leftKps_filtered;
    for (cv::DMatch& m: matches) {
        desLeft_filtered.push_back(desLeft.row(m.queryIdx));
        leftKps_filtered.push_back(leftKps.at(m.queryIdx));
    }

//    cv::BFMatcher matcher(cv::NORM_HAMMING);
//    vector<cv::DMatch> matches;
//
//    matcher.match(desLeft, desRight, matches);
//
    clock_t end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    cout << "Total time taken: " << elapsed_secs << "s" << endl;

    cv::drawMatches(leftImg, leftKps, rightImg, rightKps, matches, matchImg);

    cv::imshow("matches", matchImg);
    cv::waitKey(0);

    // 下面用这个自己算出的matches与第二张图的左视图进行匹配，看能否匹配成功
    string leftImgPath_2 = dataset_dir + "image_0/000001.png";
    cv::Mat leftImg2 = cv::imread(leftImgPath_2);
    vector<cv::KeyPoint> leftKps2;
    cv::Mat desLeft2;
    orb->detect(leftImg2, leftKps2);
    orb->compute(leftImg2, leftKps2, desLeft2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<cv::DMatch> matches1_2;

    matcher.match(desLeft_filtered, desLeft2, matches1_2);
    cv::Mat matches2_img;
    cv::drawMatches(leftImg, leftKps_filtered, leftImg2, leftKps2, matches1_2, matches2_img);
    cv::imshow("1 to 2", matches2_img);
    cv::waitKey(0);

    return 0;
}
