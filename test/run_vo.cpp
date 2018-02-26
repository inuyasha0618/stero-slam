// -------------- test the visual odometry -------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

#define MAX_FRAME 1000

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    myslam::Config::setParamFile(argv[1]);
    myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );

    string dataset_dir = myslam::Config::getParam<string> ( "dataset_dir" );

    myslam::Camera::Ptr camera ( new myslam::Camera );

    cv::namedWindow( "Road facing camera", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::namedWindow( "Trajectory", cv::WINDOW_AUTOSIZE );// Create a window for display.

    char filename_l[100];
    char filename_r[100];

    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);

    for ( int i=0; i<MAX_FRAME; i++ )
    {
        sprintf(filename_l, "/home/slam/datasets/kitti/00/image_0/%06d.png", i);
        sprintf(filename_r, "/home/slam/datasets/kitti/00/image_1/%06d.png", i);

        cout << filename_l << endl;

        cv::Mat img_left = cv::imread(filename_l);
        cv::Mat img_right = cv::imread (filename_r);
        if ( img_left.data==nullptr || img_right.data==nullptr ) {
            cout << "no img data" << endl;
            break;
        }
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->img_left_ = img_left;
        pFrame->img_right_ = img_right;

        boost::timer timer;
        vo->addFrame ( pFrame );
        cout<<"VO costs time: "<<timer.elapsed()<<endl;

        if ( vo->state_ == myslam::VisualOdometry::LOST )
            break;
        Sophus::SE3 Twc = pFrame->T_c_w_.inverse();

        int x = int(Twc.translation()(0)) + 300;
        int y = int(Twc.translation()(2)) + 100;

        if (i == 1) {
            cout << "第2帧x: " << x << " y: " << y << endl;
            cv::waitKey(0);
        }

        cout << "x: " << x << "y: " << y << endl;
        cv::circle(traj, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);

        cv::rectangle( traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0,0,0), CV_FILLED);

        imshow( "Road facing camera", img_left );
        imshow( "Trajectory", traj );

        cv::waitKey(1);
    }

    return 0;
}
