#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/aruco/charuco.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <time.h>

using namespace cv;
using namespace std;

static void detectCharucoCorners(const vector<string> &list, vector<string> &list_detected, vector<Mat> &imagePoints, vector<Mat> &objectPoints, const Ptr<aruco::Dictionary> &dictionary, const Ptr<aruco::CharucoBoard> &board, Size &imgSize)
{
    imagePoints.resize(0);
    list_detected.resize(0);
    int n_img = (int)list.size();
    Mat img;
    Mat debugImg;
    Mat ids;
    for (int i = 0; i < n_img; ++i)
    {
        cout << list[i] << "... " << endl;
        vector<Mat> points, rejectedPoints;
        Mat charucoCorners, charucoIds;
        Mat imgPoints, objPoints;
        debugImg = imread(list[i], IMREAD_COLOR);
        cvtColor(debugImg, img, COLOR_BGR2GRAY);
        aruco::detectMarkers(img, dictionary, points, ids);
        if (ids.total() > 0)
        {
            list_detected.push_back(list[i]);
            aruco::refineDetectedMarkers(img, board, points, ids, rejectedPoints);
            aruco::interpolateCornersCharuco(points, ids, img, board, charucoCorners, charucoIds);

            if (charucoIds.total() > 0)
            {
                board->matchImagePoints(charucoCorners, charucoIds, objPoints, imgPoints);
                for (const auto& charucoCorner : cv::Mat_<Point2f>(charucoCorners))
                {
                    cv::circle(debugImg, charucoCorner, 3, Scalar(255, 0, 0), 2);
                }
                imagePoints.push_back(imgPoints);
                objectPoints.push_back(objPoints);
            }
        }

        // Draw for fun.
        imshow("Window", debugImg);
        waitKey(50);

        imgSize = img.size();
    }
}

static bool readStringList(const string &filename, vector<string> &l)
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
        l.push_back((string)*it);
    return true;
}

static void saveCameraParams(const string &filename, int flags, const Mat &cameraMatrix,
                             const Mat &distCoeffs, const double xi, const vector<Vec3d> &rvecs, const vector<Vec3d> &tvecs,
                             vector<string> detec_list, const Mat &idx, const double rms, const vector<Mat> &imagePoints)
{
    cout << filename << endl;
    FileStorage fs(filename, FileStorage::WRITE);

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    if (!rvecs.empty())
        fs << "nFrames" << (int)rvecs.size();

    if (flags != 0)
    {
        sprintf(buf, "flags: %s%s%s%s%s%s%s%s%s",
                flags & omnidir::CALIB_USE_GUESS ? "+use_intrinsic_guess" : "",
                flags & omnidir::CALIB_FIX_SKEW ? "+fix_skew" : "",
                flags & omnidir::CALIB_FIX_K1 ? "+fix_k1" : "",
                flags & omnidir::CALIB_FIX_K2 ? "+fix_k2" : "",
                flags & omnidir::CALIB_FIX_P1 ? "+fix_p1" : "",
                flags & omnidir::CALIB_FIX_P2 ? "+fix_p2" : "",
                flags & omnidir::CALIB_FIX_XI ? "+fix_xi" : "",
                flags & omnidir::CALIB_FIX_GAMMA ? "+fix_gamma" : "",
                flags & omnidir::CALIB_FIX_CENTER ? "+fix_center" : "");
        // cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "xi" << xi;

    // cvWriteComment( *fs, "names of images that are acturally used in calibration", 0 );
    fs << "used_imgs" << "[";
    for (int i = 0; i < (int)idx.total(); ++i)
    {
        fs << detec_list[(int)idx.at<int>(i)];
    }
    fs << "]";

    if (!rvecs.empty() && !tvecs.empty())
    {
        Mat rvec_tvec((int)rvecs.size(), 6, CV_64F);
        for (int i = 0; i < (int)rvecs.size(); ++i)
        {
            Mat(rvecs[i]).reshape(1, 1).copyTo(rvec_tvec(Rect(0, i, 3, 1)));
            Mat(tvecs[i]).reshape(1, 1).copyTo(rvec_tvec(Rect(3, i, 3, 1)));
        }
        // cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << rvec_tvec;
    }

    fs << "rms" << rms;

    if (!imagePoints.empty())
    {
        Mat imageMat((int)imagePoints.size(), (int)imagePoints[0].total(), CV_64FC2);
        for (int i = 0; i < (int)imagePoints.size(); ++i)
        {
            Mat r = imageMat.row(i).reshape(2, imageMat.cols);
            Mat imagei(imagePoints[i]);
            imagei.copyTo(r);
        }
        fs << "image_points" << imageMat;
    }
}

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv,
                                 "{w||board width}"
                                 "{h||board height}"
                                 "{sw|1.0|square width}"
                                 "{sh|1.0|square height}"
                                 "{tw|1.0|tag width}"
                                 "{th|1.0|tag height}"
                                 "{o|out_camera_params.xml|output file}"
                                 "{fs|false|fix skew}"
                                 "{fp|false|fix principal point at the center}"
                                 "{@input||input file - xml file with a list of the images, created with cpp-example-imagelist_creator tool}"
                                 "{help||show help}");
    parser.about("This is a sample for omnidirectional camera calibration. Example command line:\n"
                 "    omni_calibration_charuco -w=6 -h=9 -sw=80 -sh=80 imagelist.xml \n");
    if (parser.has("help") || !parser.has("w") || !parser.has("h"))
    {
        parser.printMessage();
        return 0;
    }

    Size boardSize(parser.get<int>("w"), parser.get<int>("h"));
    Size2d squareSize(parser.get<double>("sw"), parser.get<double>("sh"));
    Size2d tagSize(parser.get<double>("tw"), parser.get<double>("th"));
    int flags = 0;
    if (parser.get<bool>("fs"))
        flags |= omnidir::CALIB_FIX_SKEW;
    if (parser.get<bool>("fp"))
        flags |= omnidir::CALIB_FIX_CENTER;
    const string outputFilename = parser.get<string>("o");
    const string inputFilename = parser.get<string>(0);

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    // get image name list
    vector<string> image_list, detec_list;
    if (!readStringList(inputFilename, image_list))
    {
        cout << "Can not read imagelist" << endl;
        return -1;
    }

    // aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_100);
    aruco::CharucoBoard board = aruco::CharucoBoard(boardSize, squareSize.width, tagSize.width, dictionary);
    Ptr<aruco::Dictionary> ptrDictionary = makePtr<aruco::Dictionary>(dictionary);
    Ptr<aruco::CharucoBoard> ptrBoard = makePtr<aruco::CharucoBoard>(board);

    // find corners in images
    vector<Mat> imagePoints;
    vector<Mat> objectPoints;
    Size imageSize;
    if (!tagSize.empty())
        detectCharucoCorners(image_list, detec_list, imagePoints, objectPoints, ptrDictionary, ptrBoard, imageSize);

    // run calibration, some images are discarded in calibration process because they are failed
    // in initialization. Retained image indexes are in idx variable.
    Mat K, D, xi, idx;
    vector<Vec3d> rvecs, tvecs;
    double _xi, rms;
    TermCriteria criteria(3, 2000, 1e-8);
    rms = omnidir::calibrate(objectPoints, imagePoints, imageSize, K, xi, D, rvecs, tvecs, flags, criteria, idx);
    cout << rms << endl;
    cout << K << " " << D << endl;
    _xi = xi.at<double>(0);
    cout << "Saving camera params to " << outputFilename << endl;
    saveCameraParams(outputFilename, flags, K, D, _xi,
                     rvecs, tvecs, detec_list, idx, rms, imagePoints);

    // Add rvecs/tvecs visualizations


    // Undistortion
    Mat R, R_special, P, map1, map2;
    R = Mat::eye(3, 3, CV_64F);
    // R_special = Mat::zeros(3, 3, CV_64F);
    // P = Mat::eye(3, 3, CV_64F);
    // double fx = K.at<double>(0, 0) / 2;
    // double fy = K.at<double>(1, 1) / 2;
    // double cx = imageSize.width / 2;
    // double cy = imageSize.height / 2;
    // R.at<double>(0, 0) = pow(2, 0.5) / 2;
    // R.at<double>(2, 2) = pow(2, 0.5) / 2;
    // R.at<double>(0, 2) = - pow(2, 0.5) / 2;
    // R.at<double>(2, 0) = pow(2, 0.5) / 2;
    // P.at<double>(0, 0) = fx;
    // P.at<double>(1, 1) = fy;
    // P.at<double>(0, 2) = cx;
    // P.at<double>(1, 2) = cy;

    Mat img, undistortedImg;
    img = imread("/home/jnader/Images/fisheye_images/images/cam_0_0.png", IMREAD_GRAYSCALE);

    omnidir::initUndistortRectifyMap(K, D, xi, R, P, Size(imageSize.width * 2, imageSize.height * 2), CV_32FC1, map1, map2, omnidir::RECTIFY_LONGLATI);
    remap(img, undistortedImg, map1, map2, INTER_AREA);
    imwrite("/tmp/img_LONGILATI.png", undistortedImg);
    // R_special.at<double>(1, 0) = 1;
    // R_special.at<double>(2, 0) = 1;
    // R_special.at<double>(0, 2) = 1;

    omnidir::initUndistortRectifyMap(K, D, xi, R, P, Size(imageSize.width * 2, imageSize.height * 2), CV_32FC1, map1, map2, omnidir::RECTIFY_CYLINDRICAL);
    remap(img, undistortedImg, map1, map2, INTER_AREA);
    imwrite("/tmp/img_CYLINDRICAL.png", undistortedImg);

    omnidir::initUndistortRectifyMap(K, D, xi, R, P, Size(imageSize.width * 2, imageSize.height * 2), CV_32FC1, map1, map2, omnidir::RECTIFY_PERSPECTIVE);
    remap(img, undistortedImg, map1, map2, INTER_AREA);
    imwrite("/tmp/img_PERSPECTIVE.png", undistortedImg);

    omnidir::initUndistortRectifyMap(K, D, xi, R, P, Size(imageSize.width * 2, imageSize.height * 2), CV_32FC1, map1, map2, omnidir::RECTIFY_STEREOGRAPHIC);
    remap(img, undistortedImg, map1, map2, INTER_AREA);
    imwrite("/tmp/img_STEREOGRAPHIC.png", undistortedImg);
}
