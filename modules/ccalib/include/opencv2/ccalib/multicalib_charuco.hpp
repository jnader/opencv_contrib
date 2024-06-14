/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Baisheng Lai (laibaisheng@gmail.com), Zhejiang University,
// all rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_MULTICAMERACALIBRATIONCHARUCO_HPP__
#define __OPENCV_MULTICAMERACALIBRATIONCHARUCO_HPP__

#include "opencv2/ccalib/randpattern.hpp"
#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/aruco/charuco.hpp"
#include <string>
#include <iostream>

namespace cv { namespace multicalib {

//! @addtogroup ccalib
//! @{

#define HEAD -1
#define INVALID -2

/** @brief Class for multiple camera calibration that supports pinhole camera and omnidirection camera.
For omnidirectional camera model, please refer to omnidir.hpp in ccalib module.
It first calibrate each camera individually, then a bundle adjustment like optimization is applied to
refine extrinsic parameters. So far, it only support "random" pattern for calibration,
see randomPattern.hpp in ccalib module for details.
Images that are used should be named by "cameraIdx-timestamp.*", several images with the same timestamp
means that they are the same pattern that are photographed. cameraIdx should start from 0.

For more details, please refer to paper
    B. Li, L. Heng, K. Kevin  and M. Pollefeys, "A Multiple-Camera System
    Calibration Toolbox Using A Feature Descriptor-Based Calibration
    Pattern", in IROS 2013.
*/

class CV_EXPORTS MultiCameraCalibrationCharuco
{
public:
    enum {
        PINHOLE,
        OMNIDIRECTIONAL
        //FISHEYE
    };

    // an edge connects a camera and pattern
    struct edge
    {
        int cameraVertex;   // vertex index for camera in this edge
        int photoVertex;    // vertex index for pattern in this edge
        int photoIndex;     // photo index among photos for this camera
        Mat transform;      // transform from pattern to camera

        edge(int cv, int pv, int pi, Mat trans)
        {
            cameraVertex = cv;
            photoVertex = pv;
            photoIndex = pi;
            transform = trans;
        }
    };

    struct vertex
    {
        Mat pose;   // relative pose to the first camera. For camera vertex, it is the
                    // transform from the first camera to this camera, for pattern vertex,
                    // it is the transform from pattern to the first camera
        int timestamp;  // timestamp of photo, only available for photo vertex

        vertex(Mat po, int ts)
        {
            pose = po;
            timestamp = ts;
        }

        vertex()
        {
            pose = Mat::eye(4, 4, CV_32F);
            timestamp = -1;
        }
    };
    /* @brief Constructor
    @param cameraType camera type, PINHOLE or OMNIDIRECTIONAL
    @param nCameras number of cameras
    @fileName filename of string list that are used for calibration, the file is generated
    by imagelist_creator from OpenCv samples. The first one in the list is the pattern filename.
    @patternWidth the physical width of pattern, in user defined unit.
    @patternHeight the physical height of pattern, in user defined unit.
    @showExtration whether show extracted features and feature filtering.
    @nMiniMatches minimal number of matched features for a frame.
	@flags Calibration flags
    @criteria optimization stopping criteria.
    @detector feature detector that detect feature points in pattern and images.
    @descriptor feature descriptor.
    @matcher feature matcher.
    */
    MultiCameraCalibrationCharuco(int cameraType, int nCameras, const std::string& fileName, int tileX, int tileY, float squareWidth, float tagWidth, aruco::PredefinedDictionaryType type = aruco::PredefinedDictionaryType::DICT_4X4_100, int verbose = 0, int showExtration = 0, int flags = 0,
        TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 200, 1e-7));

    /* @brief load images
    */
    void loadImages();

    std::vector<std::string> readStringListFromFolder(const std::string& path);

    /* @brief initialize multiple camera calibration. It calibrates each camera individually.
    */
    void initialize();

    /* @brief optimization extrinsic parameters
    */
    double optimizeExtrinsics();

    /* @brief run multi-camera camera calibration, it runs loadImage(), initialize() and optimizeExtrinsics()
    */
    double run();

    /* @brief write camera parameters to file.
    */
    void writeParameters(const std::string& filename);

    /* @brief Get Camera matrices
    */
   const std::vector<cv::Mat> getCameraMatrices() const { return _cameraMatrix; }

    /* @brief Get Camera distortions
    */
   const std::vector<cv::Mat> getCameraDistortions() const { return _distortCoeffs; }

    /* @brief Get Camera Poses
    */
   const std::vector<vertex> getCameraPoses() const { return _vertexList; }

    /* @brief Get Camera xi
    */
   const std::vector<cv::Mat> getCameraXi() const { return _xi; }

private:
    std::vector<std::string> readStringList();

    int getPhotoVertex(int timestamp);

    void graphTraverse(const Mat& G, int begin, std::vector<int>& order, std::vector<int>& pre);

    void findRowNonZero(const Mat& row, Mat& idx);

    std::vector<Mat> detectCharucoCorners(const Mat& image, int imgIdx, int cameraIdx);

    void computeJacobianExtrinsic(const Mat& extrinsicParams, Mat& JTJ_inv, Mat& JTE);

    void computePhotoCameraJacobian(const Mat& rvecPhoto, const Mat& tvecPhoto, const Mat& rvecCamera,
        const Mat& tvecCamera, Mat& rvecTran, Mat& tvecTran, const Mat& objectPoints, const Mat& imagePoints, const Mat& K,
        const Mat& distort, const Mat& xi, Mat& jacobianPhoto, Mat& jacobianCamera, Mat& E);

    void compose_motion(InputArray _om1, InputArray _T1, InputArray _om2, InputArray _T2, Mat& om3, Mat& T3, Mat& dom3dom1,
        Mat& dom3dT1, Mat& dom3dom2, Mat& dom3dT2, Mat& dT3dom1, Mat& dT3dT1, Mat& dT3dom2, Mat& dT3dT2);

    void JRodriguesMatlab(const Mat& src, Mat& dst);
    void dAB(InputArray A, InputArray B, OutputArray dABdA, OutputArray dABdB);

    double computeProjectError(Mat& parameters);

    void vector2parameters(const Mat& parameters, std::vector<Vec3f>& rvecVertex, std::vector<Vec3f>& tvecVertexs);
    void parameters2vector(const std::vector<Vec3f>& rvecVertex, const std::vector<Vec3f>& tvecVertex, Mat& parameters);

    int _camType; //PINHOLE, FISHEYE or OMNIDIRECTIONAL
    int _nCamera;
    int _flags;
    aruco::PredefinedDictionaryType _dictionaryType;
	int _verbose;
    double _error;
    int _tileX, _tileY;
    float _squareWidth, _tagWidth;
    TermCriteria _criteria;
    std::string _filename;
    int _showExtraction;

    std::vector<edge> _edgeList;
    std::vector<vertex> _vertexList;
    std::vector<std::vector<cv::Mat> > _objectPointsForEachCamera;
    std::vector<std::vector<cv::Mat> > _imagePointsForEachCamera;
    std::vector<cv::Mat> _cameraMatrix;
    std::vector<cv::Mat> _distortCoeffs;
    std::vector<cv::Mat> _xi;
    std::vector<std::vector<Mat> > _omEachCamera, _tEachCamera;
    aruco::Dictionary _dictionary;
    aruco::CharucoBoard _board;
    aruco::DetectorParameters _params;
    Ptr<aruco::Dictionary> _ptrDictionary;
    Ptr<aruco::CharucoBoard> _ptrBoard;
    Ptr<aruco::DetectorParameters> _ptrParams;
};

//! @}

}} // namespace multicalib, cv
#endif