#ifndef STITCHER_H
#define STITCHER_H
#pragma once
#pragma warning(disable: 4819)
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/opencl/ocl_defs.hpp"
#include "opencl_kernels_stitching.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "global.h"
#include "autoCrop.h"
#include "Structure.h"



// 拼接类
class Stitcher {
public:
	Stitcher();
	Stitcher(PanoParam pparam);
	~Stitcher();
	// 成员变量
	PanoParam panoparam;
	int warpType;

	// 成员函数
	void setCameraParams(Projector& projector, cv::InputArray _K = cv::Mat::eye(3, 3, CV_32F), cv::InputArray _R = cv::Mat::eye(3, 3, CV_32F), cv::InputArray _T = cv::Mat::zeros(3, 1, CV_32F));
	virtual void mapForward(Projector projector, float x, float y, float& u, float& v);
	virtual void mapBackward(Projector projector, float u, float v, float& x, float& y);
	void calcRotation(Projector projector, double& thetaPitch, double& thetaYaw, double& thetaRoll);
	virtual void detectResultRoi(cv::Size src_size, Projector projector, cv::Point& dst_tl, cv::Point& dst_br);
	virtual cv::Rect buildMapsFor00(cv::Size src_size, cv::InputArray K, cv::InputArray R, cv::InputArray T, float scale, cv::OutputArray _xmap, cv::OutputArray _ymap);
	cv::Point warpFor00(cv::InputArray src, cv::InputArray K, cv::InputArray R, int interp_mode, int border_mode, cv::OutputArray dst, float scale);
	cv::Rect warpRoi(cv::Size src_size, cv::InputArray K, cv::InputArray R, float scale);
	cv::Ptr<cv::detail::Blender> createBlend(std::vector<cv::Point> corners, std::vector<cv::Size> sizes);
	void blendImages(std::vector<cv::Mat> vctfullMat, std::vector < cv::detail::CameraParams > cameras, cv::Ptr<cv::detail::ExposureCompensator> PtrCompensator, std::vector<cv::UMat> masks_warped, cv::Mat& resultMat, cv::Mat& resultMask);
	std::vector<cv::detail::ImageFeatures> detectFeature(std::vector<cv::Mat> vctfullMat);
	int findPairWiseMatcherGetLeavingMaxConnectCompntIndex(std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches, std::vector<int>& vctIndex);
	void leaveValidImages(std::vector<int> vctIndex, std::vector<cv::Mat> vctfullMat, std::vector<cv::Mat>& img_subset);
	int stitching(std::vector<cv::Mat> vctfullMat, cv::Mat &resultMask, cv::Mat& resultMat);
	int calcOptRotMat(std::vector<cv::detail::ImageFeatures> features, std::vector<cv::detail::MatchesInfo> pairwise_matches, std::vector<cv::detail::CameraParams>& cameras);
	void warpSmallImageForSeamOptAndExposure(std::vector<cv::Mat> vctfullMat, std::vector<cv::detail::CameraParams> cameras, std::vector<cv::UMat>& seemMasks_warped, cv::Ptr<cv::detail::ExposureCompensator>& PtrCompensator);
};
#endif








