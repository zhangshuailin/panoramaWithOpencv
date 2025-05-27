#include "Stitcher.h"
#define MINPICSNUM 3 
#define THREADSWITCH
#define THREADNUM 3
Stitcher::Stitcher()
{
	warpType = 0;
}

Stitcher::Stitcher(PanoParam pparam)
{
	panoparam.fdwork_scale = pparam.fdwork_scale;
	panoparam.seam_work_aspect = pparam.seam_work_aspect;
	panoparam.match_conf = pparam.match_conf;
	panoparam.conf_thresh = pparam.conf_thresh;
	panoparam.range_width = pparam.range_width;
	panoparam.resScale = pparam.resScale;
	warpType = 0;
}

Stitcher::~Stitcher()
{

}
std::vector<cv::detail::ImageFeatures> Stitcher::detectFeature(std::vector<cv::Mat> vctfullMat)
{
	size_t num_images = vctfullMat.size();
	cv::Ptr<cv::Feature2D> finder = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f);//SURF和akaze都不错
	std::vector<cv::Size> full_img_sizes(num_images);
	//图像缩放fdwork_scale，然后特征检测
	std::vector<cv::detail::ImageFeatures> features(num_images);
#ifdef THREADSWITCH
	int images_per_thread = (int)num_images / THREADNUM;
	std::vector<std::thread> threads;
	for (int t = 0; t < THREADNUM; ++t) {
		int start_idx = t * images_per_thread;
		int end_idx = (t == THREADNUM - 1) ? (int)num_images : start_idx + images_per_thread;
		threads.emplace_back([&, start_idx, end_idx, t]() {

			for (int i = start_idx; i < end_idx; ++i) {
				cv::Mat img;
				cv::resize(vctfullMat[i], img, cv::Size(), panoparam.fdwork_scale, panoparam.fdwork_scale, cv::INTER_LINEAR_EXACT);
				computeImageFeatures(finder, img, features[i]);
				features[i].img_idx = i;
				//std::cout << "Thread " << t + 1 << ": Processed image " << i << std::endl;
			}
			});
	}
	std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });// 等待所有线程完成
	return features;
#else
	uint i = 0;
	while (i < num_images)
	{
		cv::Mat img;
		cv::resize(vctfullMat[i], img, cv::Size(), panoparam.fdwork_scale, panoparam.fdwork_scale, cv::INTER_LINEAR_EXACT);
		computeImageFeatures(finder, img, features[i]);
		features[i].img_idx = i;
		++i;
	}
	return features;
#endif // THREADSWITCH

}

int Stitcher::findPairWiseMatcherGetLeavingMaxConnectCompntIndex(std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches, std::vector<int>& vctIndex)
{
	/*搜索图片i与前后range_width的图片的图像特征匹配度，找到图片A中特征点0与图片B中最佳匹配的特征点1，
	其它特征点与这两个特征点距离越大说明这个匹配的特征点越准确，所以panoparam.range_width越大越能突显图片之间的相邻关系;
	panoparam.match_conf越小，找到的特征配对点越准确
	pairwise_matches保留了配对的特征点坐标*/
	cv::Ptr<cv::detail::FeaturesMatcher> matcher = cv::makePtr<cv::detail::BestOf2NearestRangeMatcher>(panoparam.range_width, true, panoparam.match_conf);
	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();
	//ofstream f("graph.dot");
	//f << matchesGraphAsString(imgs_path, pairwise_matches, panoparam.conf_thresh);


	//保留认为是全景连通区域的图片，panoparam.conf_thresh参数约束了特征点之间配准置信度
	std::vector<int> indices = leaveBiggestComponent(features, pairwise_matches, panoparam.conf_thresh);
	//确保筛选之后保留足够的图片
	if (indices.size() < MINPICSNUM)
		return ERROR_INSUFFICIENT_VALID_IMAGES;
	vctIndex = indices;
	return OK;
}

void Stitcher::leaveValidImages(std::vector<int> vctIndex, std::vector<cv::Mat> vctfullMat, std::vector<cv::Mat>& img_subset/*, std::vector<Size>& full_img_sizes_subset*/)
{
	uint i = 0;
	while (i < vctIndex.size())
	{
		img_subset.push_back(vctfullMat[vctIndex[i]]);
		++i;
	}
}

int Stitcher::calcOptRotMat(std::vector<cv::detail::ImageFeatures> features, std::vector<cv::detail::MatchesInfo> pairwise_matches, std::vector<cv::detail::CameraParams>& cameras)
{
	//计算单应矩阵并估计出旋转矩阵
	cv::Ptr<cv::detail::Estimator> estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();
	if (!(*estimator)(features, pairwise_matches, cameras))
		return HOMOGRAPHY_ESTIMATION_FAILED;
	uint i = 0;
	while (i < cameras.size())
	{
		cameras[i].R.convertTo(cameras[i].R, CV_32F);
		++i;
	}

	//配置 Bundle Adjustment (BA) 优化器，用于全局地优化相机参数和图像间的相对位置，以获得更精准的全景图,主要优化深度信息带来的重映射误差
	cv::Ptr<cv::detail::BundleAdjusterBase> adjuster = cv::makePtr<cv::detail::BundleAdjusterRay>();//可选，最终根据效果选择了ray,不适合近景场景
	adjuster->setConfThresh(panoparam.conf_thresh);
	/*Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	refine_mask(0, 0) = 1;refine_mask(0, 1) = 1;refine_mask(0, 2) = 1;refine_mask(1, 1) = 1;refine_mask(1, 2) = 1;*/
	cv::Mat_<uchar> refine_mask = cv::Mat::ones(3, 3, CV_8U);//全部优化
	adjuster->setRefinementMask(refine_mask);
	if (!(*adjuster)(features, pairwise_matches, cameras))
		return HOMOGRAPHY_BAOPT_FAILED;

	//wavecorrect防止图片弯曲
	std::vector<cv::Mat> rmats;
	for (size_t i = 0; i < cameras.size(); ++i)
		rmats.push_back(cameras[i].R.clone());
	cv::detail::waveCorrect(rmats, cv::detail::WAVE_CORRECT_AUTO);
	for (size_t i = 0; i < cameras.size(); ++i)
		cameras[i].R = rmats[i];
	return 0;

}
void Stitcher::setCameraParams(Projector& projector, cv::InputArray _K, cv::InputArray _R, cv::InputArray _T)
{
	cv::Mat K = _K.getMat(), R = _R.getMat(), T = _T.getMat();

	CV_Assert(K.size() == cv::Size(3, 3) && K.type() == CV_32F);
	CV_Assert(R.size() == cv::Size(3, 3) && R.type() == CV_32F);
	CV_Assert((T.size() == cv::Size(1, 3) || T.size() == cv::Size(3, 1)) && T.type() == CV_32F);

	cv::Mat_<float> K_(K);
	projector.k[0] = K_(0, 0); projector.k[1] = K_(0, 1); projector.k[2] = K_(0, 2);
	projector.k[3] = K_(1, 0); projector.k[4] = K_(1, 1); projector.k[5] = K_(1, 2);
	projector.k[6] = K_(2, 0); projector.k[7] = K_(2, 1); projector.k[8] = K_(2, 2);

	cv::Mat_<float> Rinv = R.t();
	projector.rinv[0] = Rinv(0, 0); projector.rinv[1] = Rinv(0, 1); projector.rinv[2] = Rinv(0, 2);
	projector.rinv[3] = Rinv(1, 0); projector.rinv[4] = Rinv(1, 1); projector.rinv[5] = Rinv(1, 2);
	projector.rinv[6] = Rinv(2, 0); projector.rinv[7] = Rinv(2, 1); projector.rinv[8] = Rinv(2, 2);

	cv::Mat_<float> R_Kinv = R * K.inv();
	projector.r_kinv[0] = R_Kinv(0, 0); projector.r_kinv[1] = R_Kinv(0, 1); projector.r_kinv[2] = R_Kinv(0, 2);
	projector.r_kinv[3] = R_Kinv(1, 0); projector.r_kinv[4] = R_Kinv(1, 1); projector.r_kinv[5] = R_Kinv(1, 2);
	projector.r_kinv[6] = R_Kinv(2, 0); projector.r_kinv[7] = R_Kinv(2, 1); projector.r_kinv[8] = R_Kinv(2, 2);

	cv::Mat_<float> K_Rinv = K * Rinv;
	projector.k_rinv[0] = K_Rinv(0, 0); projector.k_rinv[1] = K_Rinv(0, 1); projector.k_rinv[2] = K_Rinv(0, 2);
	projector.k_rinv[3] = K_Rinv(1, 0); projector.k_rinv[4] = K_Rinv(1, 1); projector.k_rinv[5] = K_Rinv(1, 2);
	projector.k_rinv[6] = K_Rinv(2, 0); projector.k_rinv[7] = K_Rinv(2, 1); projector.k_rinv[8] = K_Rinv(2, 2);

	cv::Mat_<float> T_(T.reshape(0, 3));
	projector.t[0] = T_(0, 0);
	projector.t[1] = T_(1, 0);
	projector.t[2] = T_(2, 0);
}
void Stitcher::mapForward(Projector projector, float x, float y, float& u, float& v)
{
	float x_ = projector.r_kinv[0] * x + projector.r_kinv[1] * y + projector.r_kinv[2];
	float y_ = projector.r_kinv[3] * x + projector.r_kinv[4] * y + projector.r_kinv[5];
	float z_ = projector.r_kinv[6] * x + projector.r_kinv[7] * y + projector.r_kinv[8];

	x_ = projector.t[0] + x_ / z_ * (1 - projector.t[2]);
	y_ = projector.t[1] + y_ / z_ * (1 - projector.t[2]);

	u = projector.scale * x_;
	v = projector.scale * y_;
}
void Stitcher::mapBackward(Projector projector, float u, float v, float& x, float& y)
{
	u = u / projector.scale - projector.t[0];
	v = v / projector.scale - projector.t[1];

	float z;
	x = projector.k_rinv[0] * u + projector.k_rinv[1] * v + projector.k_rinv[2] * (1 - projector.t[2]);
	y = projector.k_rinv[3] * u + projector.k_rinv[4] * v + projector.k_rinv[5] * (1 - projector.t[2]);
	z = projector.k_rinv[6] * u + projector.k_rinv[7] * v + projector.k_rinv[8] * (1 - projector.t[2]);

	x /= z;
	y /= z;
}
void Stitcher::calcRotation(Projector projector, double& thetaPitch, double& thetaYaw, double& thetaRoll)
{
	double r11 = projector.rinv[0], r12 = projector.rinv[1], r13 = projector.rinv[2];
	double r21 = projector.rinv[3], r22 = projector.rinv[4], r23 = projector.rinv[5];
	double r31 = projector.rinv[6], r32 = projector.rinv[7], r33 = projector.rinv[8];
	/*cout << r11 << " " << r12 << " " << r13 << endl;
	cout << r21 << " " << r22 << " " << r23 << endl;
	cout << r31 << " " << r32 << " " << r33 << endl;*/

	thetaPitch = std::atan2(r32, r33); // pitch
	thetaYaw = std::atan2(-r31, std::sqrt(r32 * r32 + r33 * r33)); // yaw
	thetaRoll = std::atan2(r21, r11); // roll
	const double PI = 3.14159265358979323846;
	thetaPitch = thetaPitch * (180.0 / PI);
	thetaYaw = thetaYaw * (180.0 / PI);
	thetaRoll = thetaRoll * (180.0 / PI);
}
void Stitcher::detectResultRoi(cv::Size src_size, Projector projector, cv::Point& dst_tl, cv::Point& dst_br)
{


	float tl_uf = std::numeric_limits<float>::max();
	float tl_vf = std::numeric_limits<float>::max();
	float br_uf = -std::numeric_limits<float>::max();
	float br_vf = -std::numeric_limits<float>::max();

	float u0, v0;
	mapForward(projector, 0, 0, u0, v0);
	tl_uf = std::min(tl_uf, u0); tl_vf = std::min(tl_vf, v0);
	br_uf = std::max(br_uf, u0); br_vf = std::max(br_vf, v0);

	float u1, v1;
	mapForward(projector, 0, static_cast<float>(src_size.height - 1), u1, v1);
	tl_uf = std::min(tl_uf, u1); tl_vf = std::min(tl_vf, v1);
	br_uf = std::max(br_uf, u1); br_vf = std::max(br_vf, v1);

	float u2, v2;
	mapForward(projector, static_cast<float>(src_size.width - 1), 0, u2, v2);
	tl_uf = std::min(tl_uf, u2); tl_vf = std::min(tl_vf, v2);
	br_uf = std::max(br_uf, u2); br_vf = std::max(br_vf, v2);

	float u3, v3;
	mapForward(projector, static_cast<float>(src_size.width - 1), static_cast<float>(src_size.height - 1), u3, v3);
	tl_uf = std::min(tl_uf, u3); tl_vf = std::min(tl_vf, v3);
	br_uf = std::max(br_uf, u3); br_vf = std::max(br_vf, v3);

	dst_tl.x = static_cast<int>(tl_uf);
	dst_tl.y = static_cast<int>(tl_vf);
	dst_br.x = static_cast<int>(br_uf);
	dst_br.y = static_cast<int>(br_vf);


	double thetaPitch(0.0), thetaYaw(0.0), thetaRoll(0.0);
	calcRotation(projector, thetaPitch, thetaYaw, thetaRoll);
	if (thetaPitch < -10 && thetaYaw > 10)//往左上拉伸
	{
		dst_tl.x = (int)(u1 - abs(u3 - u1) * 0.2);
		dst_tl.y = (int)(v2 - abs(v3 - v2) * 0.4);
	}
	if (thetaPitch > 10 && thetaYaw < -10)//往右下拉伸
	{
		dst_br.x = (int)(u2 + abs(u2 - u0) * 0.2);
		dst_br.y = (int)(v1 + abs(v1 - v0) * 0.4);
	}
	if (thetaPitch > 10 && thetaYaw > 10)//往左下拉伸
	{
		dst_br.y = (int)(v3 + abs(v3 - v2) * 0.4);
		dst_tl.x = (int)(u0 - abs(u2 - u0) * 0.2);
	}
	if (thetaPitch < -10 && thetaYaw < -10)//往右上拉伸
	{
		dst_tl.y = (int)(v0 - abs(v1 - v0) * 0.4);
		dst_br.x = (int)(u3 + abs(u3 - u1) * 0.2);
	}
}
cv::Rect Stitcher::buildMapsFor00(cv::Size src_size, cv::InputArray K, cv::InputArray R, cv::InputArray T, float scale, cv::OutputArray _xmap, cv::OutputArray _ymap)
{
	Projector projector_;
	projector_.scale = scale;
	setCameraParams(projector_, K, R, T);
	cv::Point dst_tl, dst_br;
	detectResultRoi(src_size, projector_, dst_tl, dst_br);
	cv::Size dsize(dst_br.x - dst_tl.x, dst_br.y - dst_tl.y + 1);
	_xmap.create(dsize, CV_32FC1);
	_ymap.create(dsize, CV_32FC1);

#ifdef HAVE_OPENCL
	if (cv::ocl::isOpenCLActivated())
	{
		std::string warpTypeArray[8] = { "buildWarpPlaneMaps","buildWarpSphericalMaps","buildWarpCylindricalMaps" };

		cv::ocl::Kernel k(warpTypeArray[warpType].c_str(), cv::ocl::stitching::warpers_oclsrc);
		if (!k.empty())
		{
			int rowsPerWI = cv::ocl::Device::getDefault().isIntel() ? 4 : 1;
			cv::Mat k_rinv(1, 9, CV_32FC1, projector_.k_rinv), t(1, 3, CV_32FC1, projector_.t);
			cv::UMat uxmap = _xmap.getUMat(), uymap = _ymap.getUMat(), uk_rinv = k_rinv.getUMat(cv::ACCESS_READ), ut = t.getUMat(cv::ACCESS_READ);
			if (warpType == 0)
				k.args(cv::ocl::KernelArg::WriteOnlyNoSize(uxmap), cv::ocl::KernelArg::WriteOnly(uymap), cv::ocl::KernelArg::PtrReadOnly(uk_rinv), cv::ocl::KernelArg::PtrReadOnly(ut), dst_tl.x, dst_tl.y, 1 / projector_.scale, rowsPerWI);
			else
				k.args(cv::ocl::KernelArg::WriteOnlyNoSize(uxmap), cv::ocl::KernelArg::WriteOnly(uymap), cv::ocl::KernelArg::PtrReadOnly(uk_rinv), dst_tl.x, dst_tl.y, 1 / projector_.scale, rowsPerWI);
			size_t globalsize[2] = { (size_t)dsize.width, ((size_t)dsize.height + rowsPerWI - 1) / rowsPerWI };
			if (k.run(2, globalsize, NULL, true))
			{
				CV_IMPL_ADD(CV_IMPL_OCL);
				return cv::Rect(dst_tl, dst_br);
			}
		}
	}
#endif

	cv::Mat xmap = _xmap.getMat(), ymap = _ymap.getMat();

	float x, y;
	for (int v = dst_tl.y; v <= dst_br.y; ++v)
	{
		for (int u = dst_tl.x; u <= dst_br.x; ++u)
		{
			mapBackward(projector_, static_cast<float>(u), static_cast<float>(v), x, y);
			xmap.at<float>(v - dst_tl.y, u - dst_tl.x) = x;
			ymap.at<float>(v - dst_tl.y, u - dst_tl.x) = y;
		}
	}

	return cv::Rect(dst_tl, dst_br);
}
cv::Point Stitcher::warpFor00(cv::InputArray src, cv::InputArray K, cv::InputArray R, int interp_mode, int border_mode, cv::OutputArray dst, float scale)
{
	float tz[] = { 0.f, 0.f, 0.f };
	cv::Mat_<float> T(3, 1, tz);
	cv::UMat uxmap, uymap;
	cv::Rect dst_roi = buildMapsFor00(src.size(), K, R, T, scale, uxmap, uymap);
	remap(src, dst, uxmap, uymap, interp_mode, border_mode);
	return dst_roi.tl();
}
cv::Rect Stitcher::warpRoi(cv::Size src_size, cv::InputArray K, cv::InputArray R, float scale)
{
	Projector projector_;
	projector_.scale = scale;
	float tz[] = { 0.f, 0.f, 0.f };
	cv::Mat_<float> T(3, 1, tz);
	setCameraParams(projector_, K, R, T);
	cv::Point dst_tl, dst_br;
	detectResultRoi(src_size, projector_, dst_tl, dst_br);
	return cv::Rect(dst_tl, cv::Point(dst_br.x + 1, dst_br.y + 1));
}

void Stitcher::warpSmallImageForSeamOptAndExposure(std::vector<cv::Mat> vctfullMat, std::vector<cv::detail::CameraParams> cameras, std::vector<cv::UMat>& seemMasks_warped, cv::Ptr < cv::detail::ExposureCompensator >& PtrCompensator)
{
	float warped_image_scale = 0.0;
	std::vector<double> focals;//焦距中值
	for (size_t i = 0; i < cameras.size(); ++i)
		focals.push_back(cameras[i].focal);
	std::sort(focals.begin(), focals.end());
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;



	size_t num_images = vctfullMat.size();
	std::vector<cv::UMat> masks_warped(num_images);
	std::vector<cv::UMat> images_warped(num_images);
	std::vector<cv::Size> sizes(num_images);
	std::vector<cv::Point> corners(num_images);
	std::vector<cv::UMat> images_warped_f(num_images);

	// Warp images and their masks
	float scale = static_cast<float>(warped_image_scale * panoparam.resScale * panoparam.seam_work_aspect);
#ifdef THREADSWITCH
	int images_per_thread = (int)num_images / THREADNUM;
	std::vector<std::thread> threads;
	for (int t = 0; t < THREADNUM; ++t) {
		int start_idx = t * images_per_thread;
		int end_idx = (t == THREADNUM - 1) ? (int)num_images : start_idx + images_per_thread;
		threads.emplace_back([&, start_idx, end_idx, t]() {

			for (int i = start_idx; i < end_idx; ++i) {

				//WARP READY,BUILD MAP
				cv::Size matForSeamAndExpSize(uint(vctfullMat[i].cols * panoparam.seam_work_aspect * panoparam.fdwork_scale), uint(vctfullMat[i].rows * panoparam.seam_work_aspect * panoparam.fdwork_scale));
				//T
				float tz[] = { 0.f, 0.f, 0.f }; cv::Mat_<float> T(3, 1, tz);
				//K
				cv::Mat_<float> K; cameras[i].K().convertTo(K, CV_32F);
				float swa = (float)panoparam.seam_work_aspect;
				K(0, 0) *= swa; K(0, 2) *= swa;
				K(1, 1) *= swa; K(1, 2) *= swa;
				//R
				cv::Mat_<float> R = cameras[i].R;

				cv::UMat uxmap, uymap;
				cv::Rect dst_roi = buildMapsFor00(matForSeamAndExpSize, K, R, T, scale, uxmap, uymap);

				corners[i] = dst_roi.tl();

				cv::UMat masktemp(matForSeamAndExpSize, CV_8U, cv::Scalar(255));
				remap(masktemp, masks_warped[i], uxmap, uymap, cv::INTER_NEAREST, cv::BORDER_CONSTANT);

				cv::UMat matForSeamAndExp;
				resize(vctfullMat[i], matForSeamAndExp, matForSeamAndExpSize, 0, 0, cv::INTER_LINEAR);
				remap(matForSeamAndExp, images_warped[i], uxmap, uymap, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
			}
			});
	}
	std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });// 等待所有线程完成
#else
	uint i = 0;
	while (i < num_images)
	{

		//WARP READY,BUILD MAP
		cv::Size matForSeamAndExpSize(uint(vctfullMat[i].cols * panoparam.seam_work_aspect * panoparam.fdwork_scale), uint(vctfullMat[i].rows * panoparam.seam_work_aspect * panoparam.fdwork_scale));
		//T
		float tz[] = { 0.f, 0.f, 0.f }; cv::Mat_<float> T(3, 1, tz);
		//K
		cv::Mat_<float> K; cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)panoparam.seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;
		//R
		cv::Mat_<float> R = cameras[i].R;

		cv::UMat uxmap, uymap;
		cv::Rect dst_roi = buildMapsFor00(matForSeamAndExpSize, K, R, T, scale, uxmap, uymap);

		corners[i] = dst_roi.tl();

		cv::UMat masktemp(matForSeamAndExpSize, CV_8U, cv::Scalar(255));
		remap(masktemp, masks_warped[i], uxmap, uymap, cv::INTER_NEAREST, cv::BORDER_CONSTANT);

		cv::UMat matForSeamAndExp;
		resize(vctfullMat[i], matForSeamAndExp, matForSeamAndExpSize, 0, 0, cv::INTER_LINEAR);
		remap(matForSeamAndExp, images_warped[i], uxmap, uymap, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
		++i;
	}
#endif // THREADSWITCH



	//曝光补偿
	cv::Ptr<cv::detail::ExposureCompensator> compensator = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::CHANNELS_BLOCKS);
	cv::detail::BlocksCompensator* bcompensator = dynamic_cast<cv::detail::BlocksCompensator*>(compensator.get());
	int expos_comp_nr_feeds = 1;
	int expos_comp_nr_filtering = 1;
	int expos_comp_block_size = 128;
	bcompensator->setNrFeeds(expos_comp_nr_feeds);
	bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
	bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
	compensator->feed(corners, images_warped, masks_warped);
	PtrCompensator = compensator;
	//"no""voronoi""gc_color""gc_colorgrad""dp_color""dp_colorgrad"
	cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR_GRAD);
	seam_finder->find(images_warped, corners, masks_warped);
	seemMasks_warped = masks_warped;
}


cv::Ptr<cv::detail::Blender> Stitcher::createBlend(std::vector<cv::Point> corners, std::vector<cv::Size> sizes)
{
	float blend_strength = 5;
	cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, false);
	cv::Size dst_sz = cv::detail::resultRoi(corners, sizes).size();
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
	cv::detail::MultiBandBlender* mb = dynamic_cast<cv::detail::MultiBandBlender*>(blender.get());
	mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
	blender->prepare(corners, sizes);
	return blender;
}

void Stitcher::blendImages(std::vector<cv::Mat> vctfullMat, std::vector<cv::detail::CameraParams> cameras, cv::Ptr<cv::detail::ExposureCompensator> PtrCompensator, std::vector<cv::UMat> masks_warped, cv::Mat& resultMat, cv::Mat& resultMask)
{
	float warped_image_scale = 0.0;
	std::vector<double> focals;//焦距中值
	for (size_t i = 0; i < cameras.size(); ++i)
		focals.push_back(cameras[i].focal);
	std::sort(focals.begin(), focals.end());
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	//按比恢复cameras数据
	//cout << "按比恢复cameras数据start" << endl;
	int num_images = (int)(cameras.size());
	std::vector<cv::Point> corners(num_images);
	std::vector<cv::Size> sizes(num_images);
	float compose_work_aspect = (float)(1.0 / panoparam.fdwork_scale);
	float scale = warped_image_scale * panoparam.resScale * compose_work_aspect;
	int i = 0;
	while (i < num_images)
	{
		// Update intrinsics，Update corner and size
		cameras[i].focal *= compose_work_aspect;
		cameras[i].ppx *= compose_work_aspect;
		cameras[i].ppy *= compose_work_aspect;
		cv::Size sz = vctfullMat[i].size();
		cv::Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		cv::Rect roi = warpRoi(sz, K, cameras[i].R, scale);
		corners[i] = roi.tl();
		sizes[i] = roi.size();
		//cout << corners[i] << " " << sizes[i] << endl;
		++i;
	}

	cv::Ptr<cv::detail::Blender> blender = createBlend(corners, sizes);
#ifdef THREADSWITCH
	std::vector<cv::UMat> img_warped(num_images);
	std::vector<cv::UMat> mask_warped(num_images);
	int images_per_thread = (int)num_images / THREADNUM;
	std::vector<std::thread> threads;
	for (int t = 0; t < THREADNUM; ++t)
	{
		int start_idx = t * images_per_thread;
		int end_idx = (t == THREADNUM - 1) ? (int)num_images : start_idx + images_per_thread;
		threads.emplace_back([&, start_idx, end_idx, t]() {

			for (int img_idx = start_idx; img_idx < end_idx; ++img_idx) {
				//cv::UMat img_warped;
				//Warp ready,build map
				cv::Size imgSize = vctfullMat[img_idx].size();
				float tz[] = { 0.f, 0.f, 0.f }; cv::Mat_<float> T(3, 1, tz);
				cv::Mat K; cameras[img_idx].K().convertTo(K, CV_32F);
				thread_local cv::UMat uxmap, uymap;
				cv::Rect dst_roi = buildMapsFor00(imgSize, K, cameras[img_idx].R, T, scale, uxmap, uymap);
				// Warp image
				thread_local cv::UMat img; vctfullMat[img_idx].copyTo(img);
				remap(img, img_warped[img_idx], uxmap, uymap, cv::INTER_LINEAR, cv::BORDER_REFLECT);
				// Warp image mask
				thread_local cv::UMat mask(imgSize, CV_8U, 255); //thread_local cv::UMat mask_warped;
				remap(mask, mask_warped[img_idx], uxmap, uymap, cv::INTER_NEAREST, cv::BORDER_CONSTANT);

				// 接缝线处mask处理
				thread_local cv::UMat dilated_mask, seam_mask;
				cv::dilate(masks_warped[img_idx], dilated_mask, cv::Mat());
				cv::resize(dilated_mask, seam_mask, mask_warped[img_idx].size(), 0, 0, cv::INTER_NEAREST);
				bitwise_and(seam_mask, mask_warped[img_idx], mask_warped[img_idx]);
				// Compensate exposure
				PtrCompensator->apply(img_idx, cv::Point(), img_warped[img_idx], cv::Mat());
			}
			});
	}
	std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });// 等待所有线程完成
	for (int i = 0; i < num_images; i++)
	{
		// 将corner和扭曲后的img喂给拼接器
		blender->feed(img_warped[i], mask_warped[i], corners[i]);
	}

#else
	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		cv::UMat img_warped;
		//Warp ready,build map
		cv::Size imgSize = vctfullMat[img_idx].size();
		float tz[] = { 0.f, 0.f, 0.f }; cv::Mat_<float> T(3, 1, tz);
		cv::Mat K; cameras[img_idx].K().convertTo(K, CV_32F);
		cv::UMat uxmap, uymap;
		cv::Rect dst_roi = buildMapsFor00(imgSize, K, cameras[img_idx].R, T, scale, uxmap, uymap);
		// Warp image
		cv::UMat img; vctfullMat[img_idx].copyTo(img);
		remap(img, img_warped, uxmap, uymap, cv::INTER_LINEAR, cv::BORDER_REFLECT);
		// Warp image mask
		cv::UMat mask(imgSize, CV_8U, 255); cv::UMat mask_warped;
		remap(mask, mask_warped, uxmap, uymap, cv::INTER_NEAREST, cv::BORDER_CONSTANT);

		// 接缝线处mask处理
		cv::UMat dilated_mask, seam_mask;
		cv::dilate(masks_warped[img_idx], dilated_mask, cv::Mat());
		cv::resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, cv::INTER_NEAREST);


		bitwise_and(seam_mask, mask_warped, mask_warped);// cv::Mat aaaaaa; img_warped.copyTo(aaaaaa); resize(aaaaaa, aaaaaa, aaaaaa.size() / 4); imshow("123123", aaaaaa); cv::waitKey(0);
		// Compensate exposure
		PtrCompensator->apply(img_idx, cv::Point(), img_warped, cv::Mat());

		// 将corner和扭曲后的img喂给拼接器
		blender->feed(img_warped, mask_warped, corners[img_idx]);
}
#endif // THREADSWITCH
	blender->blend(resultMat, resultMask);
}


int Stitcher::stitching(std::vector<cv::Mat> vctfullMat, cv::Mat& resultMask, cv::Mat& resultMat)
{
#ifdef HAVE_OPENCL
	cv::ocl::setUseOpenCL(true);
	/*cv::ocl::setUseOpenCL(panoparam.bUseOPENCL);
	cv::ocl::Device device = cv::ocl::Device::getDefault();
	cout << device.name() << endl;
	cout << device.image2DMaxWidth() << endl;
	cout << device.image3DMaxHeight() << endl;*/
#endif // HAVE_OPENCL


#ifdef ENABLE_LOG
	int64  t0(0); int64 t = t0 = cv::getTickCount();
#endif

	std::vector<cv::detail::ImageFeatures> features = detectFeature(vctfullMat);

#ifdef ENABLE_LOG
	LOGLN("detect features:" << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec"); t = cv::getTickCount();
#endif

	std::vector<int> vctIndex;
	std::vector<cv::detail::MatchesInfo> pairwise_matches;
	int re = findPairWiseMatcherGetLeavingMaxConnectCompntIndex(features, pairwise_matches, vctIndex);
	if (re)
		return re;
	std::vector<cv::Mat> img_subset;
	leaveValidImages(vctIndex, vctfullMat, img_subset);

#ifdef ENABLE_LOG
	LOGLN("PairwiseMatchandGetValidIndex:" << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec"); t = cv::getTickCount();
#endif

	std::vector<cv::detail::CameraParams> cameras;
	re = calcOptRotMat(features, pairwise_matches, cameras);
	if (re)
		return re;

#ifdef ENABLE_LOG
	LOGLN("calcOptRotMat, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec"); t = cv::getTickCount();
#endif

	std::vector<cv::UMat> seemMasks_warped;
	cv::Ptr<cv::detail::ExposureCompensator> compensator;
	warpSmallImageForSeamOptAndExposure(img_subset, cameras, seemMasks_warped, compensator);


#ifdef ENABLE_LOG
	LOGLN("warpSmallImageForSeamOptAndExposure, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec"); t = cv::getTickCount();
#endif


	blendImages(img_subset, cameras, compensator, seemMasks_warped, resultMat, resultMask);

#ifdef ENABLE_LOG
	LOGLN("blendImages, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec"); t = cv::getTickCount();
	LOGLN("apptotal, time: " << ((cv::getTickCount() - t0) / cv::getTickFrequency()) << " sec");
#endif
	return re;
}

