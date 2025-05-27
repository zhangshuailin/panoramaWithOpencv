#include"Stitcher.h"
#include"StitcherSphere.h"
#include"StitcherCylindrical.h"
#include"tools.h"
int main()
{

	std::vector<std::string> imgs_path;
	find_jpg_files("../demo3x3/001_0003", imgs_path);

	
	std::vector<cv::Mat> vctMat;
	int re = loadAllImages(imgs_path, vctMat);
	cv::Mat resultMask;
	cv::Mat resultMat;

	PanoParam ppram;  
	Stitcher *stitcher = new StitcherSphere(ppram);
	re = stitcher->stitching(vctMat, resultMask, resultMat);
	delete stitcher;
	if (re!=OK)
		return re;
	cv::Rect roi = FindLargestCrop(resultMask, 1);
	cv::imwrite("resultname.jpg", resultMat(roi));
}
