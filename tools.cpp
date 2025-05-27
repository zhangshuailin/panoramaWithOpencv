#include "tools.h"
#define MINPICSNUM 3 //参与拼接的最小图片数目
void find_jpg_files(const std::filesystem::path& dir, std::vector<std::string>& files) {
	for (const auto& entry : std::filesystem::directory_iterator(dir)) {
		if (entry.is_regular_file() && (entry.path().extension() == ".JPG" || entry.path().extension() == ".jpg")) {
			files.push_back(entry.path().string());
		}
		else if (entry.is_directory()) {
			find_jpg_files(entry.path(), files); // 递归调用
		}
	}
}

int loadAllImages(std::vector<std::string> imgs_path, std::vector<cv::Mat>& vctMat)
{
	//加载所有要拼接的图像
	uint num_images = static_cast<int>(imgs_path.size());
	if (num_images < MINPICSNUM)
		return ERROR_INSUFFICIENT_VALID_IMAGES;

	std::vector<cv::Mat> vectfull_img(num_images);
	std::vector<cv::Size> full_img_sizes(num_images);
	uint i = 0;
	while (i < num_images)
	{
		vectfull_img[i] = cv::imread(imgs_path[i]);
		if (vectfull_img[i].data == NULL || vectfull_img[i].empty())
			return ERROR_READ_IMAGE_ERROR;
		++i;
	}
	vctMat = vectfull_img;
	return OK;
}
