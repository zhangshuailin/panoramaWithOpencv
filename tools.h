#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "global.h"
void find_jpg_files(const std::filesystem::path& dir, std::vector<std::string>& files);
int loadAllImages(std::vector<std::string> imgs_path, std::vector<cv::Mat>& vctMat);


