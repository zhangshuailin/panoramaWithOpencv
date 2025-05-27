#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "global.h"
#include <optional>
// 定义 Line 结构体
struct Line {
	int start;
	int end;
};

// 判断 Line 是否有效
bool IsLineValid(const Line& line);
// 计算 Line 的长度
int Length(const Line& line);
// 在一列中找到最长的 Line
std::optional<Line> FindLongestLineInColumn(const cv::Mat& column);
// 从 seed 位置开始，向左右扩展，寻找最大的 Rect
cv::Rect FindLargestCropFromSeed(const std::vector<Line>& lines, const Line& invalid_line, int seed);
// 主函数：寻找最大的可裁剪区域
cv::Rect FindLargestCrop( cv::Mat& mask, int samplingDistance);
