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
// ���� Line �ṹ��
struct Line {
	int start;
	int end;
};

// �ж� Line �Ƿ���Ч
bool IsLineValid(const Line& line);
// ���� Line �ĳ���
int Length(const Line& line);
// ��һ�����ҵ���� Line
std::optional<Line> FindLongestLineInColumn(const cv::Mat& column);
// �� seed λ�ÿ�ʼ����������չ��Ѱ������ Rect
cv::Rect FindLargestCropFromSeed(const std::vector<Line>& lines, const Line& invalid_line, int seed);
// ��������Ѱ�����Ŀɲü�����
cv::Rect FindLargestCrop( cv::Mat& mask, int samplingDistance);
