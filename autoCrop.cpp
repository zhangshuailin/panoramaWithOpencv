#include "autoCrop.h"
// 判断 Line 是否有效
bool IsLineValid(const Line& line) {
	return line.start < line.end;
}

// 计算 Line 的长度
int Length(const Line& line) {
	return line.end - line.start;
}

// 在一列中找到最长的 Line
std::optional<Line> FindLongestLineInColumn(const cv::Mat& column) {
	std::optional<Line> current_line;
	std::optional<Line> longest_line;

	for (int i = 0; i < column.rows; i++) {
		if (column.at<uchar>(i, 0) == 255) { // 前景像素
			if (!current_line) {
				current_line = { i, i + 1 };
			}
			else {
				current_line->end = i + 1;
			}
		}
		else { // 背景像素
			if (current_line) {
				if (!longest_line || Length(*current_line) > Length(*longest_line)) {
					longest_line = current_line;
				}
				current_line.reset();
			}
		}
	}

	// 处理最后一行的情形
	if (current_line && (!longest_line || Length(*current_line) > Length(*longest_line))) {
		longest_line = current_line;
	}

	return longest_line;
}

// 从 seed 位置开始，向左右扩展，寻找最大的 Rect
cv::Rect FindLargestCropFromSeed(const std::vector<Line>& lines, const Line& invalid_line, int seed) {
	if (!IsLineValid(lines[seed])) {
		return cv::Rect(); // 返回空 Rect
	}

	// 初始化矩形
	int left = seed;
	int right = seed + 1;
	int top = lines[seed].start;
	int bottom = lines[seed].end;
	cv::Rect bestRect(left, top, right - left, bottom - top);

	while (left >= 0 && right <= lines.size()) {
		// 尝试向左扩展
		if (left > 0 && IsLineValid(lines[left - 1])) {
			int newTop = std::max(top, lines[left - 1].start);
			int newBottom = std::min(bottom, lines[left - 1].end);
			if (newTop < newBottom) {  // 确保有效高度
				cv::Rect newRect(left - 1, newTop, right - (left - 1), newBottom - newTop);
				if (newRect.area() > bestRect.area()) {
					bestRect = newRect;
					left--;
					top = newTop;
					bottom = newBottom;
					continue;
				}
			}
		}

		// 尝试向右扩展
		if (right < lines.size() && IsLineValid(lines[right])) {
			int newTop = std::max(top, lines[right].start);
			int newBottom = std::min(bottom, lines[right].end);
			if (newTop < newBottom) { // 确保有效高度
				cv::Rect newRect(left, newTop, right + 1 - left, newBottom - newTop);
				if (newRect.area() > bestRect.area()) {
					bestRect = newRect;
					right++;
					top = newTop;
					bottom = newBottom;
					continue;
				}
			}
		}
		break; // 无法扩展
	}
	return bestRect;
}

// 主函数：寻找最大的可裁剪区域
cv::Rect FindLargestCrop( cv::Mat& mask, int samplingDistance) {
	const int downscale = 16;
	resize(mask, mask, mask.size() / downscale,0,0,cv::INTER_NEAREST);

	// 1. 检查图像
	if (mask.empty() || mask.type() != CV_8UC1) {
		return cv::Rect(); // 返回空 Rect
	}

	// 2. 找到每列最长的 Line
	std::vector<Line> lines(mask.cols);
	for (int i = 0; i < mask.cols; i++) {
		auto longest_line = FindLongestLineInColumn(mask.col(i));
		lines[i] = longest_line.value_or(Line{ 0, 0 }); // 如果该列没有有效 Line， 则赋值 start = end =0
	}

	// 3. 多次采样，寻找最大的 Rect
	cv::Rect largestRect;
	int num_samples = 1 + mask.cols / samplingDistance;
	for (int i = 0; i < num_samples; i++) {
		int start = (i + 1) * mask.cols / (num_samples + 1);
		cv::Rect currentRect = FindLargestCropFromSeed(lines, { 0, 0 }, start);

		if (currentRect.area() > largestRect.area()) {
			largestRect = currentRect;
		}
	}


	//upsacle to initial size
	int new_width = static_cast<int>(largestRect.width * downscale);
	int new_height = static_cast<int>(largestRect.height * downscale);
	int new_x = largestRect.x * downscale;
	int new_y = largestRect.y * downscale;
	cv::Rect init_rect(new_x, new_y, new_width, new_height);
	return init_rect;
}

