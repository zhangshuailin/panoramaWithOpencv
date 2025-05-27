#include "autoCrop.h"
// �ж� Line �Ƿ���Ч
bool IsLineValid(const Line& line) {
	return line.start < line.end;
}

// ���� Line �ĳ���
int Length(const Line& line) {
	return line.end - line.start;
}

// ��һ�����ҵ���� Line
std::optional<Line> FindLongestLineInColumn(const cv::Mat& column) {
	std::optional<Line> current_line;
	std::optional<Line> longest_line;

	for (int i = 0; i < column.rows; i++) {
		if (column.at<uchar>(i, 0) == 255) { // ǰ������
			if (!current_line) {
				current_line = { i, i + 1 };
			}
			else {
				current_line->end = i + 1;
			}
		}
		else { // ��������
			if (current_line) {
				if (!longest_line || Length(*current_line) > Length(*longest_line)) {
					longest_line = current_line;
				}
				current_line.reset();
			}
		}
	}

	// �������һ�е�����
	if (current_line && (!longest_line || Length(*current_line) > Length(*longest_line))) {
		longest_line = current_line;
	}

	return longest_line;
}

// �� seed λ�ÿ�ʼ����������չ��Ѱ������ Rect
cv::Rect FindLargestCropFromSeed(const std::vector<Line>& lines, const Line& invalid_line, int seed) {
	if (!IsLineValid(lines[seed])) {
		return cv::Rect(); // ���ؿ� Rect
	}

	// ��ʼ������
	int left = seed;
	int right = seed + 1;
	int top = lines[seed].start;
	int bottom = lines[seed].end;
	cv::Rect bestRect(left, top, right - left, bottom - top);

	while (left >= 0 && right <= lines.size()) {
		// ����������չ
		if (left > 0 && IsLineValid(lines[left - 1])) {
			int newTop = std::max(top, lines[left - 1].start);
			int newBottom = std::min(bottom, lines[left - 1].end);
			if (newTop < newBottom) {  // ȷ����Ч�߶�
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

		// ����������չ
		if (right < lines.size() && IsLineValid(lines[right])) {
			int newTop = std::max(top, lines[right].start);
			int newBottom = std::min(bottom, lines[right].end);
			if (newTop < newBottom) { // ȷ����Ч�߶�
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
		break; // �޷���չ
	}
	return bestRect;
}

// ��������Ѱ�����Ŀɲü�����
cv::Rect FindLargestCrop( cv::Mat& mask, int samplingDistance) {
	const int downscale = 16;
	resize(mask, mask, mask.size() / downscale,0,0,cv::INTER_NEAREST);

	// 1. ���ͼ��
	if (mask.empty() || mask.type() != CV_8UC1) {
		return cv::Rect(); // ���ؿ� Rect
	}

	// 2. �ҵ�ÿ����� Line
	std::vector<Line> lines(mask.cols);
	for (int i = 0; i < mask.cols; i++) {
		auto longest_line = FindLongestLineInColumn(mask.col(i));
		lines[i] = longest_line.value_or(Line{ 0, 0 }); // �������û����Ч Line�� ��ֵ start = end =0
	}

	// 3. ��β�����Ѱ������ Rect
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

