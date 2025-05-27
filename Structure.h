#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
typedef struct paronomaParameter
{
	double fdwork_scale;   //提取特征配准的时候输入图像的尺寸缩放倍数
	double seam_work_aspect; //用于seam处理，在fdwork_scale缩放的基础上在缩放seam_work_aspect,过小会有拼接痕迹
	float match_conf; //筛选匹配的特征点，两个匹配的特征点与其他特征点差距越大，越有可能可靠。值越小越好
	float conf_thresh;//特征点匹配置信度，值越大越可靠，但可能会有极少的特征点
	int range_width;
	float resScale;
	paronomaParameter() : fdwork_scale(0.2), seam_work_aspect(1.0), match_conf(0.2f), conf_thresh(0.8f), range_width(6), resScale(0.8f){} // 构造函数
}PanoParam;

typedef struct StctProjector
{
	float scale;
	float k[9];
	float rinv[9];
	float r_kinv[9];
	float k_rinv[9];
	float t[3];
}Projector;