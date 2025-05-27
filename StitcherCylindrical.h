#pragma once
#include "Stitcher.h"
class StitcherCylindrical :public Stitcher
{
public:
	StitcherCylindrical();
	~StitcherCylindrical();
	StitcherCylindrical(PanoParam pparam);
	void detectResultRoi(cv::Size src_size, Projector projector, cv::Point& dst_tl, cv::Point& dst_br) override;
	void mapForward(Projector projector, float x, float y, float& u, float& v) override;
	void mapBackward(Projector projector, float u, float v, float& x, float& y) override;
};

