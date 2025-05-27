#pragma once
#include "Stitcher.h"
class StitcherSphere : public Stitcher
{
public:
	StitcherSphere();
	~StitcherSphere();
	StitcherSphere(PanoParam pparam);
	void detectResultRoi(cv::Size src_size, Projector projector, cv::Point& dst_tl, cv::Point& dst_br) override;
    void detectResultRoiByBorder(cv::Size src_size, Projector projector, cv::Point& dst_tl, cv::Point& dst_br);
	void mapForward(Projector projector, float x, float y, float& u, float& v) override;
	void mapBackward(Projector projector, float u, float v, float& x, float& y) override;
};

