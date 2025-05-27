#include "StitcherCylindrical.h"

StitcherCylindrical::StitcherCylindrical()
{
    warpType = 2;
}

StitcherCylindrical::~StitcherCylindrical()
{
}

StitcherCylindrical::StitcherCylindrical(PanoParam pparam)
{
    panoparam.fdwork_scale = pparam.fdwork_scale;
    panoparam.seam_work_aspect = pparam.seam_work_aspect;
    panoparam.match_conf = pparam.match_conf;
    panoparam.conf_thresh = pparam.conf_thresh;
    panoparam.range_width = pparam.range_width;
    panoparam.resScale = pparam.resScale;
    warpType = 2;
}

void StitcherCylindrical::detectResultRoi(cv::Size src_size, Projector projector, cv::Point& dst_tl, cv::Point& dst_br)
{
    float tl_uf = (std::numeric_limits<float>::max)();
    float tl_vf = (std::numeric_limits<float>::max)();
    float br_uf = -(std::numeric_limits<float>::max)();
    float br_vf = -(std::numeric_limits<float>::max)();

    float u, v;
    for (int y = 0; y < src_size.height; ++y)
    {
        for (int x = 0; x < src_size.width; ++x)
        {
            mapForward(projector,static_cast<float>(x), static_cast<float>(y), u, v);
            tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
            br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


void StitcherCylindrical::mapForward(Projector projector, float x, float y, float& u, float& v)
{
    float x_ = projector.r_kinv[0] * x + projector.r_kinv[1] * y + projector.r_kinv[2];
    float y_ = projector.r_kinv[3] * x + projector.r_kinv[4] * y + projector.r_kinv[5];
    float z_ = projector.r_kinv[6] * x + projector.r_kinv[7] * y + projector.r_kinv[8];

    u = projector.scale * atan2f(x_, z_);
    v = projector.scale * y_ / sqrtf(x_ * x_ + z_ * z_);
}


inline
void StitcherCylindrical::mapBackward(Projector projector, float u, float v, float& x, float& y)
{
    u /= projector.scale;
    v /= projector.scale;

    float x_ = sinf(u);
    float y_ = v;
    float z_ = cosf(u);

    float z;
    x = projector.k_rinv[0] * x_ + projector.k_rinv[1] * y_ + projector.k_rinv[2] * z_;
    y = projector.k_rinv[3] * x_ + projector.k_rinv[4] * y_ + projector.k_rinv[5] * z_;
    z = projector.k_rinv[6] * x_ + projector.k_rinv[7] * y_ + projector.k_rinv[8] * z_;

    if (z > 0) { x /= z; y /= z; }
    else x = y = -1;
}
