#include "StitcherSphere.h"

StitcherSphere::StitcherSphere()
{
   warpType = 1;
}

StitcherSphere::~StitcherSphere()
{
}

StitcherSphere::StitcherSphere(PanoParam pparam)
{
	panoparam.fdwork_scale = pparam.fdwork_scale;
	panoparam.seam_work_aspect = pparam.seam_work_aspect;
	panoparam.match_conf = pparam.match_conf;
	panoparam.conf_thresh = pparam.conf_thresh;
	panoparam.range_width = pparam.range_width;
	panoparam.resScale = pparam.resScale;
    warpType = 1;
}

void StitcherSphere::detectResultRoi(cv::Size src_size, Projector projector, cv::Point& dst_tl, cv::Point& dst_br)
{
    detectResultRoiByBorder(src_size, projector,dst_tl, dst_br);

    float tl_uf = static_cast<float>(dst_tl.x);
    float tl_vf = static_cast<float>(dst_tl.y);
    float br_uf = static_cast<float>(dst_br.x);
    float br_vf = static_cast<float>(dst_br.y);

    float x = projector.rinv[1];
    float y = projector.rinv[4];
    float z = projector.rinv[7];
    if (y > 0.f)
    {
        float x_ = (projector.k[0] * x + projector.k[1] * y) / z + projector.k[2];
        float y_ = projector.k[4] * y / z + projector.k[5];
        if (x_ > 0.f && x_ < src_size.width && y_ > 0.f && y_ < src_size.height)
        {
            tl_uf = std::min(tl_uf, 0.f); tl_vf = std::min(tl_vf, static_cast<float>(CV_PI * projector.scale));
            br_uf = std::max(br_uf, 0.f); br_vf = std::max(br_vf, static_cast<float>(CV_PI * projector.scale));
        }
    }

    x = projector.rinv[1];
    y = -projector.rinv[4];
    z = projector.rinv[7];
    if (y > 0.f)
    {
        float x_ = (projector.k[0] * x + projector.k[1] * y) / z + projector.k[2];
        float y_ = projector.k[4] * y / z + projector.k[5];
        if (x_ > 0.f && x_ < src_size.width && y_ > 0.f && y_ < src_size.height)
        {
            tl_uf = std::min(tl_uf, 0.f); tl_vf = std::min(tl_vf, static_cast<float>(0));
            br_uf = std::max(br_uf, 0.f); br_vf = std::max(br_vf, static_cast<float>(0));
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}

void StitcherSphere::detectResultRoiByBorder(cv::Size src_size, Projector projector, cv::Point& dst_tl, cv::Point& dst_br)
{
    float tl_uf = (std::numeric_limits<float>::max)();
    float tl_vf = (std::numeric_limits<float>::max)();
    float br_uf = -(std::numeric_limits<float>::max)();
    float br_vf = -(std::numeric_limits<float>::max)();

    float u, v;
    for (float x = 0; x < src_size.width; ++x)
    {
        mapForward(projector,static_cast<float>(x), 0, u, v);
        tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
        br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);

        mapForward(projector, static_cast<float>(x), static_cast<float>(src_size.height - 1), u, v);
        tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
        br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);
    }
    for (int y = 0; y < src_size.height; ++y)
    {
        mapForward(projector, 0, static_cast<float>(y), u, v);
        tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
        br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);

        mapForward(projector, static_cast<float>(src_size.width - 1), static_cast<float>(y), u, v);
        tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
        br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}

void StitcherSphere::mapForward(Projector projector, float x, float y, float& u, float& v)
{
    float x_ = projector.r_kinv[0] * x + projector.r_kinv[1] * y + projector.r_kinv[2];
    float y_ = projector.r_kinv[3] * x + projector.r_kinv[4] * y + projector.r_kinv[5];
    float z_ = projector.r_kinv[6] * x + projector.r_kinv[7] * y + projector.r_kinv[8];

    u = projector.scale * atan2f(x_, z_);
    float w = y_ / sqrtf(x_ * x_ + y_ * y_ + z_ * z_);
    v = projector.scale * (static_cast<float>(CV_PI) - acosf(w == w ? w : 0));
}

void StitcherSphere::mapBackward(Projector projector, float u, float v, float& x, float& y)
{
    u /= projector.scale;
    v /= projector.scale;

    float sinv = sinf(static_cast<float>(CV_PI) - v);
    float x_ = sinv * sinf(u);
    float y_ = cosf(static_cast<float>(CV_PI) - v);
    float z_ = sinv * cosf(u);

    float z;
    x = projector.k_rinv[0] * x_ + projector.k_rinv[1] * y_ + projector.k_rinv[2] * z_;
    y = projector.k_rinv[3] * x_ + projector.k_rinv[4] * y_ + projector.k_rinv[5] * z_;
    z = projector.k_rinv[6] * x_ + projector.k_rinv[7] * y_ + projector.k_rinv[8] * z_;

    if (z > 0) { x /= z; y /= z; }
    else x = y = -1;
}