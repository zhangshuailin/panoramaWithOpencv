#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
typedef struct paronomaParameter
{
	double fdwork_scale;   //��ȡ������׼��ʱ������ͼ��ĳߴ����ű���
	double seam_work_aspect; //����seam������fdwork_scale���ŵĻ�����������seam_work_aspect,��С����ƴ�Ӻۼ�
	float match_conf; //ɸѡƥ��������㣬����ƥ�����������������������Խ��Խ�п��ܿɿ���ֵԽСԽ��
	float conf_thresh;//������ƥ�����Ŷȣ�ֵԽ��Խ�ɿ��������ܻ��м��ٵ�������
	int range_width;
	float resScale;
	paronomaParameter() : fdwork_scale(0.2), seam_work_aspect(1.0), match_conf(0.2f), conf_thresh(0.8f), range_width(6), resScale(0.8f){} // ���캯��
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