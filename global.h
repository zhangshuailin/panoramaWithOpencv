#pragma once
#define ENABLE_LOG
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl


#define OK 0
#define ERROR_INSUFFICIENT_VALID_IMAGES -1
#define ERROR_READ_IMAGE_ERROR -2
#define HOMOGRAPHY_ESTIMATION_FAILED -3
#define HOMOGRAPHY_BAOPT_FAILED -4 //������Ե���һ��warning���Ǵ�����������Ǳ�ڵ�ƴ��Ч��
#define ERROR_OTHERS -5



