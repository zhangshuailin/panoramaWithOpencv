#pragma once
#define ENABLE_LOG
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl


#define OK 0
#define ERROR_INSUFFICIENT_VALID_IMAGES -1
#define ERROR_READ_IMAGE_ERROR -2
#define HOMOGRAPHY_ESTIMATION_FAILED -3
#define HOMOGRAPHY_BAOPT_FAILED -4 //这个可以当做一个warning而非错误用来提醒潜在的拼接效果
#define ERROR_OTHERS -5



