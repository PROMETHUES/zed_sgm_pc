#pragma once
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/ximgproc.hpp>
//#include <opencv2/viz.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <sl/Camera.hpp>

#include <libsgm.h>
#include "mat_convert.h"
 
const int disp_size = 128;
const int input_depth = 8;
const int output_depth = 8;
const double lambda = 8000.0;
const double sigma = 1.0;

template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}

struct device_buffer {
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

struct left_cam_params_VGA {
	const float fx = 350.02;
	const float fy = 350.02;
	const float cx = 331.512;
	const float cy = 189.729;
	const float k1 = -0.175923;
	const float k2 = 0.028609;
	const float p1 = 0.;
	const float p2 = 0.;
};

struct right_cam_params_VGA {
	const float fx = 350.497;
	const float fy = 350.497;
	const float cx = 323.708;
	const float cy = 195.278;
	const float k1 = -0.177537;
	const float k2 = 0.029991;
	const float p1 = 0.;
	const float p2 = 0.;
};

struct stereo_cam_params {
	const float Baseline = 120.;
	const float CV_2K = 0.00466962;
	const float CV_FHD = 0.00466962;
	const float CV_HD = 0.00466962;
	const float CV_VGA = 0.00466962;
	const float RX_2K = -0.00662491;
	const float RX_FHD = -0.00662491;
	const float RX_HD = -0.00662491;
	const float RX_VGA = -0.00662491;
	const float RZ_2K = 0.000576973;
	const float RZ_FHD = 0.000576973;
	const float RZ_HD = 0.000576973;
	const float RZ_VGA = 0.000576973;
};

//void init_Q_Matrix(cv::Mat &Q) {
//	struct left_cam_params_VGA left_params;
//	struct right_cam_params_VGA right_params;
//	struct stereo_cam_params stereo_params;
//	float q_array[4][4] = { 
//		{1., 0., 0., -left_params.cx},
//		{0., 1., 0., -left_params.cy},
//		{0., 0., 0., left_params.fx},
//		{0., 0., -1. / stereo_params.Baseline, (left_params.cx - right_params.cx) / stereo_params.Baseline}
//	};
//	Q = cv::Mat(4, 4, CV_32F, q_array);
//	std::cout << "Q= " << Q << std::endl;
//}