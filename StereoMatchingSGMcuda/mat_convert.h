#pragma once
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>

cv::Mat slMat2cvMat(sl::Mat& input);
//pcl::PointCloud<pcl::PointXYZ>::Ptr Mat2PointXYZ(cv::Mat cvMat_Point_Cloud);