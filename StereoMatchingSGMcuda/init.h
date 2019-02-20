#pragma once
#include "zed_demo.h"

boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr p_cloud);
void qm_init(cv::Mat &Q);