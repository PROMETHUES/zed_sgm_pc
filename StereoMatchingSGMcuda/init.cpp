#include "init.h"

boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr p_cloud)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> point_cloud_viewer(new pcl::visualization::PCLVisualizer("3D_Viewer"));
	point_cloud_viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_ColorHandler(p_cloud);
	point_cloud_viewer->addPointCloud<pcl::PointXYZRGB>(p_cloud, rgb_ColorHandler, "3D_Viewer");
	point_cloud_viewer->addCoordinateSystem(1.0);
	point_cloud_viewer->initCameraParameters();
	return(point_cloud_viewer);
}

void qm_init(cv::Mat &Q) {
	struct left_cam_params_VGA left_Km;
	struct right_cam_params_VGA right_Km;
	struct stereo_cam_params stereo_Km;
	float q_array[4][4] = {
		{ 1., 0., 0., -left_Km.cx },
		{ 0., 1., 0., -left_Km.cy },
		{ 0., 0., 0., left_Km.fx },
		{ 0., 0., -1. / stereo_Km.Baseline, (left_Km.cx - right_Km.cx) / stereo_Km.Baseline }
	};
	cv::Mat(4, 4, CV_32F, q_array).copyTo(Q);
	//std::cout << "Q_Matrix= " << Q << std::endl;
}