#include "init.h"

//#define BILATERAL_FILTER


int main(int argc, char* argv[])
{
	// Q matrix
	cv::Mat mQ;
	qm_init(mQ);
	std::cout << "mQ= " << mQ << std::endl;

	// Setting camera
	sl::Camera zed;
	sl::InitParameters initParameters;
	initParameters.camera_resolution = sl::RESOLUTION_VGA;
	sl::ERROR_CODE err = zed.open(initParameters);
	zed.setCameraSettings(sl::CAMERA_SETTINGS_EXPOSURE, 100, false);

	if (err != sl::SUCCESS)
	{
		std::cout << toString(err) << std::endl;
		zed.close();
		return 1;
	}

	const int width = static_cast<int>(zed.getResolution().width);
	const int height = static_cast<int>(zed.getResolution().height);
	const int input_bytes = input_depth * width * height / 8;
	const int output_bytes = output_depth * width * height / 8;

	sl::Mat d_zed_image_l(zed.getResolution(), sl::MAT_TYPE_8U_C1, sl::MEM_GPU);
	sl::Mat d_zed_image_r(zed.getResolution(), sl::MAT_TYPE_8U_C1, sl::MEM_GPU);

	//Initiate SGM
	sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);
	cv::Mat disparity(height, width, CV_8U);
	cv::Mat disparity_8u, disparity_color;
	device_buffer d_image_l(input_bytes), d_image_r(input_bytes), d_disparity(output_bytes);
	
	while (1) 
	{
		if (zed.grab() == sl::SUCCESS)
		{
#pragma omp parallel sections
			{
#pragma omp section
				zed.retrieveImage(d_zed_image_l, sl::VIEW_LEFT_GRAY, sl::MEM_GPU);
#pragma omp section
				zed.retrieveImage(d_zed_image_r, sl::VIEW_RIGHT_GRAY, sl::MEM_GPU);
			}
		}
		else continue;

#pragma omp parallel sections
		{
#pragma omp section
			cudaMemcpy2D(d_image_l.data, width, d_zed_image_l.getPtr<uchar>(sl::MEM_GPU), d_zed_image_l.getStep(sl::MEM_GPU), width, height, cudaMemcpyDeviceToDevice);
#pragma omp section
			cudaMemcpy2D(d_image_r.data, width, d_zed_image_r.getPtr<uchar>(sl::MEM_GPU), d_zed_image_r.getStep(sl::MEM_GPU), width, height, cudaMemcpyDeviceToDevice);
		}

		// Start SGM
		const auto t1 = std::chrono::system_clock::now();

		sgm.execute(d_image_l.data, d_image_r.data, d_disparity.data);
		cudaDeviceSynchronize();

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const double fps = 1e6 / duration;

		cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);

		// Original Disparity Display with color
		disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
		cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
		cv::putText(disparity_color, format_string("sgm execution time: %4.1f[msec] %4.1f[FPS]", 1e-3 * duration, fps),
			cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));
		cv::imshow("disparity", disparity_color);

		// Original GRAY scale
		//disparity.convertTo(disparity_8u, CV_8U, 255. / max_disp);
		//cv::imshow("disparity", disparity_8u);

		// Load rgb-image
		sl::Mat zed_leftview_slMat;
		cv::Mat zed_leftview_cvMat;
		
		zed.retrieveImage(zed_leftview_slMat, sl::VIEW_LEFT_GRAY, sl::MEM_CPU);
		zed_leftview_cvMat = slMat2cvMat(zed_leftview_slMat);
		cv::imshow("left_gray_view", zed_leftview_cvMat);

#ifdef BILATERAL_FILTER
		// CUDA BilateralFiter
		cv::Ptr<cv::cuda::DisparityBilateralFilter> disparity_bilateralfilter;
		cv::cuda::GpuMat zed_leftview_GpuMat, disparity_8u_GpuMat;
		cv::cuda::GpuMat b_filtered_disparity, b_filtered_disparity_color;
		zed_leftview_GpuMat.upload(zed_leftview_cvMat);
		disparity_8u_GpuMat.upload(disparity_8u);

		disparity_bilateralfilter = cv::cuda::createDisparityBilateralFilter();
		disparity_bilateralfilter->setMaxDiscThreshold(disp_size);
		disparity_bilateralfilter->apply(disparity_8u_GpuMat, zed_leftview_GpuMat, b_filtered_disparity, cv::cuda::Stream::Null());

		// CUDA Color Disparity
		cv::cuda::drawColorDisp(b_filtered_disparity, b_filtered_disparity_color, disp_size, cv::cuda::Stream());
		cv::Mat b_filtered_disparity_cvMat_color_1;
		b_filtered_disparity_color.download(b_filtered_disparity_cvMat_color_1);
		cv::imshow("filtered_color_GPU", b_filtered_disparity_cvMat_color_1);
#endif // BILATERAL_FILTER

		/* Create point cloud and fill it
		Coding according to:
		https://github.com/Chidanand/elevation-map/blob/66ebd456a891c803d625de01ebdad4a24148fc57/src/reproject_image_to_point_cloud/reproject_image_to_point_cloud.cpp
		*/

		// Get the interesting parameters from Q
		double Q03, Q13, Q23, Q32, Q33;
		Q03 = mQ.at<float>(0, 3);
		Q13 = mQ.at<float>(1, 3);
		Q23 = mQ.at<float>(2, 3);
		Q32 = mQ.at<float>(3, 2);
		Q33 = mQ.at<float>(3, 3);

#ifdef BILATERAL_FILTER
		// Download disparity from GPU
		cv::Mat b_filtered_disparity_cvMat;
		b_filtered_disparity.download(b_filtered_disparity_cvMat);
#endif // BILATERAL_FILTER

		// Create Matrix that will contain 3D coordinates of each pixel
		cv::Mat xyz_view_cvMat(disparity_8u.size(), CV_32FC3);

		// Re-project to 3D
#ifdef BILATERAL_FILTER
		cv::reprojectImageTo3D(b_filtered_disparity_cvMat, xyz_view_cvMat, mQ, false, CV_32F);
#else
		cv::reprojectImageTo3D(disparity_8u, xyz_view_cvMat, mQ, false, CV_32F);
#endif // BILATERAL_FILTER
		
		// Create point cloud and fill it with color
		std::cout << "Creating Point Cloud..." << std::endl;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

		double px, py, pz;
		uchar pr, pg, pb;

#pragma omp parallel
		{
#pragma omp for
			for (int i = 0; i < zed_leftview_cvMat.rows; i++) {
				//CUSTOM REPROJECT
				uchar* rgb_ptr = zed_leftview_cvMat.ptr<uchar>(i);

#ifdef BILATERAL_FILTER
				uchar* disp_ptr = b_filtered_disparity_cvMat.ptr<uchar>(i);
#else
				uchar* disp_ptr = disparity_8u.ptr<uchar>(i);
#endif // BILATERAL_FILTER	

				for (int j = 0; j < zed_leftview_cvMat.cols; j++) {
					//Get 3D coordinates
					uchar d = disp_ptr[j];
					if (d == 0) {
						continue; // Discard bad pixels
					}
					double pw = -1.0 * static_cast<double>(d) * Q32 + Q33;
					px = static_cast<double>(j) + Q03;
					py = static_cast<double>(i) + Q13;
					pz = Q23;

					px = px / pw;
					py = py / pw;
					pz = pz / pw;

					//Get RGB info
					pb = rgb_ptr[j];
					pg = rgb_ptr[j + 1];
					pr = rgb_ptr[j + 2];

					//Insert info into point cloud structure
					pcl::PointXYZRGB point;
					point.x = -px;
					point.y = -py;
					point.z = pz;

					uint32_t rgb = (
						static_cast<uint32_t>(pr) << 16 |
						static_cast<uint32_t>(pg) << 8 |
						static_cast<uint32_t>(pb));
					point.rgb = *reinterpret_cast<float*>(&rgb);
#pragma omp atomic
					point_cloud_ptr->points.push_back(point);
					//pcloud.push_back(point);
				}
			}
		}
		point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
		point_cloud_ptr->height = 1;

		//Create visualizer
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
		viewer = createVisualizer(point_cloud_ptr);

		//Main loop
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(10000));
		}
	}
	zed.close();
	return 0;
}
