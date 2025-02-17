/**
 * @file YoshidaCompensation.cpp
 * @brief RGB compensation using Yoshida's method
 * @author Yoshiaki Maeda
 * @date 2025/01/21
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include "YoshidaColorMixingMatrix.hpp"

/** 
 * @namespace your_functions
 * @brief Please implement these functions acconrding to your environment
 */
namespace your_functions {

	bool InitCamera() {
		// return true if the camera is successfully initialized
		return true; 
	};
	void TerminateCamera() {};

	cv::Mat Capture() {
		return cv::Mat();
	};

	void ToCameraCoordinates(const cv::Mat& src, cv::Mat& dst) {
		dst = src.clone();
	};

	void ToProjectorCoordinates(const cv::Mat& src, cv::Mat& dst) {
		dst = src.clone();
	};

}

namespace /* To avoid name collision*/ {

	/** @name Constants Group
	 *  @brief Constants used in this program
	 *  @note Please modify these values according to your environment
	 */
	/* @{ */
	constexpr int NUM_COLOR_GRADATION = 3;  //!< The number of gradations for each color (B, G, R)
	constexpr int CAMERA_WAIT_MS = 1000;  //!< The waiting time for the camera to capture an image [ms]
	constexpr int PRO_WINDOW_X = 0;  //!< The x-coordinate of the projector window
	constexpr int PRO_WINDOW_WIDTH = 1920;  //!< The width of the projector window
	constexpr int PRO_WINDOW_HEIGHT = 1080;  //!< The height of the projector window
	constexpr int ROI_IMAGE_WIDTH = 512;  //!< The width of the ROI
	constexpr int ROI_IMAGE_HEIGHT = 512;  //!< The height of the ROI
	constexpr double GAMMA[3] = { 2.2, 2.2, 2.2 };  //!< The gamma value for each color (B, G, R)
	/* @} */

	/** @name File Path Group
	 *  @brief File paths used in this program
	 *  @note Please modify these values according to your environment
	 */
	/* @{ */
	const std::string CURRENT_PATH = "./";  //!< The current folder path
	const std::string OUTPUT_PATH = CURRENT_PATH + "output/";  //!< The output folder path
	const std::string INPUT_PATH = CURRENT_PATH + "input/";  //!< The input folder path
	const std::string CAPTURED_PATTERN_PATH = OUTPUT_PATH + "captured_pattern/";  //!< The folder path for captured patterns
	const std::string COLOR_MIXING_MATRIX_PATH = OUTPUT_PATH + "color_mixing_matrix.cmm";  //!< The name of the color mixing matrix file
	/* @} */

	// Look Up Tableの計算関数
	void CalcInverseGammaLUT(cv::Mat& lut, const double gamma[3]) {
		lut = cv::Mat(1, 256, CV_8UC3);
		for (int i = 0; i < lut.cols; i++) {
			//ガンマ補正式
			lut.at<uchar>(0, i * 3 + 0) = pow(i / 255.0, 1 / gamma[0]) * 255.0;
			lut.at<uchar>(0, i * 3 + 1) = pow(i / 255.0, 1 / gamma[1]) * 255.0;
			lut.at<uchar>(0, i * 3 + 2) = pow(i / 255.0, 1 / gamma[2]) * 255.0;
		}
	}

	/**
	 * @brief Calculate cenntered cv::Rect of source size cv::Mat
	 * @param [in] src_size Source image size
	 * @param [out] dst Destination cv::Rect
	 * @param [in] window_width Window width
	 * @param [in] window_height Window height
	 * @return Returns true on success, false on failure
	 */
	bool CalcCenterROI(
		const cv::Size& src_size,
		cv::Rect& dst,
		const int window_width,
		const int window_height) {
		if (src_size.width > window_width || src_size.height > window_height) {
			std::cerr << "Invalid size" << std::endl;
			return false;
		}

		const int dst_x = (window_width - src_size.width) / 2;
		const int dst_y = (window_height - src_size.height) / 2;

		dst = cv::Rect(dst_x, dst_y, src_size.width, src_size.height);

		return true;
	}

	/**
	 * @brief Embeds an image into a specified Rect.
	 * @param [in] src Source image
	 * @param [out] dst Destination image
	 * @param [in] dst_roi Destination region of interest
	 * @param [in] window_width Window width
	 * @param [in] window_height Window height
	 * @return Returns true on success, false on failure
	 */
	bool MakeWindowSizeMat(const cv::Mat& src, cv::Mat& dst, const cv::Rect& dst_roi,
		const int window_width, const int window_height)
	{
		if (dst_roi.x < 0 || dst_roi.y < 0 || dst_roi.width <= 0 || dst_roi.height <= 0) {
			std::cerr << "Invalid roi" << std::endl;
			return false;
		}
		if (src.cols != dst_roi.width || src.rows != dst_roi.height) {
			std::cerr << "Invalid size" << std::endl;
			return false;
		}

		dst = cv::Mat::zeros(cv::Size(window_width, window_height), src.type());
		cv::Mat roi = dst(dst_roi);
		src.copyTo(roi);

		return true;
	}

	/**
	 * @brief Generate color patterns for RGB compensation
	 * @param [out] ideal_patterns Ideal color patterns
	 * @param [out] projection_patterns Projection color patterns
	 * @param [in] gamma Gamma value for each color (B, G, R)
	 */
	void GenerateColorPatterns(std::vector<cv::Mat>& ideal_patterns, std::vector<cv::Mat>& projection_patterns, const double gamma[3]) {
		std::vector<cv::Mat> projection_images; 
		std::vector<cv::Mat> ideal_images;

		cv::Mat temp_img = cv::Mat::zeros(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC3);

		// calculate inverse gamma LUT
		cv::Mat lut = cv::Mat(1, 256, CV_8UC3);      //!< matrix for look up table
		::CalcInverseGammaLUT(lut, gamma);

		int color_grad_cnt = 0;
		for (int i = 0; i < NUM_COLOR_GRADATION; i++) {
			for (int j = 0; j < NUM_COLOR_GRADATION; j++) {
				for (int k = 0; k < NUM_COLOR_GRADATION; k++) {
					std::ostringstream oss;
					oss << std::setfill('0') << std::setw(4) << color_grad_cnt++;

					// Generate the projection image
					temp_img.forEach<cv::Vec3b>([&](cv::Vec3b& p, const int* pos) {
						p[0] = 255 * i / (NUM_COLOR_GRADATION - 1);
						p[1] = 255 * j / (NUM_COLOR_GRADATION - 1);
						p[2] = 255 * k / (NUM_COLOR_GRADATION - 1);
						});

					// Save ideal pattern for calculation
					ideal_images.push_back(temp_img.clone());

					// apply inverse gamma correction
					cv::Mat temp_img_gamma = cv::Mat::zeros(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC3);
					cv::LUT(temp_img, lut, temp_img_gamma);
					
					// Save the projection pattern for projection
					projection_images.push_back(temp_img_gamma.clone());
				}
			}
		}
	}

	/**
	 * @brief Capture color patterns
	 * @param [in] projection_iamges Projection color patterns
	 * @return Captured color patterns
	 */
	std::vector<cv::Mat> CaptureColorPatterns(const std::vector<cv::Mat>& projection_iamges) {

		// Make the window for the projection
		cv::namedWindow("projection_win", cv::WINDOW_NORMAL);
		cv::moveWindow("projection_win", PRO_WINDOW_X, 0);
		cv::setWindowProperty("projection_win", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

		// ----------------------
		// Initialize the camera
		// ----------------------
		if (your_functions::InitCamera()) {
			std::cerr << "Failed to initialize the camera" << std::endl;
            return std::vector<cv::Mat>();
		}

		// -------------
		// Capture loop
		// -------------
		std::cout << "Capture start" << std::endl;

		std::vector<cv::Mat> captured_patterns;

		int capture_count = 0;
		for (const auto& proj_img : projection_iamges) {
			std::cout << "Capture count: " << capture_count++ << std::endl;

			cv::imshow("projection_win", proj_img);
			cv::waitKey(CAMERA_WAIT_MS);

			cv::Mat captured_image = your_functions::Capture();
			captured_patterns.push_back(captured_image.clone());
		}

		// ---------------------
		// Terminate the camera
		// ---------------------

		your_functions::TerminateCamera();

		std::cout << "Capture complete" << std::endl;

		return captured_patterns;

	}

	/**
	 * @brief Calculate Yoshida's color mixing matrix
	 *
	 * @param [in] ideal_patterns Color patterns which are ideal results
	 * @param [in] captured_patterns Color patterns which are captured by the camera
	 * @param [out] color_mixing_matrix Color mixing matrix for every pixel
	 * @return boolean value (true: success, false: failure)
	 */
	bool CalcYoshidaColorMixingMatrix(const std::vector<cv::Mat>& ideal_patterns, const std::vector<cv::Mat>& captured_patterns, cv::Mat& color_mixing_matrix) {

		if (ideal_patterns.empty() || captured_patterns.empty()) {
			std::cerr << "Ideal patterns or captured patterns are empty" << std::endl;
			return false;
		}

		if (ideal_patterns.size() != captured_patterns.size()) {
			std::cerr << "The number of ideal patterns and captured patterns are different" << std::endl;
			return false;
		}

		std::cout << "Start calculation of color mixing matrix" << std::endl;

		const int camera_width = captured_patterns[0].cols;
		const int camera_height = captured_patterns[0].rows;

		// Number of unknown value for each pixels）
		const int num_channel_YoshidaCMM = 12;
		
		const int num_color_patterns = ideal_patterns.size();

		cv::Mat color_mix_mat = cv::Mat::zeros(camera_height, camera_width, CV_64FC(num_channel_YoshidaCMM));

		// Parallel computation using forEach method
		color_mix_mat.forEach<cv::Vec<double, num_channel_YoshidaCMM>>([&](cv::Vec<double, num_channel_YoshidaCMM>& p, const int* pos)-> void {
			cv::Mat captured_pixel_values = cv::Mat::zeros(num_color_patterns * 3, 1, CV_64FC1);
			cv::Mat ideal_pixel_values = cv::Mat::zeros(num_color_patterns * 3, 1, CV_64FC1);

			// Load RGB values for matrix computation
			for (int i = 0; i < num_color_patterns; i++) {
				captured_pixel_values.at<double>(3 * i + 0, 0) = (double)captured_patterns[i].at<cv::Vec3b>(pos[0], pos[1])[0];
				captured_pixel_values.at<double>(3 * i + 1, 0) = (double)captured_patterns[i].at<cv::Vec3b>(pos[0], pos[1])[1];
				captured_pixel_values.at<double>(3 * i + 2, 0) = (double)captured_patterns[i].at<cv::Vec3b>(pos[0], pos[1])[2];

				ideal_pixel_values.at<double>(3 * i + 0, 0) = (double)ideal_patterns[i].at<cv::Vec3b>(pos[0], pos[1])[0];
				ideal_pixel_values.at<double>(3 * i + 1, 0) = (double)ideal_patterns[i].at<cv::Vec3b>(pos[0], pos[1])[1];
				ideal_pixel_values.at<double>(3 * i + 2, 0) = (double)ideal_patterns[i].at<cv::Vec3b>(pos[0], pos[1])[2];
			}
			// Color caluculation class instance
			YoshidaColorMixingMatrix yoshida;

			// Matrix computation	
			p = yoshida.ComputeMatrix1(captured_pixel_values.clone(), ideal_pixel_values.clone());

			});

		std::cout << "Finish calculation" << std::endl;

		color_mixing_matrix = std::move(color_mix_mat);

		return true;
	}


	/**
	 * @brief Calculate Yoshida's RGB compensation using the color mixing matrix
	 *
	 * @details This function calculates the RGB compensation using Yoshida's method,
	 * not including the pixel warping process and inverse gamma correction.
	 * @param [in]  src source image (camera coordinates)
	 * @param [out] dst destination image (camera coordinates)
	 * @param [in]  color_mix_mat color mixing matrix
	 */
	void CalcYoshidaColor(const cv::Mat& src, cv::Mat& dst, const cv::Mat& color_mix_mat) {
		std::mutex mtx;
		dst = cv::Mat(src.size(), src.type());

		// おなじみのforEach(pos[0]はrow index,pos[1]はcolumn index)
		src.forEach<cv::Vec3b>([&](cv::Vec3b& p, const int* pos)->void {
			cv::Matx<double, 3, 4> tmp_cmm;
			cv::Mat tmp_output_RGB_camera = cv::Mat::zeros(4, 1, CV_64FC1);
			cv::Mat tmp_input_RGB_projector = cv::Mat::zeros(3, 1, CV_64FC1);

			// CMMに色補正行列を代入
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 4; l++) {
					tmp_cmm(k, l) = color_mix_mat.at<cv::Vec<double, 12>>(pos[0], pos[1])[4 * k + l];
				}
			}

			// カメラの目標RGB出力を代入して計算
			tmp_output_RGB_camera.at<double>(0, 0) = p[0];
			tmp_output_RGB_camera.at<double>(1, 0) = p[1];
			tmp_output_RGB_camera.at<double>(2, 0) = p[2];
			tmp_output_RGB_camera.at<double>(3, 0) = 1.0;

			// プロジェクタに入力すべきRGBが求まる
			tmp_input_RGB_projector = tmp_cmm * tmp_output_RGB_camera;


			std::lock_guard<std::mutex> lock(mtx);
			{
				// プロジェクタに入力すべきRGBを画像に代入(ガンマ補正を考慮していない)
				for (int i = 0; i < 3/*Number of Channels*/; ++i) {
					dst.at<cv::Vec3b>(pos[0], pos[1])[i]
						= cv::saturate_cast<uchar>(tmp_input_RGB_projector.at<double>(i, 0));
				}
			}

			});


	}


	bool CalcYoshidaRGBCompensationImage(
		const std::vector<cv::Mat> &input_images,
		std::vector<cv::Mat>& output_images,
		const cv::Mat &color_mix_mat,
		const double gamma[3]=GAMMA) {

		if (input_images.empty()) {
			std::cerr << "Input images are empty" << std::endl;
			return false;
		}


		// フルスクリーン画像に変えてからベクトルに格納
		std::vector<cv::Mat> fullscreen_images;
		std::vector<cv::Rect> input_images_roi;
		for (auto& im : input_images) {
			// ウィンドウサイズの計算
			cv::Rect pro_roi;
			if (!::CalcCenterROI(im.size(), pro_roi, PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT)) {
				std::cerr << "Calculation Center ROI failed" << std::endl;
				return false;
			}
			// ROIを保存
			input_images_roi.push_back(pro_roi);

			cv::Mat tmp_resize, tmp_pro;
			// リサイズ
			cv::resize(im.clone(), tmp_resize, im.size());

			// ウィンドウサイズに変換
			if (!::MakeWindowSizeMat(tmp_resize, tmp_pro, pro_roi, PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT)) {
				std::cerr << "MakeWindowSizeMat failed" << std::endl;
				return false;
			}

			fullscreen_images.push_back(tmp_pro.clone());

		}

		std::cout << "Warping images" << std::endl;


		std::vector<cv::Mat> warped_images;
		for (auto& image : fullscreen_images) {
			cv::Mat tmp_dst;
			your_functions::ToCameraCoordinates(image, tmp_dst);
			warped_images.push_back(tmp_dst.clone());
		}

		std::cout << "Warping done" << std::endl;

		std::cout << "Calc Yoshida RGB" << std::endl;

		// Calculate inverse gamma correction array
		cv::Mat lut = cv::Mat(1, 256, CV_8UC3);
		::CalcInverseGammaLUT(lut, gamma);

		std::vector<cv::Mat> yoshida_images;

		for (auto& image : warped_images) {
			cv::Mat tmp_dst;
			::CalcYoshidaColor(image, tmp_dst, color_mix_mat);
			yoshida_images.push_back(tmp_dst.clone());
		}

		std::cout << "Calculating Yoshida RGB done" << std::endl;

		std::cout << "Warping images" << std::endl;

		for (int i = 0; i < yoshida_images.size(); i++) {
			cv::Mat& image = yoshida_images[i];
			cv::Mat tmp_dst;
			your_functions::ToProjectorCoordinates(image, tmp_dst);

			// 投影画像領域以外を黒で埋める
			cv::Mat tmp_pro = cv::Mat::zeros(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC3);
			// 必要な部分に代入
			cv::Rect pro_roi = input_images_roi[i];
			cv::Mat roi_tmp_pro(tmp_pro, pro_roi);
			cv::Mat roi_tmp_dst(tmp_dst, pro_roi);
			roi_tmp_dst.copyTo(roi_tmp_pro);
			output_images.push_back(tmp_pro.clone());
		}

		std::cout << "Warping done" << std::endl;

		std::cout << "Inverse gamma correction" << std::endl;

		for (auto& image : output_images) {
			cv::Mat tmp_dst;
			cv::LUT(image, lut, tmp_dst);
			image = tmp_dst.clone();
		}

		std::cout << "Inverse gamma correction done" << std::endl;

		std::cout << "Calc Yoshida RGB done" << std::endl;

		return true;

	}


}  // namespace
