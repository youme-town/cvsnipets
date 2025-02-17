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
	constexpr double GAAMA[3] = { 2.2, 2.2, 2.2 };  //!< The gamma value for each color (B, G, R)
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


	void GenerateColorPatterns(std::vector<cv::Mat>& ideal_patterns, std::vector<cv::Mat>& projection_patterns, const double gamma[3]) {
		std::vector<cv::Mat> projection_images; 
		std::vector<cv::Mat> ideal_images;

		cv::Mat temp_img = cv::Mat::zeros(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC3);

		// calculate inverse gamma LUT
		cv::Mat lut = cv::Mat(1, 256, CV_8UC3);      //!< ルックアップテーブル用配列
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

		for (const auto& proj_img : projection_iamges) {
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
	 * @brief Calculate Yoshida's RGB compensation
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
		const std::string& input_folder_path,
		const std::string& output_folder_path,
		const cv::Mat& pro_to_cam,
		const cv::Mat& cam_to_pro,
		const cv::Mat& c2p_map,
		const cv::Mat& p2c_map) {

	#ifdef DEBUG
		int cnt = 0;
	#endif // DEBUG

		namespace us = useful_functions;
		namespace rf = related_functions;

		std::cout << "Read color mixing matrix file" << std::endl;
		// 色補正行列の読み込み
		cv::Mat color_mix_mat;
		std::string cmm_path = std::string(OUTPUT_PATH) + std::string(COLOR_MIXING_MATRIX_NAME) + ".mat";
		if (!us::LoadMatBinary(cmm_path, color_mix_mat)) {
			std::cerr << "Failed to read the color mixing matrix" << std::endl;
			return false;
		}
		std::cout << "Reading done" << std::endl;

		std::cout << "Read images" << std::endl;

		// 画像の読み込み
		std::vector<us::ImageData> images;
		images = us::LoadImagesAndNamesFromFolder(input_folder_path);

		std::cout << "Reading done" << std::endl;


	#ifdef DEBUG

		std::cout << "Images size: " << images.size() << std::endl;

		for (auto& im : images) {
			std::cout << "Image name: " << im.filename << std::endl;
			std::string filename = std::string(OUTPUT_PATH) + std::string(DEBUG_IMAGE_PATH) + im.filename_noext + "_debug.png";
			us::ImgWrite(filename, im.image);
		}

	#endif // DEBUG

		// vector<cv::Mat>に格納
		std::vector<cv::Mat> input_images;

		// ウィンドウサイズの計算
		cv::Rect pro_roi;
		if (!us::CalcCenterROI(cv::Size(EXP_IMAGE_WIDTH, EXP_IMAGE_HEIGHT), pro_roi, PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT)) {
			std::cerr << "Calculation ROI failed" << std::endl;
			return false;
		}


		// フルスクリーン画像に変えてからベクトルに格納
		for (auto& im : images) {
			cv::Mat tmp_resize, tmp_pro;
			// リサイズ
			cv::resize(im.image.clone(), tmp_resize, cv::Size(EXP_IMAGE_WIDTH, EXP_IMAGE_HEIGHT));

			// ウィンドウサイズに変換
			if (!us::MakeWindowSizeMat(tmp_resize, tmp_pro, pro_roi, PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT)) {
				std::cerr << "MakeWindowSizeMat failed" << std::endl;
				return false;
			}

			input_images.push_back(tmp_pro.clone());


		}

		std::cout << "Warping images" << std::endl;


		std::vector<cv::Mat> warped_images;
		for (auto& image : input_images) {
			cv::Mat tmp_dst;
			tmp_dst = rf::YourWarpImageFunctionToCameraCoordinates(image, p2c_map, c2p_map);
			warped_images.push_back(tmp_dst.clone());
		}

	#ifdef DEBUG
		cnt = 0;
		for (auto& im : warped_images) {
			std::string zeronum = us::GetZeroPaddingNumberString(cnt++, 4);
			std::string filename = std::string(OUTPUT_PATH) + std::string(DEBUG_IMAGE_PATH) + "_debug_warped" + zeronum + ".png";
			us::ImgWrite(filename, im);
		}
	#endif // DEBUG

		std::cout << "Warping done" << std::endl;

		std::cout << "Calc Yoshida RGB" << std::endl;

		// Calculate inverse gamma correction array
		cv::Mat lut = cv::Mat(1, 256, CV_8UC3);
		us::CalcInvGammaLUT(lut, GAMMA);

		std::vector<cv::Mat> yoshida_images;

		for (auto& image : warped_images) {
			cv::Mat tmp_dst;
			CalcYoshidaColor(image, tmp_dst, color_mix_mat);
			yoshida_images.push_back(tmp_dst.clone());
		}

	#ifdef DEBUG
		cnt = 0;
		for (auto& im : yoshida_images) {
			std::string zeronum = us::GetZeroPaddingNumberString(cnt++, 4);
			std::string filename = std::string(OUTPUT_PATH) + std::string(DEBUG_IMAGE_PATH) + "_debug_calc" + zeronum + ".png";
			us::ImgWrite(filename, im);
		}
	#endif // DEBUG

		std::cout << "Calc Yoshida RGB done" << std::endl;

		std::cout << "Warping images" << std::endl;

		std::vector<cv::Mat> projection_images;

		for (auto& image : yoshida_images) {
			cv::Mat tmp_dst;
			tmp_dst = rf::YourWarpImageFunctionToProjectorCoordinates(image, p2c_map, c2p_map);

			// 投影画像領域以外を黒で埋める
			// 黒い画像を作成
			cv::Mat tmp_pro = cv::Mat::zeros(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC3);
			// 必要な部分に代入
			cv::Mat roi_tmp_pro(tmp_pro, pro_roi);
			cv::Mat roi_tmp_dst(tmp_dst, pro_roi);
			roi_tmp_dst.copyTo(roi_tmp_pro);
			projection_images.push_back(tmp_pro.clone());
		}
		std::cout << "Warping done" << std::endl;

	#ifdef DEBUG
		cnt = 0;
		for (auto& im : warped_images) {
			std::string zeronum = us::GetZeroPaddingNumberString(cnt++, 4);
			std::string filename = std::string(OUTPUT_PATH) + std::string(DEBUG_IMAGE_PATH) + "_debug_calc_warped_" + zeronum + ".png";
			us::ImgWrite(filename, im);
		}
	#endif // DEBUG

		std::cout << "Save images" << std::endl;

		// projection_imagesの画像をimagesに戻す
		for (size_t i = 0; i < images.size(); ++i) {
			// ガンマ補正
			cv::LUT(projection_images[i], lut, projection_images[i]);
			images[i].image = projection_images[i];
		}

		// prefixを指定して画像を保存
		us::SaveProcessedImages(images, output_folder_path, "Yoshida_compensated_");

		std::cout << "Save done" << std::endl;

		return true;

	}


}  // namespace