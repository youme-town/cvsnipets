/**
 * @file YoshidaCompensation.cpp
 * @brief RGB color compensation using Yoshida's method
 * @author Yoshiaki Maeda
 * @date 2025/01/21
 */
#include <iostream>
#include <filesystem>
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

	bool Capture(cv::Mat& dst) {
		// Capture an image from the camera and store it in dst
		return true;
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
	 * @brief Creates a full-screen window
	 * @param [in] window_name Name of the window
	 * @param [in] move_window_width Width to move the window
	 */
	void MakeFullScreenWindow(const std::string& window_name, const int move_window_width) {
		//投影するウィンドウの作成
		cv::namedWindow(window_name, cv::WINDOW_NORMAL);
		cv::moveWindow(window_name, move_window_width, 0);
		cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	}

	/**
	 * @brief Checks the existence of a folder and creates it if it doesn't exist
	 * @param [in] folder_path Path to the folder
	 * @return Returns true if the folder was created or already exists, false otherwise
	 */
	bool CheckAndCreateFolder(const std::string& folder_path)
	{
		if (!std::filesystem::exists(folder_path)) {
			std::filesystem::create_directories(std::filesystem::path(folder_path));
			return true;
		}
		return false;
	}

	/**
	 * @brief Splits a full file path into folder path and file name
	 * @param [in] fullPath Full file path
	 * @param [out] folderPath Extracted folder path
	 * @param [out] fileName Extracted file name
	 */
	void SplitPath(const std::string& fullPath, std::string& folderPath, std::string& fileName) {
		std::filesystem::path path(fullPath);
		folderPath = path.parent_path().string();
		fileName = path.filename().string();
	}

	/**
	 * @brief Wrapper function for cv::imwrite
	 * @details Automatically creates folders and saves images
	 * @param [in] save_path Path to save the image
	 * @param [in] img Image to be saved
	 * @return Returns true on success, false on failure
	 */
	bool ImgWrite(const std::string& full_path, const cv::Mat& img)
	{
		std::string folderPath, fileName;
		SplitPath(full_path, folderPath, fileName);
		CheckAndCreateFolder(folderPath);
		return cv::imwrite(full_path, img);
	}

	/**
	 * @brief Displays an image centered in a window of specified screen size.
	 *
	 * @details Creates a black image of the given screen size, places the input image centered within it, and displays the result in the specified window.
	 * @param [in] window_name Name of the window where the image will be displayed.
	 * @param [in] img Image to be displayed.
	 * @param [in] screen_size Size of the screen (window) where the image will be centered.
	 * @exception None
	 */
	void ImgShowCenter(const std::string& window_name, const cv::Mat& img, cv::Size screen_size) {

		cv::Mat screen = cv::Mat::zeros(screen_size, img.type());
		const int x = (screen_size.width - img.cols) / 2;
		const int y = (screen_size.height - img.rows) / 2;

		img.copyTo(screen(cv::Rect(x, y, img.cols, img.rows)));
		cv::imshow(window_name, screen);
	}

	std::string GetZeroPaddingNumberString(const int main_num, const int num_digits)
	{
		std::ostringstream oss;
		// 桁数エラー
		if (std::to_string(main_num).length() > num_digits) {
			throw std::invalid_argument("Number of digits in main_num exceeds num_digits");
		}
		oss << std::setfill('0') << std::setw(num_digits) << main_num;
		return oss.str();
	}

	// -------------------------------------------------------------------------------------
	// RGB compensation functions
	// -------------------------------------------------------------------------------------

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

			cv::Mat captured_image;
			if (!your_functions::Capture(captured_image)) {
				std::cerr << "Failed to capture an image" << std::endl;
			}

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
	void CalcYoshidaImage(const cv::Mat& src, cv::Mat& dst, const cv::Mat& color_mix_mat) {
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


	bool YoshidaRGBCompensationImage(
		const std::vector<cv::Mat>& input_images,
		std::vector<cv::Mat>& output_images,
		const cv::Mat& color_mix_mat,
		const double gamma[3] = GAMMA) {

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
			::CalcYoshidaImage(image, tmp_dst, color_mix_mat);
			yoshida_images.push_back(tmp_dst.clone());
		}

		std::cout << "Calculating Yoshida RGB done" << std::endl;

		std::cout << "Warping images" << std::endl;

		for (int i = 0; i < yoshida_images.size(); i++) {
			cv::Mat& image = yoshida_images[i];
			cv::Mat tmp_yoshida;
			your_functions::ToProjectorCoordinates(image, tmp_yoshida);

			// 入力画像サイズで処理
			cv::Mat tmp_pro = cv::Mat::zeros(input_images[i].size(), CV_8UC3);
			cv::Rect pro_roi = input_images_roi[i];
			tmp_yoshida(pro_roi).copyTo(tmp_pro);

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

	/**
	 * @brief 画像更新処理本体プログラム
	 *
	 * @details プロジェクタ座標系の画像を更新する処理を行う
	 *
	 * @param [in] captured_img キャプチャ画像
	 * @param [in,out] update_img 更新画像
	 * @param [in] target_img ターゲット画像
	 * @param [out] abs_error_img 絶対値差分画像
	 * @param [in] optimization_gain 最適化ゲイン
	 *
	 * @return 成功した場合はtrue, 失敗した場合はfalse
	 */
	bool UpdateImg(
		const cv::Mat& captured_img,
		cv::Mat& update_img,
		const cv::Mat& target_img,
		cv::Mat& abs_error_img,
		const float optimization_gain) {

		if (captured_img.empty() || target_img.empty() || update_img.empty()) {
			std::cerr << "Image is empty" << std::endl;
			return false;
		}

		if (captured_img.size() != target_img.size() || captured_img.size() != update_img.size()) {
			std::cerr << "Image size is different" << std::endl;
			return false;
		}

		if (captured_img.type() != target_img.type() || captured_img.type() != update_img.type()) {
			std::cerr << "Image type is different" << std::endl;
			return false;
		}

		update_img.forEach<cv::Vec3b>([&](cv::Vec3b& p, const int* pos) {
			cv::Vec3b captured_pixel = captured_img.at<cv::Vec3b>(pos[0], pos[1]);
			cv::Vec3b target_pixel = target_img.at<cv::Vec3b>(pos[0], pos[1]);

			// NOTE:差分画像なので必ず符号ありにする
			cv::Vec3s diff_pixel = cv::Vec3s(target_pixel) - cv::Vec3s(captured_pixel);

			// 更新前の画像
			cv::Vec3b before_update_pixel = update_img.at<cv::Vec3b>(pos[0], pos[1]);

			// 更新画像
			p[0] = cv::saturate_cast<uchar>(before_update_pixel[0] + diff_pixel[0] * optimization_gain);
			p[1] = cv::saturate_cast<uchar>(before_update_pixel[1] + diff_pixel[1] * optimization_gain);
			p[2] = cv::saturate_cast<uchar>(before_update_pixel[2] + diff_pixel[2] * optimization_gain);

			});

		cv::absdiff(target_img, update_img, abs_error_img);

		return true;
	}


	// 最適化色補償
	bool RGBOptimization(
		const cv::Mat& target_img,
		const cv::Mat& first_projection_img,
		const std::string& save_folder,
		const int num_iterations,
		const int gain,
		const double gamma[3] = GAMMA) {

		cv::Mat projection_img;	//< 投影画像
		cv::Mat update_img;		//< 更新画像

		// ウィンドウを作成
		::MakeFullScreenWindow("Projection", PRO_WINDOW_X);

		// フルスクリーン用のROIを計算
		cv::Rect center_roi;
		::CalcCenterROI(target_img.size(), center_roi, PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT);

		// ----------------------
		// Initialize the camera
		// ----------------------

		if (!your_functions::InitCamera()) {
			std::cerr << "Failed to initialize the camera" << std::endl;
			return false;
		}

		// -----------------------
		// Start the optimization
		// -----------------------

		for (int num_i = -1; num_i < num_iterations; ++num_i) {
			std::cout << "Iteration: " << num_i << std::endl;

			if (num_i == -1) {
				// 目標画像を投影
				projection_img = first_projection_img.clone();
				update_img = first_projection_img.clone();
			}
			else if (num_i == 0) {
			 // 最初の投影画像を設定
				projection_img = first_projection_img.clone();
				update_img = first_projection_img.clone();
			}

			// ファイル名用のゼロ埋め番号
			std::string zero_padding_num = ::GetZeroPaddingNumberString(num_i, 4);

			// 更新画像を投影
			::ImgShowCenter("Projection", projection_img, cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT));

			// カメラバッファに反映されるまで待つ
			cv::waitKey(CAMERA_WAIT_MS);

			// --------
			// Capture
			// --------

			// NOTE: Please implement your capature function
			cv::Mat captured_img;

			if (!your_functions::Capture(captured_img)) {
				std::cerr << "Failed to capture the image" << std::endl;
				return false;
			}

			// -------
			// Update
			// -------

			// Warp the captured image

			cv::Mat tmp_captured_img;
			your_functions::ToProjectorCoordinates(captured_img, tmp_captured_img);

			cv::Mat warped_captured_img = tmp_captured_img(center_roi).clone();

			cv::Mat error_img;

			// NOTE: 位置ズレ抑制のためにガウシアンフィルタをかけたければどうぞ
			//cv::GaussianBlur(warped_captured_img, warped_captured_img, cv::Size(5, 5), 0);

			// 画像を更新
			if (!::UpdateImg(warped_captured_img, update_img, target_img, error_img, gain)) {
				std::cerr << "Failed to update the image" << std::endl;
				return false;
			}

		}

		// ---------------------
		// Terminate the camera
		// ---------------------

		your_functions::TerminateCamera();

		// ---------------------
		// End the optimization
		// ---------------------

		std::cout << "Optimization complete" << std::endl;

		return true;

	}


}  // namespace
