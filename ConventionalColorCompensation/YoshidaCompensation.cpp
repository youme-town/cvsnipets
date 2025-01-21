/**
 * @file YoshidaCompensation.cpp
 * @brief RGB compensation using Yoshida's method
 * @author Yoshiaki Maeda
 * @date 2025/01/21
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include "YoshidaColorMixingMatrix.hpp"

/** @name Constants Group
 *  @brief Constants used in this program
 *  @note Please modify these values according to your environment
 */
/* @{ */
constexpr int COLOR_GRADATION = 3;  //!< The number of gradations for each color (R, G, B)
constexpr int CAMERA_WAIT_MS = 1000;  //!< The waiting time for the camera to capture an image [ms]
constexpr int PRO_WINDOW_X = 0;  //!< The x-coordinate of the projector window
constexpr int PRO_WINDOW_WIDTH = 1920;  //!< The width of the projector window
constexpr int PRO_WINDOW_HEIGHT = 1080;  //!< The height of the projector window
constexpr int ROI_IMAGE_WIDTH = 512;  //!< The width of the ROI
constexpr int ROI_IMAGE_HEIGHT = 512;  //!< The height of the ROI

/* @} */

/** @name File Path Group
 *  @brief File paths used in this program
 *  @note Please modify these values according to your environment
 */
/* @{ */
constexpr std::string_view CURRENT_PATH = "./";  //!< The current folder path
constexpr std::string_view OUTPUT_PATH = CURRENT_PATH + "output/";  //!< The output folder path



static bool AcquireYoshidaColorMixingMat() {
	namespace us = useful_functions;
	namespace rf = related_functions;

	std::vector<cv::Mat> projectionImages;  //!< �v���W�F�N�^�œ��e����摜�̔z��

	// Generate Images (�K���}�␳���l�����Ă���)
	cv::Mat tempImg = cv::Mat::zeros(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC3);
	std::vector<cv::Mat> planes, planes_gamma;
	const int colorNum = COLOR_GRADATION * COLOR_GRADATION * COLOR_GRADATION;

	// calculate inverse gamma
	cv::Mat lut = cv::Mat(1, 256, CV_8UC3);      //< ���b�N�A�b�v�e�[�u���p�z��
	us::CalcInvGammaLUT(lut, GAMMA);

	int color_grad_cnt = 0;
	for (int i = 0; i < COLOR_GRADATION; i++) {
		for (int j = 0; j < COLOR_GRADATION; j++) {
			for (int k = 0; k < COLOR_GRADATION; k++) {
				std::ostringstream oss;
				oss << std::setfill('0') << std::setw(4) << color_grad_cnt++;

				// �v���W�F�N�^���e�摜�̐���
				for (int y = 0; y < tempImg.rows; y++) {
					for (int x = 0; x < tempImg.cols; x++) {
						tempImg.at<cv::Vec3b>(y, x)[0] = 255 * i / (COLOR_GRADATION - 1);
						tempImg.at<cv::Vec3b>(y, x)[1] = 255 * j / (COLOR_GRADATION - 1);
						tempImg.at<cv::Vec3b>(y, x)[2] = 255 * k / (COLOR_GRADATION - 1);
					}
				}

				//��������gamma
				cv::Mat tempImg_gamma = cv::Mat::zeros(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC3);
				cv::LUT(tempImg, lut, tempImg_gamma);  // ���b�N�A�b�v�e�[�u���ϊ�
				//�����܂�gamma

				// Save the projection image for debug
				us::ImgWrite(std::string(OUTPUT_PATH) + std::string(PROJECTION_PATTERN_PATH) + "procjection_pattern_" + oss.str() + ".png", tempImg_gamma);

				projectionImages.push_back(tempImg_gamma.clone());
			}
		}
	}

	std::cout << "Patterns generation complete" << std::endl;

	//���e����E�B���h�E�̍쐬
	cv::namedWindow("projection_win", cv::WINDOW_NORMAL);
	cv::moveWindow("projection_win", PRO_WINDOW_X, 0);
	cv::setWindowProperty("projection_win", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

	// ----------------------
	// Initialize the camera
	// ----------------------
	//rf::InitYourCam(cam);

	// -------------
	// Capture loop
	// -------------

	std::vector<cv::Mat> captured_patterns;

	for (int i = 0; i < projectionImages.size(); i++) {
		cv::imshow("projection_win", projectionImages[i]);
		// �f�B�X�v���C�ɕ\��->�J�����o�b�t�@�ɔ��f�����܂ő҂�
		// �K�v�ȑ҂����Ԃ͎g���J�����Ɉˑ�
		cv::waitKey(CAMERA_WAIT_MS);

		std::ostringstream os_cam;
		os_cam << std::setfill('0') << std::setw(4) << i;

		// ------------------
		// Capture the image
		// ------------------

		// NOTE: Please implement the function to capture the image
		cv::Mat image;	//!< Please assign the captured image to this variable

		std::string raw_path = std::string(OUTPUT_PATH) + std::string(RAW_IMAGE_PATH) + "captured_raw_" + os_cam.str() + ".CR2";
		std::string png_path = std::string(OUTPUT_PATH) + std::string(CAPTURED_PATTERN_PATH) + "captured_pattern_" + os_cam.str() + ".png";
		if (!rf::CaptureWithYourCam(cam, raw_path, png_path, image)) {
			std::cerr << "Failed to capture the image" << std::endl;
			return false;
		}

		//�B�e���ꂽ�摜���x�N�g���Ƀv�b�V������
		captured_patterns.push_back(image.clone());
	}

	// ---------------------
	// Terminate the camera
	// ---------------------

	//rf::TerminateYourCam(cam);

	std::cout << "Capture complete" << std::endl;

	// �F�␳�s��̌v�Z����

	// �B�e���ꂽ�摜����J�����̉𑜓x���擾
	const int camera_width = captured_patterns[0].cols;
	const int camera_height = captured_patterns[0].rows;

	// �B�e�摜�Ƃ̔�r�p�F�p�^�[���̐���
	cv::Mat tempPtn = cv::Mat::zeros(camera_height, camera_width, CV_8UC3);
	std::vector<cv::Mat> idealPatterns;

	int pattern_grad_cnt = 0;

	for (int i = 0; i < COLOR_GRADATION; i++) {
		for (int j = 0; j < COLOR_GRADATION; j++) {
			for (int k = 0; k < COLOR_GRADATION; k++) {
				std::ostringstream oss;
				oss << std::setfill('0') << std::setw(4) << pattern_grad_cnt++;

				tempPtn.forEach<cv::Vec3b>([&](cv::Vec3b& p, const int* pos) {
					p[0] = 255 * i / (COLOR_GRADATION - 1);
					p[1] = 255 * j / (COLOR_GRADATION - 1);
					p[2] = 255 * k / (COLOR_GRADATION - 1);
					});

				// Save the ideal pattern for debug
				us::ImgWrite(std::string(OUTPUT_PATH) + std::string(IDEAL_PATTERN_PATH) + "ideal_pattern_" + oss.str() + ".png", tempPtn);

				idealPatterns.push_back(tempPtn.clone());
			}
		}
	}

	// �F�␳�s��̌v�Z����
	YoshidaColorMixingMatrix yoshida;

	// �F�ϊ��s��̗v�f���i=�`�����l�����j
	const int num_channel_YoshidaCMM = 12;

	cv::Mat color_mix_mat = cv::Mat::zeros(camera_height, camera_width, CV_64FC(num_channel_YoshidaCMM));
	cv::Mat captured_pixel_values = cv::Mat::zeros(colorNum * 3, 1, CV_64FC1);
	cv::Mat ideal_pixel_values = cv::Mat::zeros(colorNum * 3, 1, CV_64FC1);

	std::cout << "�F�ϊ��v�Z�J�n" << std::endl;

	const int num_captured = captured_patterns.size();
	std::mutex mtx;
	color_mix_mat.forEach<cv::Vec<double, num_channel_YoshidaCMM>>([&](cv::Vec<double, num_channel_YoshidaCMM>& p, const int* pos)->void {
		cv::Mat captured_pixel_values = cv::Mat::zeros(colorNum * 3, 1, CV_64FC1);
		cv::Mat ideal_pixel_values = cv::Mat::zeros(colorNum * 3, 1, CV_64FC1);
		// Load RGB values for matrix computation
		for (int i = 0; i < num_captured; i++) {
			captured_pixel_values.at<double>(3 * i, 0) = (double)captured_patterns[i].at<cv::Vec3b>(pos[0], pos[1])[0];
			captured_pixel_values.at<double>(3 * i + 1, 0) = (double)captured_patterns[i].at<cv::Vec3b>(pos[0], pos[1])[1];
			captured_pixel_values.at<double>(3 * i + 2, 0) = (double)captured_patterns[i].at<cv::Vec3b>(pos[0], pos[1])[2];

			ideal_pixel_values.at<double>(3 * i, 0) = (double)idealPatterns[i].at<cv::Vec3b>(pos[0], pos[1])[0];
			ideal_pixel_values.at<double>(3 * i + 1, 0) = (double)idealPatterns[i].at<cv::Vec3b>(pos[0], pos[1])[1];
			ideal_pixel_values.at<double>(3 * i + 2, 0) = (double)idealPatterns[i].at<cv::Vec3b>(pos[0], pos[1])[2];
		}
		// Color Caluculation Class Instance
		YoshidaColorMixingMatrix yoshida;

		// Matrix Computation	
		p = yoshida.ComputeMatrix1(captured_pixel_values.clone(), ideal_pixel_values.clone());

		});


	std::cout << "�F�ϊ��v�Z����" << std::endl;

	std::cout << "mat file�ւ̏�������" << std::endl;

	std::string cmm_path = std::string(OUTPUT_PATH) + std::string(COLOR_MIXING_MATRIX_NAME) + ".mat";

	if (!us::SaveMatBinary(cmm_path, color_mix_mat)) {
		std::cerr << "Failed to write the color mixing matrix" << std::endl;
		return false;
	}

	std::cout << "mat file�ւ̏������݊���" << std::endl;

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
static void CalcYoshidaColor(const cv::Mat& src, cv::Mat& dst, const cv::Mat& color_mix_mat) {
	std::mutex mtx;
	dst = cv::Mat(src.size(), src.type());

	// ���Ȃ��݂�forEach(pos[0]��row index,pos[1]��column index)
	src.forEach<cv::Vec3b>([&](cv::Vec3b& p, const int* pos)->void {
		cv::Matx<double, 3, 4> tmp_cmm;
		cv::Mat tmp_output_RGB_camera = cv::Mat::zeros(4, 1, CV_64FC1);
		cv::Mat tmp_input_RGB_projector = cv::Mat::zeros(3, 1, CV_64FC1);

		// CMM�ɐF�␳�s�����
		for (int k = 0; k < 3; k++) {
			for (int l = 0; l < 4; l++) {
				tmp_cmm(k, l) = color_mix_mat.at<cv::Vec<double, 12>>(pos[0], pos[1])[4 * k + l];
			}
		}

		// �J�����̖ڕWRGB�o�͂������Čv�Z
		tmp_output_RGB_camera.at<double>(0, 0) = p[0];
		tmp_output_RGB_camera.at<double>(1, 0) = p[1];
		tmp_output_RGB_camera.at<double>(2, 0) = p[2];
		tmp_output_RGB_camera.at<double>(3, 0) = 1.0;

		// �v���W�F�N�^�ɓ��͂��ׂ�RGB�����܂�
		tmp_input_RGB_projector = tmp_cmm * tmp_output_RGB_camera;


		std::lock_guard<std::mutex> lock(mtx);
		{
			// �v���W�F�N�^�ɓ��͂��ׂ�RGB���摜�ɑ��(�K���}�␳���l�����Ă��Ȃ�)
			for (int i = 0; i < 3/*Number of Channels*/; ++i) {
				dst.at<cv::Vec3b>(pos[0], pos[1])[i]
					= cv::saturate_cast<uchar>(tmp_input_RGB_projector.at<double>(i, 0));
			}

		}

		});


}


static bool CalcYoshidaRGBCompensationImage(
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
	// �F�␳�s��̓ǂݍ���
	cv::Mat color_mix_mat;
	std::string cmm_path = std::string(OUTPUT_PATH) + std::string(COLOR_MIXING_MATRIX_NAME) + ".mat";
	if (!us::LoadMatBinary(cmm_path, color_mix_mat)) {
		std::cerr << "Failed to read the color mixing matrix" << std::endl;
		return false;
	}
	std::cout << "Reading done" << std::endl;

	std::cout << "Read images" << std::endl;

	// �摜�̓ǂݍ���
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

	// vector<cv::Mat>�Ɋi�[
	std::vector<cv::Mat> input_images;

	// �E�B���h�E�T�C�Y�̌v�Z
	cv::Rect pro_roi;
	if (!us::CalcCenterROI(cv::Size(EXP_IMAGE_WIDTH, EXP_IMAGE_HEIGHT), pro_roi, PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT)) {
		std::cerr << "Calculation ROI failed" << std::endl;
		return false;
	}


	// �t���X�N���[���摜�ɕς��Ă���x�N�g���Ɋi�[
	for (auto& im : images) {
		cv::Mat tmp_resize, tmp_pro;
		// ���T�C�Y
		cv::resize(im.image.clone(), tmp_resize, cv::Size(EXP_IMAGE_WIDTH, EXP_IMAGE_HEIGHT));

		// �E�B���h�E�T�C�Y�ɕϊ�
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

		// ���e�摜�̈�ȊO�����Ŗ��߂�
		// �����摜���쐬
		cv::Mat tmp_pro = cv::Mat::zeros(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC3);
		// �K�v�ȕ����ɑ��
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

	// projection_images�̉摜��images�ɖ߂�
	for (size_t i = 0; i < images.size(); ++i) {
		// �K���}�␳
		cv::LUT(projection_images[i], lut, projection_images[i]);
		images[i].image = projection_images[i];
	}

	// prefix���w�肵�ĉ摜��ۑ�
	us::SaveProcessedImages(images, output_folder_path, "Yoshida_compensated_");

	std::cout << "Save done" << std::endl;

	return true;

}


