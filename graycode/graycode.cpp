/**
 * @file graycode.cpp
 * @brief Acquire the correspondence between the camera and the projector using Graycode projection method
 * @author Yoshiaki MAEDA
 * @date 2025/3/14
 */
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\structured_light.hpp>
#include <filesystem>
#include <fstream>
#include <string>




namespace {

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

	// Projector window size
	constexpr int PRO_WINDOW_WIDTH = 1920; //!< Projector window width
	constexpr int PRO_WINDOW_HEIGHT = 1080; //!< Projector window height

	//! Projection Window top-left corner x-coordinate
	constexpr int PRO_WINDOW_X = 3440 + 1920;

	//! Camera wait time during capture in milliseconds
	constexpr int CAMERA_WAIT_MS = 2000;

	constexpr int GRAYCODEWIDTHSTEP = 1;
	constexpr int GRAYCODEHEIGHTSTEP = 1;
	constexpr int GRAYCODEWIDTH = PRO_WINDOW_WIDTH / GRAYCODEWIDTHSTEP;
	constexpr int GRAYCODEHEIGHT = PRO_WINDOW_HEIGHT / GRAYCODEHEIGHTSTEP;
	constexpr int WHITETHRESHOLD = 5;
	constexpr int BLACKTHRESHOLD = 60;

	//! Output file path
	constexpr std::string_view OUTPUT_PATH = "./output/";

	//! SAVEPATHS
	// TODO: �K�؂ȏꏊ�ɕύX����
	constexpr std::string_view GRAYCODE_IMAGE_PATH = "graycode_images/";
	constexpr std::string_view C2P_IMAGE_PATH = "c2p_images/";

	// Camera to Projector Point Correspondence Structure
	struct C2P {
		int cx;
		int cy;
		int px;
		int py;
		C2P(int camera_x, int camera_y, int proj_x, int proj_y) {
			cx = camera_x;
			cy = camera_y;
			px = proj_x;
			py = proj_y;
		}
	};

}

int graycode() {
	namespace yf = your_functions;

	// -----------------------------------
	// ----- Prepare graycode images -----
	// -----------------------------------
	cv::structured_light::GrayCodePattern::Params params;
	params.width = GRAYCODEWIDTH;
	params.height = GRAYCODEHEIGHT;
	auto pattern = cv::structured_light::GrayCodePattern::create(params);

	// �p�r:decode����positive��negative�̉�f�l�̍���
	//      ���whiteThreshold�ȏ�ł����f�̂�decode����
	pattern->setWhiteThreshold(WHITETHRESHOLD);
	// �p�r:ShadowMask�v�Z���� white - black > blackThreshold
	//      �Ȃ�ΑO�i�i�O���C�R�[�h��F�������j�Ɣ��ʂ���
	// ����͂����ݒ肵�Ă��Q�Ƃ���邱�Ƃ͂Ȃ����ꉞ�Z�b�g���Ă���
	pattern->setBlackThreshold(BLACKTHRESHOLD);

	std::vector<cv::Mat> graycodes;
	pattern->generate(graycodes);

#ifdef GETGRAYCODE
	int counter = 0;
	// �t�H���_�����݂��Ȃ��ꍇ�͍쐬
	if (!std::filesystem::exists("./original_graycode/")) {
		std::filesystem::create_directories("./original_graycode/");
	}


	for (auto elem : graycodes) {
		std::ostringstream osss;
		osss << std::setfill('0') << std::setw(2) << counter++;
		imwrite("./original_graycode/graycode_" + osss.str() + ".png", elem);
	}
#endif
#ifndef GETGRAYCODE

	cv::Mat blackCode, whiteCode;
	pattern->getImagesForShadowMasks(blackCode, whiteCode);
	graycodes.push_back(blackCode), graycodes.push_back(whiteCode);

	// -----------------------------
	// ----- Prepare cv window -----
	// -----------------------------
	cv::namedWindow("Pattern", cv::WINDOW_NORMAL);
	cv::resizeWindow("Pattern", GRAYCODEWIDTH, GRAYCODEHEIGHT);
	// NOTE: EDIT HERE!!
	// �v���W�F�N�^�Ƀt���X�N���[���\��
	cv::moveWindow("Pattern", PRO_WINDOW_X, 0);
	cv::setWindowProperty("Pattern", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);


	// ----------------------------------
	// ----- Wait camera adjustment -----
	// ----------------------------------

	// NOTE: �����̃J�����̏��������������Ă�������
	if (!yf::InitCamera()) {
		std::cerr << "Failed to initialize the camera" << std::endl;
		return -1;
	}

	// ��ԍׂ��O���C�R�[�h��\�����ăJ�����̐ݒ���s��
	cv::imshow("Pattern", graycodes[graycodes.size() - 3]);

	// --------------------------------
	// ----- Capture the graycode -----
	// --------------------------------

	std::vector<cv::Mat> captured;
	int cnt = 0;

	// �t�H���_�����݂��Ȃ��ꍇ�͍쐬


	for (auto gimg : graycodes) {
		cv::imshow("Pattern", gimg);
		// �f�B�X�v���C�ɕ\��->�J�����o�b�t�@�ɔ��f�����܂ő҂�
		// �K�v�ȑ҂����Ԃ͎g���J�����Ɉˑ�
		cv::waitKey(CAMERA_WAIT_MS);

		std::cout << "Capturing gray code patern No." << cnt << std::endl;

		std::ostringstream oss;
		oss << std::setfill('0') << std::setw(2) << cnt++;

		/// �O���C�X�P�[���ŎB�e
		/// NOTE: cv:Mat img�ɎB�e�����摜������̂Ŏ����̃J�����ɍ��킹�Ď���
		///	      ���Ă�������
		cv::Mat grayscale_img, tmp;

		std::string raw_path = std::string(OUTPUT_PATH) + std::string(GRAYCODE_IMAGE_PATH) + "captured_graycode_" + oss.str() + ".raw";
		std::string png_path = std::string(OUTPUT_PATH) + std::string(GRAYCODE_IMAGE_PATH) + "captured_graycode_" + oss.str() + ".png";

		if (!yf::Capture(tmp)) {
			std::cerr << "Failed to capture the image" << std::endl;
			return -1;
		}
		// ���m�N�������󂯕t���Ȃ��̂ŕϊ�
		tmp.convertTo(grayscale_img, CV_8UC1);

		//grayscale_img=cv::imread(png_path, cv::IMREAD_GRAYSCALE);

		captured.push_back(grayscale_img.clone());

	}

	/// -----------------------------
	/// ---- Finalize the camera ---- 
	/// -----------------------------

	// NOTE: �����̃J�����̏I���������������Ă�������
	/*cam.Terminate();*/

	// -------------------------------
	// ----- Decode the graycode -----
	// -------------------------------
	// pattern->decode()�͎����}�b�v�̉�͂Ɏg���֐��Ȃ̂ō���͎g��Ȃ�
	// pattern->getProjPixel()���g���Ċe�J������f�Ɏʂ����v���W�F�N�^��f�̍��W���v�Z

	std::cout << "Decoding the graycode..." << std::endl;
	cv::Mat white = captured.back();
	captured.pop_back();
	cv::Mat black = captured.back();
	captured.pop_back();

	int camHeight = captured[0].rows;
	int camWidth = captured[0].cols;

	// c2pX��c2pY��2�`�����l����16�r�b�g�摜�ɓ���
	cv::Mat c2p(camHeight, camWidth, CV_16UC3, cv::Scalar(0, 0, 0));
	cv::Mat p2c(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_16UC3, cv::Scalar(0, 0, 0));

	// cv::inpaint�p��mask�摜
	cv::Mat c2p_mask_img(camHeight, camWidth, CV_8UC1, cv::Scalar(0));

	// p2c��c2p�Ƌt�őΉ�����ꂽ�Ƃ����0�ɂ���
	cv::Mat p2c_mask_img(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC1, cv::Scalar(255));


	// c2pList��������
	std::vector<C2P> c2pList;

	// �摜�T�C�Y�̊m�F
	CV_Assert(white.size() == black.size() && white.size() == c2p.size());

	// �~���[�e�b�N�X
	std::mutex c2pListMutex;
	std::mutex p2cMutex;
	std::mutex p2cMaskImgMutex;

	cv::parallel_for_(cv::Range(0, camHeight), [&](const cv::Range& range) {
		for (int y = range.start; y < range.end; y++) {
			for (int x = 0; x < camWidth; x++) {
				// white��black�̉�f�l���擾
				cv::uint8_t whitePixel = white.at<cv::uint8_t>(y, x);
				cv::uint8_t blackPixel = black.at<cv::uint8_t>(y, x);

				// �������v�Z
				int diff = whitePixel - blackPixel;

				// ������臒l�𒴂���ꍇ�ɏ���
				if (diff > BLACKTHRESHOLD) {

					cv::Point pixel;
					// �Ή�����v���W�F�N�^�̍��W���擾
					if (!pattern->getProjPixel(captured, x, y, pixel)) {
						// c2p�ɍ��W���i�[
						c2p.at<cv::Vec3w>(y, x)[0] = cv::saturate_cast<cv::uint16_t>(pixel.x);
						c2p.at<cv::Vec3w>(y, x)[1] = cv::saturate_cast<cv::uint16_t>(pixel.y);
						c2p.at<cv::Vec3w>(y, x)[2] = 0;

						// p2c�ɍ��W���i�[
						{
							std::lock_guard<std::mutex> lock(p2cMutex);
							p2c.at<cv::Vec3w>(pixel.y, pixel.x)[0] = cv::saturate_cast<cv::uint16_t>(x);
							p2c.at<cv::Vec3w>(pixel.y, pixel.x)[1] = cv::saturate_cast<cv::uint16_t>(y);
							p2c.at<cv::Vec3w>(pixel.y, pixel.x)[2] = 0;
						}

						// �Ή�����ꂽ�獕������
						{
							std::lock_guard<std::mutex> lock(p2cMaskImgMutex);
							p2c_mask_img.at<unsigned char>(pixel.y, pixel.x) = 0;
						}

						// c2pList�ɑΉ��֌W��ǉ�
						{
							std::lock_guard<std::mutex> lock(c2pListMutex);
							c2pList.push_back(C2P(x, y, pixel.x * GRAYCODEWIDTHSTEP, pixel.y * GRAYCODEHEIGHTSTEP));
						}

					}
					else {
					 // c2p�̑Ή������Ă��Ȃ��Ƃ���ɂ�3�`�����l���ڂɍő�l������
					 // ���ƂŃG���[�Ƃ��Ĉ����i�������ɐԂ��\�������j
						c2p.at<cv::Vec3w>(y, x)[0] = 0;
						c2p.at<cv::Vec3w>(y, x)[1] = 0;
						c2p.at<cv::Vec3w>(y, x)[2] = (std::numeric_limits<uint16_t>::max)();

						// mask�摜�ɔ�������
						c2p_mask_img.at<unsigned char>(y, x) = 255;
					}
				}
			}
		}
		});

	// ---------------------------
	// ----- Save C2P as csv -----
	// ---------------------------
	std::cout << "Saving C2P as csv..." << std::endl;

	// �f�[�^���o�b�t�@�ɒ~��
	std::ostringstream oss;
	for (const auto& elem : c2pList) {
		oss << elem.cx << ", " << elem.cy << ", " << elem.px << ", " << elem.py << '\n';
	}

	// ��x�Ƀt�@�C���֏�������
	std::ofstream os("c2p.csv");
	os << oss.str();
	os.close();



	// ----------------------------
	// ----- Visualize result -----
	// ----------------------------
	std::cout << "Visualizing results..." << std::endl;

	// MAP�����i�������C0~255���J��Ԃ� ���K���Ȃ�ver.�j
	cv::Mat viz = cv::Mat::zeros(camHeight, camWidth, CV_8UC3);

	viz.forEach<cv::Vec3b>([&](cv::Vec3b& p, const int* pos) {
		p[0] = static_cast<cv::uint8_t>(c2p.at<cv::Vec3w>(pos[0], pos[1])[0]);
		p[1] = static_cast<cv::uint8_t>(c2p.at<cv::Vec3w>(pos[0], pos[1])[1]);
		p[2] = static_cast<cv::uint8_t>(c2p.at<cv::Vec3w>(pos[0], pos[1])[2]);
		});

	cv::Mat viz2 = cv::Mat::zeros(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC3);

	viz2.forEach<cv::Vec3b>([&](cv::Vec3b& p, const int* pos) {
		p[0] = static_cast<cv::uint8_t>(p2c.at<cv::Vec3w>(pos[0], pos[1])[0]);
		p[1] = static_cast<cv::uint8_t>(p2c.at<cv::Vec3w>(pos[0], pos[1])[1]);
		p[2] = static_cast<cv::uint8_t>(p2c.at<cv::Vec3w>(pos[0], pos[1])[2]);
		});


	// ����������ǔ�C2P�}�b�v�i0~255�ɐ��K���j
	cv::Mat result_img(cv::Size(camWidth, camHeight), CV_8UC3, cv::Scalar(0, 0, 0));

	result_img.forEach<cv::Vec3b>([&](cv::Vec3b& p, const int* pos) {
		p[0] = (unsigned char)((c2p.at<cv::Vec3w>(pos[0], pos[1])[0]) * 255 / PRO_WINDOW_WIDTH);
		p[1] = (unsigned char)((c2p.at<cv::Vec3w>(pos[0], pos[1])[1]) * 255 / PRO_WINDOW_HEIGHT);
		p[2] = cv::saturate_cast<cv::uint8_t>(c2p.at<cv::Vec3w>(pos[0], pos[1])[2]);
		});


	cv::Mat result_img2(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_8UC3, cv::Scalar(0, 0, 0));

	result_img2.forEach<cv::Vec3b>([&](cv::Vec3b& p, const int* pos) {
		p[0] = (unsigned char)((p2c.at<cv::Vec3w>(pos[0], pos[1])[0]) * 255 / camWidth);
		p[1] = (unsigned char)((p2c.at<cv::Vec3w>(pos[0], pos[1])[1]) * 255 / camHeight);
		p[2] = cv::saturate_cast<cv::uint8_t>(p2c.at<cv::Vec3w>(pos[0], pos[1])[2]);
		});


	// OPTIMIZE: �Ȃ񂩃R�[�h�����ʂɒ����C������
	///----------
	/// inpaint
	///----------

	cv::Mat c2px(cv::Size(camWidth, camHeight), CV_16UC1, cv::Scalar(0));
	cv::Mat c2py(cv::Size(camWidth, camHeight), CV_16UC1, cv::Scalar(0));
	cv::Mat p2cx(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_16UC1, cv::Scalar(0));
	cv::Mat p2cy(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_16UC1, cv::Scalar(0));

	// Split c2p and p2c into 2 channels
	cv::Mat c2p_out[2] = { c2px, c2py };
	cv::Mat p2c_out[2] = { p2cx, p2cy };

	int from_to[] = { 0,0, 1,1 };

	cv::mixChannels(&c2p, 1, c2p_out, 2, from_to, 2);

	cv::mixChannels(&p2c, 1, p2c_out, 2, from_to, 2);


	// inpaint
	cv::Mat inpainted_telea_c2px(cv::Size(camWidth, camHeight), CV_16UC1, cv::Scalar(0));
	cv::Mat inpainted_telea_c2py(cv::Size(camWidth, camHeight), CV_16UC1, cv::Scalar(0));
	cv::Mat inpainted_telea_p2cx(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_16UC1, cv::Scalar(0));
	cv::Mat inpainted_telea_p2cy(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_16UC1, cv::Scalar(0));

	cv::Mat inpainted_ns_c2px(cv::Size(camWidth, camHeight), CV_16UC1, cv::Scalar(0));
	cv::Mat inpainted_ns_c2py(cv::Size(camWidth, camHeight), CV_16UC1, cv::Scalar(0));
	cv::Mat inpainted_ns_p2cx(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_16UC1, cv::Scalar(0));
	cv::Mat inpainted_ns_p2cy(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_16UC1, cv::Scalar(0));

	cv::inpaint(c2px, c2p_mask_img, inpainted_telea_c2px, 3, cv::INPAINT_TELEA);
	cv::inpaint(c2py, c2p_mask_img, inpainted_telea_c2py, 3, cv::INPAINT_TELEA);
	cv::inpaint(p2cx, p2c_mask_img, inpainted_telea_p2cx, 3, cv::INPAINT_TELEA);
	cv::inpaint(p2cy, p2c_mask_img, inpainted_telea_p2cy, 3, cv::INPAINT_TELEA);

	cv::inpaint(c2px, c2p_mask_img, inpainted_ns_c2px, 3, cv::INPAINT_NS);
	cv::inpaint(c2py, c2p_mask_img, inpainted_ns_c2py, 3, cv::INPAINT_NS);
	cv::inpaint(p2cx, p2c_mask_img, inpainted_ns_p2cx, 3, cv::INPAINT_NS);
	cv::inpaint(p2cy, p2c_mask_img, inpainted_ns_p2cy, 3, cv::INPAINT_NS);

	// merge
	cv::Mat inpainted_telea_c2p(cv::Size(camWidth, camHeight), CV_16UC2, cv::Scalar(0, 0));
	cv::Mat inpainted_telea_p2c(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_16UC2, cv::Scalar(0, 0));
	cv::Mat inpainted_ns_c2p(cv::Size(camWidth, camHeight), CV_16UC2, cv::Scalar(0, 0));
	cv::Mat inpainted_ns_p2c(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_16UC2, cv::Scalar(0, 0));

	cv::Mat inpainted_telea_c2p_in[2] = { inpainted_telea_c2px, inpainted_telea_c2py };
	cv::Mat inpainted_telea_p2c_in[2] = { inpainted_telea_p2cx, inpainted_telea_p2cy };
	cv::Mat inpainted_ns_c2p_in[2] = { inpainted_ns_c2px, inpainted_ns_c2py };
	cv::Mat inpainted_ns_p2c_in[2] = { inpainted_ns_p2cx, inpainted_ns_p2cy };

	cv::mixChannels(inpainted_telea_c2p_in, 2, &inpainted_telea_c2p, 1, from_to, 2);
	cv::mixChannels(inpainted_telea_p2c_in, 2, &inpainted_telea_p2c, 1, from_to, 2);
	cv::mixChannels(inpainted_ns_c2p_in, 2, &inpainted_ns_c2p, 1, from_to, 2);
	cv::mixChannels(inpainted_ns_p2c_in, 2, &inpainted_ns_p2c, 1, from_to, 2);



	// ---------------------------
	// ----- Save C2P as xml -----
	// ---------------------------

	// �t�H���_�����݂��Ȃ��ꍇ�͍쐬
	if (!std::filesystem::exists(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH))) {
		std::filesystem::create_directories(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH));
	}

	// XML�`����c2p��ۑ�
	cv::FileStorage fs(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "c2p.xml", cv::FileStorage::WRITE);
	fs << "c2p" << c2p;
	fs.release();

	// XML�`����p2c��ۑ�
	cv::FileStorage fs2(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "p2c.xml", cv::FileStorage::WRITE);
	fs2 << "p2c" << p2c;
	fs2.release();

	// XML�`����inpainted_telea_c2p��ۑ�
	cv::FileStorage fs3(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_telea_c2p.xml", cv::FileStorage::WRITE);
	fs3 << "inpainted_telea_c2p" << inpainted_telea_c2p;
	fs3.release();

	// XML�`����inpainted_ns_c2p��ۑ�
	cv::FileStorage fs4(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_ns_c2p.xml", cv::FileStorage::WRITE);
	fs4 << "inpainted_ns_c2p" << inpainted_ns_c2p;
	fs4.release();

	// XML�`����inpainted_telea_p2c��ۑ�
	cv::FileStorage fs5(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_telea_p2c.xml", cv::FileStorage::WRITE);
	fs5 << "inpainted_telea_p2c" << inpainted_telea_p2c;
	fs5.release();

	// XML�`����inpainted_ns_p2c��ۑ�
	cv::FileStorage fs6(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_ns_p2c.xml", cv::FileStorage::WRITE);
	fs6 << "inpainted_ns_p2c" << inpainted_ns_p2c;
	fs6.release();

	cv::Mat inpainted_telea_c2p_3ch(cv::Size(camWidth, camHeight), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat inpainted_ns_c2p_3ch(cv::Size(camWidth, camHeight), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat inpainted_telea_p2c_3ch(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat inpainted_ns_p2c_3ch(cv::Size(PRO_WINDOW_WIDTH, PRO_WINDOW_HEIGHT), CV_8UC3, cv::Scalar(0, 0, 0));

	inpainted_telea_c2p_3ch.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int* position) -> void {
		pixel[0] = (unsigned char)((inpainted_telea_c2p.at<cv::Vec2w>(position[0], position[1])[0]) * 255 / PRO_WINDOW_WIDTH);
		pixel[1] = (unsigned char)((inpainted_telea_c2p.at<cv::Vec2w>(position[0], position[1])[1]) * 255 / PRO_WINDOW_HEIGHT);
		pixel[2] = 0;
		});

	inpainted_ns_c2p_3ch.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int* position) -> void {
		pixel[0] = (unsigned char)((inpainted_ns_c2p.at<cv::Vec2w>(position[0], position[1])[0]) * 255 / PRO_WINDOW_WIDTH);
		pixel[1] = (unsigned char)((inpainted_ns_c2p.at<cv::Vec2w>(position[0], position[1])[1]) * 255 / PRO_WINDOW_HEIGHT);
		pixel[2] = 0;
		});

	inpainted_telea_p2c_3ch.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int* position) -> void {
		pixel[0] = (unsigned char)((inpainted_telea_p2c.at<cv::Vec2w>(position[0], position[1])[0]) * 255 / camWidth);
		pixel[1] = (unsigned char)((inpainted_telea_p2c.at<cv::Vec2w>(position[0], position[1])[1]) * 255 / camHeight);
		pixel[2] = 0;
		});

	inpainted_ns_p2c_3ch.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int* position) -> void {
		pixel[0] = (unsigned char)((inpainted_ns_p2c.at<cv::Vec2w>(position[0], position[1])[0]) * 255 / camWidth);
		pixel[1] = (unsigned char)((inpainted_ns_p2c.at<cv::Vec2w>(position[0], position[1])[1]) * 255 / camHeight);
		pixel[2] = 0;
		});

	// ---------------------------------------
	// ----- Save C2P Visualized results -----
	// ---------------------------------------

	std::cout << "Saving C2P Visualized results..." << std::endl;

	// �摜��ۑ�
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_telea_c2p.png", inpainted_telea_c2p_3ch);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_ns_c2p.png", inpainted_ns_c2p_3ch);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_telea_p2c.png", inpainted_telea_p2c_3ch);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_ns_p2c.png", inpainted_ns_p2c_3ch);

	// �摜��\��
	cv::imshow("inpainted_telea_c2p", inpainted_telea_c2p_3ch);
	cv::imshow("inpainted_ns_c2p", inpainted_ns_c2p_3ch);
	cv::imshow("inpainted_telea_p2c", inpainted_telea_p2c_3ch);
	cv::imshow("inpainted_ns_p2c", inpainted_ns_p2c_3ch);

	// Visualized C2P for debug
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "default_c2p.png", viz);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "default_p2c.png", viz2);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "true_c2p.png", result_img);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "true_p2c.png", result_img2);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(GRAYCODE_IMAGE_PATH) + "c2p_mask_image.png", c2p_mask_img);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(GRAYCODE_IMAGE_PATH) + "p2c_mask_image.png", p2c_mask_img);

	cv::imshow("result", viz);
	cv::imshow("result2", viz2);
	cv::imshow("result3", result_img);
	cv::imshow("result4", result_img2);

	cv::waitKey(0);
#endif // !GETGRAYCODE
	return 0;
}



