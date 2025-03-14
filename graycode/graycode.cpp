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
	// TODO: 適切な場所に変更する
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

	// 用途:decode時にpositiveとnegativeの画素値の差が
	//      常にwhiteThreshold以上である画素のみdecodeする
	pattern->setWhiteThreshold(WHITETHRESHOLD);
	// 用途:ShadowMask計算時に white - black > blackThreshold
	//      ならば前景（グレイコードを認識した）と判別する
	// 今回はこれを設定しても参照されることはないが一応セットしておく
	pattern->setBlackThreshold(BLACKTHRESHOLD);

	std::vector<cv::Mat> graycodes;
	pattern->generate(graycodes);

#ifdef GETGRAYCODE
	int counter = 0;
	// フォルダが存在しない場合は作成
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
	// プロジェクタにフルスクリーン表示
	cv::moveWindow("Pattern", PRO_WINDOW_X, 0);
	cv::setWindowProperty("Pattern", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);


	// ----------------------------------
	// ----- Wait camera adjustment -----
	// ----------------------------------

	// NOTE: 自分のカメラの初期化を実装してください
	if (!yf::InitCamera()) {
		std::cerr << "Failed to initialize the camera" << std::endl;
		return -1;
	}

	// 一番細いグレイコードを表示してカメラの設定を行う
	cv::imshow("Pattern", graycodes[graycodes.size() - 3]);

	// --------------------------------
	// ----- Capture the graycode -----
	// --------------------------------

	std::vector<cv::Mat> captured;
	int cnt = 0;

	// フォルダが存在しない場合は作成


	for (auto gimg : graycodes) {
		cv::imshow("Pattern", gimg);
		// ディスプレイに表示->カメラバッファに反映されるまで待つ
		// 必要な待ち時間は使うカメラに依存
		cv::waitKey(CAMERA_WAIT_MS);

		std::cout << "Capturing gray code patern No." << cnt << std::endl;

		std::ostringstream oss;
		oss << std::setfill('0') << std::setw(2) << cnt++;

		/// グレイスケールで撮影
		/// NOTE: cv:Mat imgに撮影した画像が入るので自分のカメラに合わせて実装
		///	      してください
		cv::Mat grayscale_img, tmp;

		std::string raw_path = std::string(OUTPUT_PATH) + std::string(GRAYCODE_IMAGE_PATH) + "captured_graycode_" + oss.str() + ".raw";
		std::string png_path = std::string(OUTPUT_PATH) + std::string(GRAYCODE_IMAGE_PATH) + "captured_graycode_" + oss.str() + ".png";

		if (!yf::Capture(tmp)) {
			std::cerr << "Failed to capture the image" << std::endl;
			return -1;
		}
		// モノクロしか受け付けないので変換
		tmp.convertTo(grayscale_img, CV_8UC1);

		//grayscale_img=cv::imread(png_path, cv::IMREAD_GRAYSCALE);

		captured.push_back(grayscale_img.clone());

	}

	/// -----------------------------
	/// ---- Finalize the camera ---- 
	/// -----------------------------

	// NOTE: 自分のカメラの終了処理を実装してください
	/*cam.Terminate();*/

	// -------------------------------
	// ----- Decode the graycode -----
	// -------------------------------
	// pattern->decode()は視差マップの解析に使う関数なので今回は使わない
	// pattern->getProjPixel()を使って各カメラ画素に写ったプロジェクタ画素の座標を計算

	std::cout << "Decoding the graycode..." << std::endl;
	cv::Mat white = captured.back();
	captured.pop_back();
	cv::Mat black = captured.back();
	captured.pop_back();

	int camHeight = captured[0].rows;
	int camWidth = captured[0].cols;

	// c2pXとc2pYを2チャンネルの16ビット画像に統合
	cv::Mat c2p(camHeight, camWidth, CV_16UC3, cv::Scalar(0, 0, 0));
	cv::Mat p2c(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_16UC3, cv::Scalar(0, 0, 0));

	// cv::inpaint用のmask画像
	cv::Mat c2p_mask_img(camHeight, camWidth, CV_8UC1, cv::Scalar(0));

	// p2cはc2pと逆で対応が取れたところを0にする
	cv::Mat p2c_mask_img(PRO_WINDOW_HEIGHT, PRO_WINDOW_WIDTH, CV_8UC1, cv::Scalar(255));


	// c2pListを初期化
	std::vector<C2P> c2pList;

	// 画像サイズの確認
	CV_Assert(white.size() == black.size() && white.size() == c2p.size());

	// ミューテックス
	std::mutex c2pListMutex;
	std::mutex p2cMutex;
	std::mutex p2cMaskImgMutex;

	cv::parallel_for_(cv::Range(0, camHeight), [&](const cv::Range& range) {
		for (int y = range.start; y < range.end; y++) {
			for (int x = 0; x < camWidth; x++) {
				// whiteとblackの画素値を取得
				cv::uint8_t whitePixel = white.at<cv::uint8_t>(y, x);
				cv::uint8_t blackPixel = black.at<cv::uint8_t>(y, x);

				// 差分を計算
				int diff = whitePixel - blackPixel;

				// 差分が閾値を超える場合に処理
				if (diff > BLACKTHRESHOLD) {

					cv::Point pixel;
					// 対応するプロジェクタの座標を取得
					if (!pattern->getProjPixel(captured, x, y, pixel)) {
						// c2pに座標を格納
						c2p.at<cv::Vec3w>(y, x)[0] = cv::saturate_cast<cv::uint16_t>(pixel.x);
						c2p.at<cv::Vec3w>(y, x)[1] = cv::saturate_cast<cv::uint16_t>(pixel.y);
						c2p.at<cv::Vec3w>(y, x)[2] = 0;

						// p2cに座標を格納
						{
							std::lock_guard<std::mutex> lock(p2cMutex);
							p2c.at<cv::Vec3w>(pixel.y, pixel.x)[0] = cv::saturate_cast<cv::uint16_t>(x);
							p2c.at<cv::Vec3w>(pixel.y, pixel.x)[1] = cv::saturate_cast<cv::uint16_t>(y);
							p2c.at<cv::Vec3w>(pixel.y, pixel.x)[2] = 0;
						}

						// 対応が取れたら黒を入れる
						{
							std::lock_guard<std::mutex> lock(p2cMaskImgMutex);
							p2c_mask_img.at<unsigned char>(pixel.y, pixel.x) = 0;
						}

						// c2pListに対応関係を追加
						{
							std::lock_guard<std::mutex> lock(c2pListMutex);
							c2pList.push_back(C2P(x, y, pixel.x * GRAYCODEWIDTHSTEP, pixel.y * GRAYCODEHEIGHTSTEP));
						}

					}
					else {
					 // c2pの対応が取れていないところには3チャンネル目に最大値を入れる
					 // ことでエラーとして扱う（可視化時に赤く表示される）
						c2p.at<cv::Vec3w>(y, x)[0] = 0;
						c2p.at<cv::Vec3w>(y, x)[1] = 0;
						c2p.at<cv::Vec3w>(y, x)[2] = (std::numeric_limits<uint16_t>::max)();

						// mask画像に白を入れる
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

	// データをバッファに蓄積
	std::ostringstream oss;
	for (const auto& elem : c2pList) {
		oss << elem.cx << ", " << elem.cy << ", " << elem.px << ", " << elem.py << '\n';
	}

	// 一度にファイルへ書き込む
	std::ofstream os("c2p.csv");
	os << oss.str();
	os.close();



	// ----------------------------
	// ----- Visualize result -----
	// ----------------------------
	std::cout << "Visualizing results..." << std::endl;

	// MAP可視化（ただし，0~255を繰り返す 正規化なしver.）
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


	// ここから改良版C2Pマップ（0~255に正規化）
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


	// OPTIMIZE: なんかコードが無駄に長い気がする
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

	// フォルダが存在しない場合は作成
	if (!std::filesystem::exists(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH))) {
		std::filesystem::create_directories(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH));
	}

	// XML形式でc2pを保存
	cv::FileStorage fs(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "c2p.xml", cv::FileStorage::WRITE);
	fs << "c2p" << c2p;
	fs.release();

	// XML形式でp2cを保存
	cv::FileStorage fs2(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "p2c.xml", cv::FileStorage::WRITE);
	fs2 << "p2c" << p2c;
	fs2.release();

	// XML形式でinpainted_telea_c2pを保存
	cv::FileStorage fs3(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_telea_c2p.xml", cv::FileStorage::WRITE);
	fs3 << "inpainted_telea_c2p" << inpainted_telea_c2p;
	fs3.release();

	// XML形式でinpainted_ns_c2pを保存
	cv::FileStorage fs4(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_ns_c2p.xml", cv::FileStorage::WRITE);
	fs4 << "inpainted_ns_c2p" << inpainted_ns_c2p;
	fs4.release();

	// XML形式でinpainted_telea_p2cを保存
	cv::FileStorage fs5(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_telea_p2c.xml", cv::FileStorage::WRITE);
	fs5 << "inpainted_telea_p2c" << inpainted_telea_p2c;
	fs5.release();

	// XML形式でinpainted_ns_p2cを保存
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

	// 画像を保存
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_telea_c2p.png", inpainted_telea_c2p_3ch);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_ns_c2p.png", inpainted_ns_c2p_3ch);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_telea_p2c.png", inpainted_telea_p2c_3ch);
	cv::imwrite(std::string(OUTPUT_PATH) + std::string(C2P_IMAGE_PATH) + "inpainted_ns_p2c.png", inpainted_ns_p2c_3ch);

	// 画像を表示
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



