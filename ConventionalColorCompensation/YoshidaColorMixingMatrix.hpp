#pragma once
/*
 *  Implementation of color mixing matrix considering environment light
 *  2012.02.02 Daisuke Iwai
 *  2019.09.05 rewrited by Takefumi Hiraki
 *  2025.01.21 revised by Yoshiaki Maeda
 *
 *  Reference:
 *  Takenobu Yoshida, Chinatsu Horii, and Kosuke Sato, "A Virtual Color Reconstruction
 *  System for Real Heritage with Light Projection," In Proceedings of International
 *  Conference on Visual System and MultiMedia (VSMM), pp.161-168, 2003.
 */
#include <iostream>
#include <opencv2/opencv.hpp>

class YoshidaColorMixingMatrix
{

public:
    YoshidaColorMixingMatrix() {};
    ~YoshidaColorMixingMatrix() {};

    /*
     *  This method computes color mixing matrix and save it as col_mix_mat[].
     */
    cv::Mat ComputeMatrix12(cv::Mat cam_rgb, cv::Mat prj_rgb1, cv::Mat prj_rgb2)
    {

        cv::Mat colorMix12 = cv::Mat::zeros(21, 1, CV_64FC1);
        /*
         *  The color mixing matrix converts camera's captured (or target) RGB value to the
         *  corresponding projector's input RGB value.
         *
         *  |r_p|   |a_11 a_12 a_13 a_14||r_c|
         *  |g_p|   |a_21 a_22 a_23 a_24||g_c|
         *  |b_p| = |a_31 a_32 a_33 a_34||b_c|
         *                               | 1 |
         *  input RGB values for projector = color mixing matrix * captured RGB values for camera
         *
         *  Solve unknown color mixing matrix(a_11~a_34) with n correspondences
         *
         *  |r_p1|   |r_c1 g_c1 b_c1 1  0    0    0   0  0    0    0   0||a_11|
         *  |g_p1|   | 0    0    0   0 r_c1 g_c1 b_c1 1  0    0    0   0||a_12|
         *  |b_p1|   | 0    0    0   0  0    0    0   0 r_c1 g_c1 b_c1 1||a_13|
         *  | :  | = |                      :                           || :  |
         *  |r_pn|   |r_cn g_cn b_cn 1  0    0    0   0  0    0    0   0||a_32|
         *  |g_pn|   | 0    0    0   0 r_cn g_cn b_cn 1  0    0    0   0||a_33|
         *  |b_pn|   | 0    0    0   0  0    0    0   0 r_cn g_cn b_cn 1||a_34|
         *  y      =                         A                             x
         */
        int colorNum = 7;
        /*int colorNum = prj_rgb1.rows / 3;*/

        if (colorNum < 7) {
            std::cout << "More than 7 (>=7) correspondences are required." << std::endl;
            exit(-1);
        }
        cv::Mat Y = cam_rgb;
        cv::Mat A = cv::Mat::zeros(colorNum * 3, 21, CV_64FC1);

        //std::cout << "Y: " << Y.t() << std::endl;

        // AÇ…ÉJÉÅÉâÇ≈ÉLÉÉÉvÉ`ÉÉÇµÇΩâÊëfÇÃRGBílÇäiî[
        for (int n = 0; n < colorNum * 3; n++) {
            for (int m = 0; m < 3; m++) {
                int k = n / 3;
                int l = n % 3;
                if (m == l) {
                    A.at<double>(n, m * 7) = prj_rgb1.at<double>(k * 3, 0);
                    A.at<double>(n, m * 7 + 1) = prj_rgb1.at<double>(k * 3 + 1, 0);
                    A.at<double>(n, m * 7 + 2) = prj_rgb1.at<double>(k * 3 + 2, 0);
                    A.at<double>(n, m * 7 + 3) = prj_rgb2.at<double>(k * 3, 0);
                    A.at<double>(n, m * 7 + 4) = prj_rgb2.at<double>(k * 3 + 1, 0);
                    A.at<double>(n, m * 7 + 5) = prj_rgb2.at<double>(k * 3 + 2, 0);
                    A.at<double>(n, m * 7 + 6) = 1;
                }
                else {
                    A.at<double>(n, m * 7) = 0;
                    A.at<double>(n, m * 7 + 1) = 0;
                    A.at<double>(n, m * 7 + 2) = 0;
                    A.at<double>(n, m * 7 + 3) = 0;
                    A.at<double>(n, m * 7 + 4) = 0;
                    A.at<double>(n, m * 7 + 5) = 0;
                    A.at<double>(n, m * 7 + 6) = 0;
                }
            }
        }

        if (cv::determinant((A.t() * A).inv()) == 0) {
            printf("ErrorÅFãtçsóÒÇ™ãÅÇ‹ÇËÇ‹ÇπÇÒÅD\n");
        }/*
        std::cout << "A: " << (A.t() * A).inv() << std::endl;*/

        // çsóÒââéZÇ≈XÇãÅÇﬂÇÈ
        colorMix12 = (A.t() * A).inv() * A.t() * Y;
        //colorMix12 = A.inv() * Y;

        return colorMix12;
    };

    cv::Mat ComputeMatrix1(const cv::Mat cam_rgb, const cv::Mat prj_rgb) const
    {
        cv::Mat colorMix1 = cv::Mat::zeros(12, 1, CV_64FC1);
        /*
         *  The color mixing matrix converts camera's captured (or target) RGB value to the
         *  corresponding projector's input RGB value.
         *
         *  |r_p|   |a_11 a_12 a_13 a_14||r_c|
         *  |g_p|   |a_21 a_22 a_23 a_24||g_c|
         *  |b_p| = |a_31 a_32 a_33 a_34||b_c|
         *                               | 1 |
         *  input RGB values for projector = color mixing matrix * captured RGB values for camera
         *
         *  Solve unknown color mixing matrix(a_11~a_34) with n correspondences
         *
         *  |r_p1|   |r_c1 g_c1 b_c1 1  0    0    0   0  0    0    0   0||a_11|
         *  |g_p1|   | 0    0    0   0 r_c1 g_c1 b_c1 1  0    0    0   0||a_12|
         *  |b_p1|   | 0    0    0   0  0    0    0   0 r_c1 g_c1 b_c1 1||a_13|
         *  | :  | = |                      :                           || :  |
         *  |r_pn|   |r_cn g_cn b_cn 1  0    0    0   0  0    0    0   0||a_32|
         *  |g_pn|   | 0    0    0   0 r_cn g_cn b_cn 1  0    0    0   0||a_33|
         *  |b_pn|   | 0    0    0   0  0    0    0   0 r_cn g_cn b_cn 1||a_34|
         *  y      =                         A                             x
         */

        int colorNum = prj_rgb.rows / 3;

        if (colorNum < 4) {
            std::cout << "More than 4 (>=4) correspondences are required." << std::endl;
            exit(-1);
        }
        cv::Mat Y = prj_rgb;
        cv::Mat A = cv::Mat::zeros(colorNum * 3, 12, CV_64FC1);
        // std::cout << "Y: " << Y.t() << std::endl;

        for (int n = 0; n < colorNum * 3; n++) {
            for (int m = 0; m < 3; m++) {
                int k = n / 3;
                int l = n % 3;
                if (m == l) {
                    A.at<double>(n, m * 4) = cam_rgb.at<double>(k * 3, 0);
                    A.at<double>(n, m * 4 + 1) = cam_rgb.at<double>(k * 3 + 1, 0);
                    A.at<double>(n, m * 4 + 2) = cam_rgb.at<double>(k * 3 + 2, 0);
                    A.at<double>(n, m * 4 + 3) = 1;
                }
                else {
                    A.at<double>(n, m * 4) = 0;
                    A.at<double>(n, m * 4 + 1) = 0;
                    A.at<double>(n, m * 4 + 2) = 0;
                    A.at<double>(n, m * 4 + 3) = 0;
                }
            }
        }

        // std::cout << "A: " << A << std::endl;

        colorMix1 = (A.t() * A).inv() * A.t() * Y;

        return colorMix1;
    };

    /*
     *  This method converts RGB value with color mixing matrix, and returns it as float[3].
     */
     //void ComputeProjectorRGB(float cam[], float prj[])
     //{
     //    cv::Mat M(3, 4, CV_64FC1);  // color mixing matrix
     //    cv::Mat P(3, 1, CV_64FC1);  // projector's RGB
     //    cv::Mat C(4, 1, CV_64FC1);  // camera's RGB

     //    for (int r = 0; r < 3; r++) {
     //        for (int c = 0; c < 4; c++) {
     //            M.at<double>(r, c) = col_mix_mat[c + r * 4];
     //        }
     //    }

     //    C.at<double>(0, 0) = cam[0];
     //    C.at<double>(1, 0) = cam[1];
     //    C.at<double>(2, 0) = cam[2];
     //    C.at<double>(3, 0) = 1.0;

     //    P = M * C;

     //    for (int i = 0; i < 3; i++) {
     //        prj[i] = P.at<double>(i, 0);
     //    }
     //};
};