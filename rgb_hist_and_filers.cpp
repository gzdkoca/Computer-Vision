#include <iostream>
#include <stdio.h>
#include <opencv2/core/mat.hpp> 
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/highgui.hpp> 
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    //PART 1
    //1. loading the image
    string image_path = samples::findFile("C:/Users/gozde/Desktop/lab2_code_and_data/data/underexposed.png");
    Mat src = imread(image_path, IMREAD_COLOR);

    vector<Mat> bgr_planes;
    split(src, bgr_planes); //to split the image into channels
    int histSize = 256;
    float range[] = { 0, 256 }; 
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;

    //2. RGB histogram
    //calulating histogram for RGB channels seperately for the origina source image
    cv::calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat hist_b(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat hist_g(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat hist_r(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    //normalizing
    cv::normalize(b_hist, b_hist, 0, hist_b.rows, NORM_MINMAX, -1, Mat());
    cv::normalize(g_hist, g_hist, 0, hist_g.rows, NORM_MINMAX, -1, Mat());
    cv::normalize(r_hist, r_hist, 0, hist_r.rows, NORM_MINMAX, -1, Mat());


    for (int i = 1; i < histSize; i++)
    {
        line(hist_b, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(hist_g, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(hist_r, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }

    //displaying the soruce image and RGB histograms
    cv::imshow("Source image", src);
    cv::imshow("Histogram for blue channel", hist_b);
    cv::imshow("Histogram for green channel", hist_g);
    cv::imshow("Histogram for red channel", hist_r);


    ///////////////////////////////////////////////////////////////////////

    //3. Equilazing RGB seperately 

    Mat b_hist_eq, g_hist_eq, r_hist_eq;

    cv::equalizeHist(bgr_planes[0], b_hist_eq); //equalizing blue channel
    cv::equalizeHist(bgr_planes[1], r_hist_eq); //equalizing green channel
    cv::equalizeHist(bgr_planes[2], g_hist_eq); //equalizing red channel

    //creating new plane
    vector<Mat> bgr_planes_eq(3);
    Mat src_rgb_eq;
    
    //copying equalized hists to the new plane
    b_hist_eq.copyTo(bgr_planes_eq[0]); //blue hist to bgr_planes_eq[0]
    g_hist_eq.copyTo(bgr_planes_eq[1]); //green hist to bgr_planes_eq[1]
    r_hist_eq.copyTo(bgr_planes_eq[2]); //red hist to bgr_planes_eq[2]
    cv::merge(bgr_planes_eq, src_rgb_eq);  //rgb equialized image

    Mat eq_b_hist, eq_g_hist, eq_r_hist;
    
    //calculating histogram
    cv::calcHist(&b_hist_eq, 1, 0, Mat(), eq_b_hist, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&g_hist_eq, 1, 0, Mat(), eq_g_hist, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&r_hist_eq, 1, 0, Mat(), eq_r_hist, 1, &histSize, histRange, uniform, accumulate);

    Mat hist_b_eq(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat hist_g_eq(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat hist_r_eq(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    //normalizing
    cv::normalize(eq_b_hist, eq_b_hist, 0, hist_b_eq.rows, NORM_MINMAX, -1, Mat());
    cv::normalize(eq_g_hist, eq_g_hist, 0, hist_g_eq.rows, NORM_MINMAX, -1, Mat());
    cv::normalize(eq_r_hist, eq_r_hist, 0, hist_r_eq.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(hist_b_eq, Point(bin_w * (i - 1), hist_h - cvRound(eq_b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(eq_b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(hist_g_eq, Point(bin_w * (i - 1), hist_h - cvRound(eq_g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(eq_g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(hist_r_eq, Point(bin_w * (i - 1), hist_h - cvRound(eq_r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(eq_r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }

    //4. Displaying equalized image and histograms
    cv::imshow("RGB equalized image", src_rgb_eq);
    cv::imshow("Equalized histogram for blue channel", hist_b_eq);
    cv::imshow("Equalized histogram for green channel", hist_g_eq);
    cv::imshow("Equalized histogram for red channel", hist_r_eq);

    ///////////////////////////////////////////

    //5. Equalizing Luminance channel
    
    Mat hist_eq_image;
    cv::cvtColor(src, hist_eq_image, COLOR_BGR2Lab); //converting the image from BGR to Lab color space

    //extracting the L channel
    vector<Mat> image_planes(3);
    cv::split(hist_eq_image, image_planes);  //Luminance(L) channel in image_planes[0] (L-a-b)
     
    Mat image_hist_eq;
    cv::equalizeHist(image_planes[0], image_hist_eq); //equalizing only the L channel

    image_hist_eq.copyTo(image_planes[0]);
    cv::merge(image_planes, hist_eq_image); 

    //converting back to RGB
    cv::Mat image_eq_l;
    cv::cvtColor(hist_eq_image, image_eq_l, COLOR_Lab2BGR);

    //displaying the image with equalized only the luminance (L) channel. 
    cv::imshow("Image Luminance equalized", image_eq_l);

    //histogram
    Mat eq_b_hist_l, eq_g_hist_l, eq_r_hist_l;

    //calculating histogram
    cv::calcHist(&image_planes[0], 1, 0, Mat(), eq_b_hist_l, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&image_planes[1], 1, 0, Mat(), eq_g_hist_l, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&image_planes[2], 1, 0, Mat(), eq_r_hist_l, 1, &histSize, histRange, uniform, accumulate);

    Mat hist_b_eq_l(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat hist_g_eq_l(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat hist_r_eq_l(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    cv::normalize(eq_b_hist_l, eq_b_hist_l, 0, hist_b_eq_l.rows, NORM_MINMAX, -1, Mat());
    cv::normalize(eq_g_hist_l, eq_g_hist_l, 0, hist_g_eq_l.rows, NORM_MINMAX, -1, Mat());
    cv::normalize(eq_r_hist_l, eq_r_hist_l, 0, hist_r_eq_l.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(hist_b_eq_l, Point(bin_w * (i - 1), hist_h - cvRound(eq_b_hist_l.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(eq_b_hist_l.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(hist_g_eq_l, Point(bin_w * (i - 1), hist_h - cvRound(eq_g_hist_l.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(eq_g_hist_l.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(hist_r_eq_l, Point(bin_w * (i - 1), hist_h - cvRound(eq_r_hist_l.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(eq_r_hist_l.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }

    //displaying equalized histograms
    cv::imshow("Equalized histogram for blue channel luminance", hist_b_eq_l);
    cv::imshow("Equalized histogram for green channel luminance", hist_g_eq_l);
    cv::imshow("Equalized histogram for red channel luminance", hist_r_eq_l);
    

    //PART 2: Image Filtering
    //median blur
    Mat src_median_blur;
    cv::medianBlur(image_eq_l, src_median_blur, (5,5));
    cv::imshow("Median blur", src_median_blur);

    //gaussian blur
    Mat src_gaussian_blur;
    cv::GaussianBlur(image_eq_l, src_gaussian_blur, Size(5, 5), 0);
    cv::imshow("Gaussian blur", src_gaussian_blur);

    //bilateral filter
    Mat scr_bilateral_filter;
    cv::bilateralFilter(image_eq_l, scr_bilateral_filter, 5, 70, 30);
    cv::imshow("Bilateral filter", scr_bilateral_filter);
    
    cv::waitKey(0);
    return EXIT_SUCCESS;
}



    