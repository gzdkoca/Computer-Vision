#include <iostream>
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp> 
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

// Feature extraction and image completion with ORB

int main(int argc, char** argv) {

    // 1. loading the corrupted image and the patches
    Mat image_cor = imread("C:/Users/gozde/Desktop/scrovegni/scrovegni/image_to_complete.jpg");
    Mat patch_0 = imread("C:/Users/gozde/Desktop/scrovegni/scrovegni/patch_0.jpg");
    Mat patch_1 = imread("C:/Users/gozde/Desktop/scrovegni/scrovegni/patch_1.jpg");
    Mat patch_2 = imread("C:/Users/gozde/Desktop/scrovegni/scrovegni/patch_2.jpg");
    Mat patch_3 = imread("C:/Users/gozde/Desktop/scrovegni/scrovegni/patch_3.jpg");

    /*
    // displaying the corrupted image and the patches
    cv::namedWindow("Corrupted Image", WINDOW_NORMAL);
    imshow("Corrupted Image", image_cor);
    imshow("Patch 0", patch_0);
    imshow("Patch 1", patch_1);
    imshow("Patch 2", patch_2);
    imshow("Patch 3", patch_3);
    */

    // 2. Extracting features of the image
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    
    orb->setMaxFeatures(50000); // maximum number of features are set to a higher value to get more features and imrove matching
    std::vector<cv::KeyPoint> keypoints_img, keypoints_img_cor, keypoints_0, keypoints_1, keypoints_2, keypoints_3;
    cv::Mat image_descriptor, image_cor_descriptor, patch0_descriptor, patch1_descriptor, patch2_descriptor, patch3_descriptor;

    orb->detectAndCompute(image_cor, cv::Mat(), keypoints_img_cor, image_cor_descriptor);
    orb->detectAndCompute(patch_0, cv::Mat(), keypoints_0, patch0_descriptor);
    orb->detectAndCompute(patch_1, cv::Mat(), keypoints_1, patch1_descriptor);
    orb->detectAndCompute(patch_2, cv::Mat(), keypoints_2, patch2_descriptor);
    orb->detectAndCompute(patch_3, cv::Mat(), keypoints_3, patch3_descriptor);

    // drawing keypoints for corrupted image and patches
    cv::Mat image_cor_orb, patch_0_orb, patch_1_orb, patch_2_orb, patch_3_orb;

    cv::drawKeypoints(image_cor, keypoints_img_cor, image_cor_orb);
    cv::drawKeypoints(patch_0, keypoints_0, patch_0_orb);
    cv::drawKeypoints(patch_1, keypoints_1, patch_1_orb);
    cv::drawKeypoints(patch_2, keypoints_2, patch_2_orb);
    cv::drawKeypoints(patch_3, keypoints_3, patch_3_orb);

    // displaying keypoints for corrupted image and patches
    cv::namedWindow("output", WINDOW_NORMAL);
    imshow("output", image_cor_orb);
    imshow("Patch 0 SHIFT", patch_0_orb);
    imshow("Patch 1 SHIFT", patch_1_orb);
    imshow("Patch 2 SHIFT", patch_2_orb);
    imshow("Patch 3 SHIFT", patch_3_orb);

    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Coputing the matches
    std::vector<cv::DMatch> matches0, matches1, matches2, matches3;
    matcher.match(image_cor_descriptor, patch0_descriptor, matches0);
    matcher.match(image_cor_descriptor, patch1_descriptor, matches1);
    matcher.match(image_cor_descriptor, patch2_descriptor, matches2);
    matcher.match(image_cor_descriptor, patch3_descriptor, matches3);
 
    // patch0
    std::vector<cv::DMatch> refinedMatches0;
    double distanceThreshold = 10;  // setting up a threshold
    for (const auto& match : matches0)
    {
        if (match.distance < distanceThreshold)
            refinedMatches0.push_back(match);
    }

    // patch1
    std::vector<cv::DMatch> refinedMatches1;
    for (const auto& match : matches1)
    {
        if (match.distance < distanceThreshold)
            refinedMatches1.push_back(match);
    }

    // patch2
    std::vector<cv::DMatch> refinedMatches2;
    for (const auto& match : matches2)
    {
        if (match.distance < distanceThreshold)
            refinedMatches2.push_back(match);
    }

    // patch3
    std::vector<cv::DMatch> refinedMatches3;
    for (const auto& match : matches3)
    {
        if (match.distance < distanceThreshold)
            refinedMatches3.push_back(match);
    }

    // drawing matches
    cv::Mat matchImage0, matchImage1, matchImage2, matchImage3;
    cv::drawMatches(image_cor, keypoints_img_cor, patch_0, keypoints_0, refinedMatches0, matchImage0);
    cv::drawMatches(image_cor, keypoints_img_cor, patch_1, keypoints_1, refinedMatches1, matchImage1);
    cv::drawMatches(image_cor, keypoints_img_cor, patch_2, keypoints_2, refinedMatches2, matchImage2);
    cv::drawMatches(image_cor, keypoints_img_cor, patch_3, keypoints_3, refinedMatches3, matchImage3);
    
    cv::namedWindow("Matches0", WINDOW_NORMAL);
    cv::namedWindow("Matches1", WINDOW_NORMAL);
    cv::namedWindow("Matches2", WINDOW_NORMAL);
    cv::namedWindow("Matches3", WINDOW_NORMAL);

    cv::imshow("Matches0", matchImage0);
    cv::imshow("Matches1", matchImage1);
    cv::imshow("Matches2", matchImage2);
    cv::imshow("Matches3", matchImage3);

 
    // 5. Refining the matches
    
    std::vector<cv::Point2f> points01, points02, points11, points12, points21, points22, points31, points32;
    // patch0
    for (int i = 0; i < refinedMatches0.size(); i++) {
        points01.push_back(keypoints_img_cor[refinedMatches0[i].queryIdx].pt);
        points02.push_back(keypoints_0[refinedMatches0[i].trainIdx].pt);
    }
    // patch1
    for (int i = 0; i < refinedMatches1.size(); i++) {
        points11.push_back(keypoints_img_cor[refinedMatches1[i].queryIdx].pt);
        points12.push_back(keypoints_1[refinedMatches1[i].trainIdx].pt);
    }
    // patch2
    for (int i = 0; i < refinedMatches2.size(); i++) {
        points21.push_back(keypoints_img_cor[refinedMatches2[i].queryIdx].pt);
        points22.push_back(keypoints_2[refinedMatches2[i].trainIdx].pt);
    }
    // patch3
    for (int i = 0; i < refinedMatches3.size(); i++) {
        points31.push_back(keypoints_img_cor[refinedMatches3[i].queryIdx].pt);
        points32.push_back(keypoints_3[refinedMatches3[i].trainIdx].pt);
    }
      
    // computing homogropies for patches
    cv::Mat homography0 = cv::findHomography(points01, points02, cv::RANSAC);
    cv::Mat homography1 = cv::findHomography(points11, points12, cv::RANSAC);
    cv::Mat homography2 = cv::findHomography(points21, points22, cv::RANSAC);
    cv::Mat homography3 = cv::findHomography(points31, points32, cv::RANSAC);   

    // overlaying patches with corrupted image
    cv::Mat result = image_cor.clone();
    cv::warpPerspective(image_cor, result, homography0, image_cor.size());
    cv::Mat roi(result, cv::Rect(0, 0, patch_0.cols, patch_0.rows));
    patch_0.copyTo(roi);
    
    cv::warpPerspective(image_cor, result, homography1, image_cor.size());
    cv::Mat roi1(result, cv::Rect(0, 0, patch_1.cols, patch_1.rows));
    patch_1.copyTo(roi1);

    cv::warpPerspective(image_cor, result, homography2, image_cor.size());
    cv::Mat roi2(result, cv::Rect(0, 0, patch_2.cols, patch_2.rows));
    patch_2.copyTo(roi2);
    
    cv::warpPerspective(image_cor, result, homography3, image_cor.size()); 
    cv::Mat roi3(result, cv::Rect(0, 0, patch_3.cols, patch_3.rows));  
    patch_3.copyTo(roi3);
    cv::namedWindow("Overlayed Image - ORB", WINDOW_NORMAL); 
    imshow("Overlayed Image - ORB", result); 
    
    cv::waitKey(0);
    return EXIT_SUCCESS;

}


