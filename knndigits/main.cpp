#pragma warning(disable: 4819)

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

Mat img;
Point ptPrev(-1, -1);

void on_mouse(int event, int x, int y, int flags, void*)
{
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		ptPrev = Point(-1, -1);
	else if (event == EVENT_LBUTTONDOWN)
		ptPrev = Point(x, y);
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		Point pt(x, y);
		if (ptPrev.x < 0)
			ptPrev = pt;
		line(img, ptPrev, pt, Scalar::all(255), 40, LINE_AA, 0);
		ptPrev = pt;

		imshow("img", img);
	}
}

int main()
{
	Mat digits = imread("digits.png", IMREAD_GRAYSCALE);

	if (digits.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat train_images, train_labels;

	for (int j = 0; j < 50; j++) {
		for (int i = 0; i < 100; i++) {
			Mat roi, roi_float, roi_flatten;
			roi = digits(Rect(i * 20, j * 20, 20, 20)).clone();
			roi.convertTo(roi_float, CV_32F);
			roi_flatten = roi_float.reshape(1, 1);

			train_images.push_back(roi_flatten);
			train_labels.push_back(j / 5);
		}
	}

	// 훈련 데이터를 이용하여 KNN 훈련
	Ptr<KNearest> knn = KNearest::create();
	knn->train(train_images, ROW_SAMPLE, train_labels);

	// 입력 이미지 생성
	img = Mat::zeros(400, 400, CV_8U);

	imshow("img", img);
	setMouseCallback("img", on_mouse, 0);

	while (1) {
		int c = waitKey(0);
		
		if (c == 27) {
			break;
		} else if (c == ' ') {
			Mat img_resize, img_blur, img_float;
			Mat test_img, res_float, res_int;

			resize(img, img_resize, Size(20, 20));
			GaussianBlur(img_resize, img_blur, Size(0, 0), 1.);
			//imshow("img_blur", img_blur);

			img_blur.convertTo(img_float, CV_32F);
			test_img = img_float.reshape(1, 1);

			knn->findNearest(test_img, 3, res_float);
			res_float.convertTo(res_int, CV_32S);

			cout << res_int.at<int>(0, 0) << endl;
			img = Mat::zeros(400, 400, CV_8U);
			imshow("img", img);
		} else if (c == 'c') {
			img = Mat::zeros(400, 400, CV_8U);
			imshow("img", img);
		}
	}

	return 0;
}
