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

#if _DEBUG
	HOGDescriptor hog(Size(20, 20), // _winSize
		Size(8, 8),		// _blockSize
		Size(4, 4),		// _blockStride,
		Size(4, 4),		// _cellSize,
		9);				// _nbins,
#else
	HOGDescriptor hog(Size(20, 20), // _winSize
		Size(10, 10),	// _blockSize
		Size(5, 5),		// _blockStride,
		Size(5, 5),		// _cellSize,
		9);				// _nbins,
#endif

	int descriptor_size = hog.getDescriptorSize();
	cout << "Descriptor Size : " << descriptor_size << endl;

	Mat train_hog, train_labels;

	for (int j = 0; j < 50; j++) {
		for (int i = 0; i < 100; i++) {
			Mat roi = digits(Rect(i * 20, j * 20, 20, 20)).clone();

			vector<float> desc;
			hog.compute(roi, desc);

			Mat desc_mat(desc, true);
			train_hog.push_back(desc_mat.t());
			train_labels.push_back(j / 5);
		}
	}

	Ptr<SVM> svm = SVM::create();
	svm->setGamma(0.50625);
	svm->setC(12.5);
	svm->setKernel(SVM::RBF);
	svm->setType(SVM::C_SVC);
	Ptr<TrainData> td = TrainData::create(train_hog, ROW_SAMPLE, train_labels);
	svm->train(td);
//	svm->save("svmdigits.yml");

	// 입력 이미지 생성
	img = Mat::zeros(400, 400, CV_8U);

	imshow("img", img);
	setMouseCallback("img", on_mouse, 0);

	while (1) {
		int c = waitKey(0);

		if (c == 27) {
			break;
		} else if (c == ' ') {
			Mat img_resize, img_blur;
			Mat test_img, res_float, res_int;

			resize(img, img_resize, Size(20, 20));
			GaussianBlur(img_resize, img_blur, Size(0, 0), 1.);
			//imshow("img_blur", img_blur);

			vector<float> desc;
			hog.compute(img_resize, desc);

			Mat desc_mat(desc, true);
			svm->predict(desc_mat.t(), res_float);

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
