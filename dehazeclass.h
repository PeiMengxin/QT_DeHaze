#ifndef _DEHAZECLASS_H
#define _DEHAZECLASS_H

#include <opencv2\opencv.hpp>

#define GIF_FLAG_GIF 0
#define GIF_FLAG_WGIF 1

#define AUTO_FLAG_NONE 0
#define AUTO_FLAG_AUTOCOLOR 1
#define AUTO_FLAG_AUTOCONTRAST 2


class DeHaze
{
public:
	DeHaze();
	DeHaze(int nsize, double w, int Max_A, double Min_t, int r_filter, double eps, double lambda, int gifflag, int autoflag);
	~DeHaze();

	cv::Mat src;
	int nsize;
	int Max_A;
	int r_filter;
	double w;
	double eps;
	double lambda;
	double Min_t;
	int gifflag;
	int autoflag;

	cv::Mat dehaze();
	void loadimage(const std::string& filename);
	void loadimage(cv::Mat img);

private:
	cv::Mat autocontrost(cv::Mat src);
	cv::Mat autocolor(cv::Mat src);
	void getA(cv::Mat src, cv::Mat dark, int nsize, int *A_BGR, int thread_A);
	void getT(cv::Mat dark, cv::Mat t, double w);
	cv::Mat guidedFilter(cv::Mat I, cv::Mat p, int r, double eps);
	void MinFilter(cv::Mat src, cv::Mat dst, int nsize);
	unsigned char getMin(cv::Mat img, int row, int col, int nsize);
	void setMin(cv::Mat img, unsigned char Value_Min, int row, int col, int nsize);
	void MinOfChannel(cv::Mat src, cv::Mat dst);
	void recoverImg(cv::Mat src, cv::Mat recover, cv::Mat t_filter, int *A_BGR, double Min_t);
	cv::Mat weightguidedFilter(cv::Mat I, cv::Mat p, int r, double eps, double lambda);

};

#endif // !_DEHAZECLASS_H
