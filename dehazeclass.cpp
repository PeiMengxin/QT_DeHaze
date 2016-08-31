#include "dehazeclass.h"

using namespace cv;
using namespace std;

DeHaze::DeHaze()
{
	nsize = 15;
	Max_A = 230;
	r_filter = 120;
	w = 0.95;
	eps = 0.01;
	lambda = 0.001;
	Min_t = 0.1;
	gifflag = GIF_FLAG_WGIF;
	autoflag = AUTO_FLAG_AUTOCONTRAST;
}

DeHaze::DeHaze(
	int insize = 15,
	double iw = 0.95,
	int iMax_A = 230, 
	double iMin_t = 0.1, 
	int ir_filter = 120, 
	double ieps = 0.01, 
	double ilambda = 0.001,
	int igifflag = GIF_FLAG_WGIF, 
	int iautoflag = AUTO_FLAG_AUTOCONTRAST)
{
	nsize = insize;
	Max_A = iMax_A;
	r_filter = ir_filter;
	w = iw;
	eps = ieps;
	lambda = ilambda;
	Min_t = iMin_t;
	gifflag = igifflag;
	autoflag = iautoflag;
}

DeHaze::~DeHaze()
{
	src.~Mat();
}

void DeHaze::loadimage(const string& filename)
{
	src = imread(filename);
}

void DeHaze::loadimage(cv::Mat img)
{
	img.copyTo(src);
}

cv::Mat DeHaze::dehaze()
{
	int Height = src.rows;
	int Width = src.cols;
	if (Height == 0)
	{
		Mat img;
		return img;
	}

	Mat img_gray(Height, Width, CV_8UC1);
	cvtColor(src, img_gray, CV_BGR2GRAY);//src-->gray，求原图的灰度图
	
	Mat img_channelmin(Height, Width, CV_8UC1);
	Mat img_channelmin_low(Height / 2, Width / 2, CV_8UC1);
	MinOfChannel(src, img_channelmin);//求每个像素点BGR3个通道中的最小值，并存储在矩阵img_channelmin

	Mat img_dark(Height, Width, CV_8UC1);
	Mat img_dark_low(Height / 2, Width / 2, CV_8UC1);

	resize(img_channelmin, img_channelmin_low, cvSize(Width / 2, Height / 2));//下采样

	//MinFilter(img_channelmin, img_dark, nsize);//求暗通道图，即对img_channelmin进行最小值滤波
	MinFilter(img_channelmin_low, img_dark_low, nsize);

	resize(img_dark_low, img_dark, cvSize(Width, Height));//上采样
	
	int A_BGR[3] = { 0 };
	getA(src, img_dark, nsize, A_BGR, Max_A);//求BGR每个通道的大气光强

	Mat img_transmassion(Height, Width, CV_8UC1);
	getT(img_dark, img_transmassion, w);//估计透射率

	Mat img_t_filter(Height, Width, CV_8UC1);
	Mat img_recover(Height, Width, CV_8UC3);

	switch (gifflag)
	{
	case GIF_FLAG_GIF:
	{
		img_t_filter = guidedFilter(img_gray, img_transmassion, r_filter, eps);//对透射率图进行导向滤波，得到更精细的透射率图
		recoverImg(src, img_recover, img_t_filter, A_BGR, Min_t);//对原图进行修复
		break;
	}
	case GIF_FLAG_WGIF:
	{
		img_t_filter = weightguidedFilter(img_gray, img_transmassion, r_filter, 0.01, 0.001);
		recoverImg(src, img_recover, img_t_filter, A_BGR, Min_t);//对原图进行修复
		break;
	}
	default:
		break;
	}

	Mat rec;

	switch (autoflag)
	{
	case AUTO_FLAG_AUTOCOLOR:
	{
		rec = autocolor(img_recover);
		break;
	}
	case AUTO_FLAG_AUTOCONTRAST:
	{
		rec = autocontrost(img_recover);
		break;
	}
	default:
	{
		rec = img_recover;
		break;
	}
	}

	return rec;

}

cv::Mat DeHaze::autocontrost(cv::Mat src)
{
	//进行自动对比度校正
	Mat matface;
	src.copyTo(matface);
	double HistRed[256] = { 0 };
	double HistGreen[256] = { 0 };
	double HistBlue[256] = { 0 };
	int bluemap[256] = { 0 };
	int redmap[256] = { 0 };
	int greenmap[256] = { 0 };

	double dlowcut = 0.1;
	double dhighcut = 0.1;
	for (int i = 0; i < matface.rows; i++)
	{
		for (int j = 0; j < matface.cols; j++)
		{
			int iblue = matface.at<Vec3b>(i, j)[0];
			int igreen = matface.at<Vec3b>(i, j)[1];
			int ired = matface.at<Vec3b>(i, j)[2];
			HistBlue[iblue]++;
			HistGreen[igreen]++;
			HistRed[ired]++;
		}
	}
	int PixelAmount = (int)(matface.rows*matface.cols);
	int T_L = (int)(PixelAmount*dlowcut*0.01);
	int T_H = (int)(PixelAmount*dhighcut*0.01);
	int isum = 0;
	// blue
	int iminblue = 0; int imaxblue = 0;
	for (int y = 0; y < 256; y++)//这两个操作我基本能够了解了
	{
		isum = (int)(isum + HistBlue[y]);
		if (isum >= T_L)
		{
			iminblue = y;
			break;
		}
	}
	isum = 0;
	for (int y = 255; y >= 0; y--)
	{
		isum = (int)(isum + HistBlue[y]);
		if (isum >= T_H)
		{
			imaxblue = y;
			break;
		}
	}
	//red
	isum = 0;
	int iminred = 0; int imaxred = 0;
	for (int y = 0; y < 256; y++)//这两个操作我基本能够了解了
	{
		isum = (int)(isum + HistRed[y]);
		if (isum >= T_L)
		{
			iminred = y;
			break;
		}
	}
	isum = 0;
	for (int y = 255; y >= 0; y--)
	{
		isum = (int)(isum + HistRed[y]);
		if (isum >= T_H)
		{
			imaxred = y;
			break;
		}
	}
	//green
	isum = 0;
	int imingreen = 0; int imaxgreen = 0;
	for (int y = 0; y < 256; y++)//这两个操作我基本能够了解了
	{
		isum = (int)(isum + HistGreen[y]);
		if (isum >= T_L)
		{
			imingreen = y;
			break;
		}
	}
	isum = 0;
	for (int y = 255; y >= 0; y--)
	{
		isum = (int)(isum + HistGreen[y]);
		if (isum >= T_H)
		{
			imaxgreen = y;
			break;
		}
	}
	//自动对比度
	int imin = 255; int imax = 0;
	if (imin > iminblue)
		imin = iminblue;
	if (imin > iminred)
		imin = iminred;
	if (imin > imingreen)
		imin = imingreen;
	iminblue = imin;
	imingreen = imin;
	iminred = imin;
	if (imax < imaxblue)
		imax = imaxblue;
	if (imax < imaxgreen)
		imax = imaxgreen;
	if (imax < imaxred)
		imax = imaxred;
	imaxred = imax;
	imaxgreen = imax;
	imaxblue = imax;
	/////////////////
	//blue
	for (int y = 0; y < 256; y++)
	{
		if (y <= iminblue)
		{
			bluemap[y] = 0;
		}
		else
		{
			if (y > imaxblue)
			{
				bluemap[y] = 255;
			}
			else
			{
				//  BlueMap(Y) = (Y - MinBlue) / (MaxBlue - MinBlue) * 255      '线性隐射
				float ftmp = (float)(y - iminblue) / (imaxblue - iminblue);
				bluemap[y] = (int)(ftmp * 255);
			}
		}

	}
	//red
	for (int y = 0; y < 256; y++)
	{
		if (y <= iminred)
		{
			redmap[y] = 0;
		}
		else
		{
			if (y > imaxred)
			{
				redmap[y] = 255;
			}
			else
			{
				//  BlueMap(Y) = (Y - MinBlue) / (MaxBlue - MinBlue) * 255      '线性隐射
				float ftmp = (float)(y - iminred) / (imaxred - iminred);
				redmap[y] = (int)(ftmp * 255);
			}
		}

	}
	//green
	for (int y = 0; y < 256; y++)
	{
		if (y <= imingreen)
		{
			greenmap[y] = 0;
		}
		else
		{
			if (y > imaxgreen)
			{
				greenmap[y] = 255;
			}
			else
			{
				//  BlueMap(Y) = (Y - MinBlue) / (MaxBlue - MinBlue) * 255      '线性隐射
				float ftmp = (float)(y - imingreen) / (imaxgreen - imingreen);
				greenmap[y] = (int)(ftmp * 255);
			}
		}

	}
	//查表
	for (int i = 0; i < matface.rows; i++)
	{
		for (int j = 0; j < matface.cols; j++)
		{
			matface.at<Vec3b>(i, j)[0] = bluemap[matface.at<Vec3b>(i, j)[0]];
			matface.at<Vec3b>(i, j)[1] = greenmap[matface.at<Vec3b>(i, j)[1]];
			matface.at<Vec3b>(i, j)[2] = redmap[matface.at<Vec3b>(i, j)[2]];
		}
	}
	return matface;
}

cv::Mat DeHaze::autocolor(cv::Mat src)
{
	//进行自动色阶校正
	Mat matface;
	src.copyTo(matface);
	double HistRed[256] = { 0 };
	double HistGreen[256] = { 0 };
	double HistBlue[256] = { 0 };
	int bluemap[256] = { 0 };
	int redmap[256] = { 0 };
	int greenmap[256] = { 0 };

	double dlowcut = 0.1;
	double dhighcut = 0.1;
	for (int i = 0; i < matface.rows; i++)
	{
		for (int j = 0; j < matface.cols; j++)
		{
			int iblue = matface.at<Vec3b>(i, j)[0];
			int igreen = matface.at<Vec3b>(i, j)[1];
			int ired = matface.at<Vec3b>(i, j)[2];
			HistBlue[iblue]++;
			HistGreen[igreen]++;
			HistRed[ired]++;
		}
	}
	int PixelAmount = matface.rows*matface.cols;
	int T_L = (int)(PixelAmount*dlowcut*0.01);
	int T_H = (int)(PixelAmount*dhighcut*0.01);
	int isum = 0;
	// blue
	int iminblue = 0; int imaxblue = 0;
	for (int y = 0; y < 256; y++)//这两个操作我基本能够了解了
	{
		isum = (int)(isum + HistBlue[y]);
		if (isum >= T_L)
		{
			iminblue = y;
			break;
		}
	}
	isum = 0;
	for (int y = 255; y >= 0; y--)
	{
		isum = (int)(isum + HistBlue[y]);
		if (isum >= T_H)
		{
			imaxblue = y;
			break;
		}
	}
	//red
	isum = 0;
	int iminred = 0; int imaxred = 0;
	for (int y = 0; y < 256; y++)//这两个操作我基本能够了解了
	{
		isum = (int)(isum + HistRed[y]);
		if (isum >= T_L)
		{
			iminred = y;
			break;
		}
	}
	isum = 0;
	for (int y = 255; y >= 0; y--)
	{
		isum = (int)(isum + HistRed[y]);
		if (isum >= T_H)
		{
			imaxred = y;
			break;
		}
	}
	//green
	isum = 0;
	int imingreen = 0; int imaxgreen = 0;
	for (int y = 0; y < 256; y++)//这两个操作我基本能够了解了
	{
		isum = (int)(isum + HistGreen[y]);
		if (isum >= T_L)
		{
			imingreen = y;
			break;
		}
	}
	isum = 0;
	for (int y = 255; y >= 0; y--)
	{
		isum = (int)(isum + HistGreen[y]);
		if (isum >= T_H)
		{
			imaxgreen = y;
			break;
		}
	}
	//自动色阶

	//blue
	for (int y = 0; y < 256; y++)
	{
		if (y <= iminblue)
		{
			bluemap[y] = 0;
		}
		else
		{
			if (y > imaxblue)
			{
				bluemap[y] = 255;
			}
			else
			{
				//  BlueMap(Y) = (Y - MinBlue) / (MaxBlue - MinBlue) * 255      '线性隐射
				float ftmp = (float)(y - iminblue) / (imaxblue - iminblue);
				bluemap[y] = (int)(ftmp * 255);
			}
		}

	}
	//red
	for (int y = 0; y < 256; y++)
	{
		if (y <= iminred)
		{
			redmap[y] = 0;
		}
		else
		{
			if (y > imaxred)
			{
				redmap[y] = 255;
			}
			else
			{
				//  BlueMap(Y) = (Y - MinBlue) / (MaxBlue - MinBlue) * 255      '线性隐射
				float ftmp = (float)(y - iminred) / (imaxred - iminred);
				redmap[y] = (int)(ftmp * 255);
			}
		}

	}
	//green
	for (int y = 0; y < 256; y++)
	{
		if (y <= imingreen)
		{
			greenmap[y] = 0;
		}
		else
		{
			if (y > imaxgreen)
			{
				greenmap[y] = 255;
			}
			else
			{
				//  BlueMap(Y) = (Y - MinBlue) / (MaxBlue - MinBlue) * 255      '线性隐射
				float ftmp = (float)(y - imingreen) / (imaxgreen - imingreen);
				greenmap[y] = (int)(ftmp * 255);
			}
		}

	}
	//查表
	for (int i = 0; i < matface.rows; i++)
	{
		for (int j = 0; j < matface.cols; j++)
		{
			matface.at<Vec3b>(i, j)[0] = bluemap[matface.at<Vec3b>(i, j)[0]];
			matface.at<Vec3b>(i, j)[1] = greenmap[matface.at<Vec3b>(i, j)[1]];
			matface.at<Vec3b>(i, j)[2] = redmap[matface.at<Vec3b>(i, j)[2]];
		}
	}
	return matface;
}

void DeHaze::getA(cv::Mat src, cv::Mat dark, int nsize, int *A_BGR, int thread_A)
{
	double min_dark = 0;
	double max_dark = 0;
	Point min_loc;
	Point max_loc;

	Rect rect;
	rect.x = 0;
	rect.y = 0;
	rect.width = dark.cols;
	rect.height = dark.rows;

	cvSetImageROI(&(IplImage)dark, rect);
	minMaxLoc(dark, &min_dark, &max_dark, &min_loc, &max_loc);//在暗通道图像中找到最大值及其位置
	cvResetImageROI(&(IplImage)dark);

	double A_R = 0;
	double A_G = 0;
	double A_B = 0;

	int r = (nsize - 1) / 2;
	rect.x = max_loc.x - r;
	rect.y = max_loc.y - r;
	rect.width = nsize;
	rect.height = nsize;

	Mat Channel_BGR[3];

	split(src, Channel_BGR);

	cvSetImageROI(&(IplImage)Channel_BGR[0], rect);
	minMaxLoc(Channel_BGR[0], &min_dark, &A_B, &min_loc, &max_loc);
	cvResetImageROI(&(IplImage)Channel_BGR[0]);

	cvSetImageROI(&(IplImage)Channel_BGR[1], rect);
	minMaxLoc(Channel_BGR[1], &min_dark, &A_G, &min_loc, &max_loc);
	cvResetImageROI(&(IplImage)Channel_BGR[1]);

	cvSetImageROI(&(IplImage)Channel_BGR[2], rect);
	minMaxLoc(Channel_BGR[2], &min_dark, &A_R, &min_loc, &max_loc);
	cvResetImageROI(&(IplImage)Channel_BGR[2]);

	if (A_B > thread_A)
		A_B = thread_A;
	if (A_G > thread_A)
		A_G = thread_A;
	if (A_R > thread_A)
		A_R = thread_A;
	A_BGR[0] = (uchar)A_B;
	A_BGR[1] = (uchar)A_G;
	A_BGR[2] = (uchar)A_R;
}

void DeHaze::getT(cv::Mat dark, cv::Mat t, double w)
{
	uchar J = 0;
	uchar *prow_dark;
	uchar *prow_t;
	for (int row = 0; row < dark.rows; row++)
	{
		prow_dark = dark.ptr<uchar>(row);
		prow_t = t.ptr<uchar>(row);
		for (int col = 0; col < dark.cols; col++)
		{
			J = prow_dark[col];
			prow_t[col] = (uchar)(255 - w*J);//利用公式得到透射率，注：该透射率图并未归一化（t的值在0~255）
		}
	}
}

cv::Mat DeHaze::guidedFilter(cv::Mat I, cv::Mat p, int r, double ilambda)
{
	cv::Mat _I;
	I.convertTo(_I, CV_64FC1);
	_I = _I / 255;//归一化

	cv::Mat _p;
	p.convertTo(_p, CV_64FC1);
	_p = _p / 255;

	//[hei, wid] = size(I);
	int hei = I.rows;
	int wid = I.cols;

	//N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
	cv::Mat N;
	cv::boxFilter(//方框滤波
		cv::Mat::ones(hei, wid, I.type()),//Input Mat
		N, //Output Mat
		CV_64FC1, //type
		cv::Size(r, r),//滤波核大小
		cv::Point(-1, -1),//默认值Point(-1,-1)表示这个锚点（被平滑的点）在核的中心
		true,//bool normalize，表示是否归一化处理
		BORDER_DEFAULT//边界模式
		);

	//mean_I = boxfilter(I, r) ./ N;
	cv::Mat mean_I;
	cv::boxFilter(_I, mean_I, CV_64FC1, cv::Size(r, r));
	//cv::divide(mean_I, N, mean_I);

	//mean_p = boxfilter(p, r) ./ N;
	cv::Mat mean_p;
	cv::boxFilter(_p, mean_p, CV_64FC1, cv::Size(r, r));
	//cv::divide(mean_p, N, mean_p);

	//mean_Ip = boxfilter(I.*p, r) ./ N;
	cv::Mat mean_Ip;
	cv::boxFilter(_I.mul(_p), mean_Ip, CV_64FC1, cv::Size(r, r));
	//cv::divide(mean_Ip, N, mean_Ip);

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;
	cv::Mat mean_II;
	cv::boxFilter(_I.mul(_I), mean_II, CV_64FC1, cv::Size(r, r));
	//cv::divide(mean_II, N, mean_II);

	//var_I = mean_II - mean_I .* mean_I;
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;	
	cv::Mat a = cov_Ip / (var_I + ilambda);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
	mean_a = mean_a / N;

	//mean_b = boxfilter(b, r) ./ N;
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
	mean_b = mean_b / N;

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
	cv::Mat q = mean_a.mul(_I) + mean_b;

	return q;
}

void DeHaze::MinFilter(cv::Mat src, cv::Mat dst, int nsize)
{
	int r = (nsize - 1) / 2;

	Mat src_ex(src.rows + nsize - 1, src.cols + nsize - 1, CV_8UC1);

	copyMakeBorder(src, src_ex, r, r, r, r, BORDER_DEFAULT);

	unsigned char Value_Min = 0;
	uchar *prow;
	for (int row = 0; row < src.rows; row++)
	{
		prow = dst.ptr<uchar>(row);
		for (int col = 0; col < src.cols; col++)
		{
			Value_Min = getMin(src_ex, row + r, col + r, nsize);
			prow[col] = Value_Min;
		}
	}
}

unsigned char DeHaze::getMin(cv::Mat img, int row, int col, int nsize)
{
	unsigned char min = 0;
	uchar *prow;
	int r = (nsize - 1) / 2;
	int row_L = row - r;
	int row_H = row + r;
	int col_L = col - r;
	int col_H = col + r;
	min = *(img.data + img.step[0] * row + img.step[1] * col);
	for (int i = row_L; i <= row_H; i++)
	{
		prow = img.data + img.step[0] * i;
		for (int j = col_L; j <= col_H; j++)
		{
			if (min > *(prow + j))
			{
				min = *(prow + j);
			}
		}
	}

	return min;
}

void DeHaze::setMin(cv::Mat img, unsigned char Value_Min, int row, int col, int nsize)
{
	int r = (nsize - 1) / 2;
	int row_L = row - r;
	int row_H = row + r;
	int col_L = col - r;
	int col_H = col + r;
	uchar *prow;
	for (int i = row_L; i <= row_H; i++)
	{
		prow = img.ptr<uchar>(i);
		for (int j = col_L; j <= col_H; j++)
		{
			prow[j] = Value_Min;
		}
	}
}

void DeHaze::MinOfChannel(cv::Mat src, cv::Mat dst)
{
	unsigned char Value_R = 0;
	unsigned char Value_G = 0;
	unsigned char Value_B = 0;
	unsigned char Value_Min = 0;

	uchar *pdata = src.ptr<uchar>(0);
	uchar *prow = dst.ptr<uchar>(0);
	for (int row = 0; row < src.rows; row++)
	{
		pdata = src.ptr<uchar>(row);
		prow = dst.ptr<uchar>(row);
		for (int col = 0; col < src.cols; col++)
		{
			// = *(src.data + src.step[0] * row + src.step[1] * col);
			//Value_G = *(src.data + src.step[0] * row + src.step[1] * col + 1);
			//Value_R = *(src.data + src.step[0] * row + src.step[1] * col + 2);
			Value_B = *pdata;
			pdata++;
			Value_G = *pdata;
			pdata++;
			Value_R = *pdata;
			pdata++;
			Value_Min = Value_B;
			if (Value_Min > Value_G)
			{
				Value_Min = Value_G;
			}
			if (Value_Min > Value_R)
			{
				Value_Min = Value_R;
			}

			prow[col] = Value_Min;
		}
	}
}

void DeHaze::recoverImg(cv::Mat src, cv::Mat recover, cv::Mat t_filter, int *A_BGR, double Min_t)
{
	double tx = 0;
	double *pdata;
	uchar *prow_src;
	uchar *prow_recover;
	int temp = 0;
	for (int row = 0; row < src.rows; row++)
	{
		pdata = t_filter.ptr<double>(row);//指向矩阵每一行起始的指针
		prow_src = src.ptr<uchar>(row);
		prow_recover = recover.ptr<uchar>(row);
		for (int col = 0; col < src.cols; col++)
		{
			tx = pdata[col];

			if (tx < Min_t)
				tx = Min_t;//保证透射率>=Min_t

			temp = (int)((*prow_src - A_BGR[0]) / tx) + A_BGR[0];
			if (temp < 0)
			{
				*prow_recover = 0;
			}
			else if (temp > 255)
			{
				*prow_recover = 255;
			}
			else
			{
				*prow_recover = (uchar)temp;
			}
			prow_recover++;
			prow_src++;

			temp = (int)((*prow_src - A_BGR[1]) / tx) + A_BGR[1];
			if (temp < 0)
			{
				*prow_recover = 0;
			}
			else if (temp > 255)
			{
				*prow_recover = 255;
			}
			else
			{
				*prow_recover = (uchar)temp;
			}
			prow_recover++;
			prow_src++;

			temp = (int)((*prow_src - A_BGR[2]) / tx) + A_BGR[2];
			if (temp < 0)
			{
				*prow_recover = 0;
			}
			else if (temp > 255)
			{
				*prow_recover = 255;
			}
			else
			{
				*prow_recover = (uchar)temp;
			}
			prow_recover++;
			prow_src++;
		}
	}
}

cv::Mat DeHaze::weightguidedFilter(cv::Mat I, cv::Mat p, int r, double eps, double lambda)
{
	Mat _I;
	I.convertTo(_I, CV_64FC1);
	_I = _I / 255;//归一化

	Mat _p;
	p.convertTo(_p, CV_64FC1);
	_p = _p / 255;

	//[hei, wid] = size(I);
	int hei = I.rows;
	int wid = I.cols;

	//N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
	Mat N;
	boxFilter(//方框滤波
		Mat::ones(hei, wid, I.type()),//Input Mat
		N, //Output Mat
		CV_64FC1, //type
		Size(r, r),//滤波核大小
		Point(-1, -1),//默认值Point(-1,-1)表示这个锚点（被平滑的点）在核的中心
		true,//bool normalize，表示是否归一化处理
		BORDER_DEFAULT//边界模式
		);
	//mean_I = boxfilter(I, r) ./ N;
	Mat mean_I;
	boxFilter(_I, mean_I, CV_64FC1, Size(r, r));

	Mat mean_Ir3;
	boxFilter(_I, mean_Ir3, CV_64FC1, Size(3, 3));

	//mean_p = boxfilter(p, r) ./ N;
	Mat mean_p;
	boxFilter(_p, mean_p, CV_64FC1, Size(r, r));
	//divide(mean_p, N, mean_p);

	//mean_Ip = boxfilter(I.*p, r) ./ N;
	Mat mean_Ip;
	boxFilter(_I.mul(_p), mean_Ip, CV_64FC1, Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;
	Mat mean_II;
	boxFilter(_I.mul(_I), mean_II, CV_64FC1, Size(r, r));

	Mat mean_IIr3;
	boxFilter(_I.mul(_I), mean_IIr3, CV_64FC1, Size(3, 3));

	//var_I = mean_II - mean_I .* mean_I;
	Mat var_I = mean_II - mean_I.mul(mean_I);

	//var_Ir3 = mean_IIr3 - mean_Ir3 .* mean_Ir3;
	Mat var_Ir3 = mean_IIr3 - mean_Ir3.mul(mean_Ir3);

	Scalar scalar = mean(1 / (var_Ir3 + eps));

	Mat Tp;
	Tp = (var_Ir3 + eps)*scalar.val[0];

	//GaussianBlur(Tp, Tp, Size(5, 5), 0);

	//a = cov_Ip ./ (var_I + lambda/Tp); % Eqn. (5) in the paper;	
	Mat a = cov_Ip / (var_I + lambda / Tp);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
	Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;
	Mat mean_a;
	boxFilter(a, mean_a, CV_64FC1, Size(r, r));
	mean_a = mean_a / N;


	//mean_b = boxfilter(b, r) ./ N;
	Mat mean_b;
	boxFilter(b, mean_b, CV_64FC1, Size(r, r));
	mean_b = mean_b / N;

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
	Mat q = mean_a.mul(_I) + mean_b;
	cvPow(&(IplImage)q, &(IplImage)q, 1.03125);

	return q;
}
