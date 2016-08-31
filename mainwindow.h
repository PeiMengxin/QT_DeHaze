#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QLabel>
#include "dehazethread.h"
#include <opencv2/opencv.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_Btn_OpenFile_clicked();

    void on_Btn_Start_clicked();

    void on_Btn_Suspend_clicked();

    void on_Btn_Stop_clicked();

    void handleResult(cv::Mat _result);

    void handleResultStr(QString result);

private:
    void showCvImage(QLabel *lable_, const cv::Mat img);
    Ui::MainWindow *ui;

    cv::Mat image;
    cv::Mat result;
    cv::TickMeter tm;
    cv::VideoCapture capture;

    DehazeThread dehaze_thread;

signals:
    void nextWork(cv::Mat img);
};

#endif // MAINWINDOW_H
