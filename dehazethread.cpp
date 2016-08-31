#include "dehazethread.h"

DehazeThread::DehazeThread()
{
    qRegisterMetaType<cv::Mat>("cv::Mat");
}

void DehazeThread::setImage(const cv::Mat img)
{
    dehaze.loadimage(img);
}

void DehazeThread::run()
{
    cv::TickMeter tm;
    tm.reset();
    tm.start();
    emit(readyResult(dehaze.dehaze()));
    tm.stop();
    qDebug()<<tr("dehaze image %1").arg(tm.getTimeMilli());
}

void DehazeThread::stop()
{
    //this->terminate();
}

void DehazeThread::startDehaze(cv::Mat img)
{

    if(dehaze.src.empty())
        return;
    dehaze.loadimage(img);

    start();
}
