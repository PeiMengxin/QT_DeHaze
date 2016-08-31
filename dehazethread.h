#ifndef DEHAZETHREAD_H
#define DEHAZETHREAD_H

#include <QObject>
#include <QThread>
#include "dehazeclass.h"
#include <QMessageBox>
#include <QDebug>
#include <QMetaType>

class DehazeThread : public QThread
{
    Q_OBJECT

public:
    DehazeThread();

    void setImage(const cv::Mat img);

    void run();
    void stop();

private:
     DeHaze dehaze;

private slots:
     void startDehaze(cv::Mat img);

signals:
     void readyResult(cv::Mat dst);
     void readyResultStr(QString dst);
};

#endif // DEHAZETHREAD_H
