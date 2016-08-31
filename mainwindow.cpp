#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QPixmap>
#include <QImage>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_Btn_OpenFile_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("open video"), tr(""), tr("Video Files(*.avi);;Image Files(*.jpg *.bmp);"));

    if(filename.isEmpty())
        return;

    capture.open(filename.toLocal8Bit().data());

    if(!capture.isOpened())
        return;

    capture>>image;

    if(image.empty())
    {
        QMessageBox::warning(this,tr("warning"),tr("image is empty"), QMessageBox::Yes);
    }

    showCvImage(ui->img_lable_src, image);

}

void MainWindow::on_Btn_Start_clicked()
{
    connect(&dehaze_thread,SIGNAL(finished()),&dehaze_thread,SLOT(QObject::deleteLater()));
    connect(&dehaze_thread, SIGNAL(readyResult(cv::Mat)),this,SLOT(handleResult(cv::Mat)));
    connect(this, SIGNAL(nextWork(cv::Mat)), &dehaze_thread, SLOT(startDehaze(cv::Mat)));

    dehaze_thread.setImage(image);
    dehaze_thread.start();

}

void MainWindow::on_Btn_Suspend_clicked()
{
    qDebug()<<(dehaze_thread.currentThreadId());
}

void MainWindow::on_Btn_Stop_clicked()
{
    //dehaze_thread.stop();
    qDebug()<<tr("dehzae thread finished is %1").arg(dehaze_thread.isFinished());
    qDebug()<<tr("dehzae thread running is %1").arg(dehaze_thread.isRunning());
}

void MainWindow::handleResult(cv::Mat _result)
{
    tm.stop();
    qDebug()<<tr("thread work %1").arg(tm.getTimeMilli());

    cv::resize(_result,result,image.size());
    showCvImage(ui->img_lable_src, image);
    showCvImage(ui->img_lable_dst,result);

    capture>>image;

    if(image.data){
        cv::Mat temp;
        cv::resize(image,temp,cv::Size(320,240));
        emit(nextWork(temp));
    }

    tm.reset();
    tm.start();
}

void MainWindow::handleResultStr(QString result)
{
    ui->img_lable_dst->setText(result);
}

void MainWindow::showCvImage(QLabel *lable_, const cv::Mat img)
{
    cv::Mat temp;
    cv::cvtColor(img,temp,CV_BGR2RGB);

    QImage qimg = QImage(temp.data,temp.cols,temp.rows,temp.cols*temp.channels(), QImage::Format_RGB888);

    lable_->setPixmap(QPixmap::fromImage(qimg));
    lable_->update();
}
