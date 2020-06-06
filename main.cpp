#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>
/*---------------------------------------------------------------
在连续的视频中对火焰及水柱的轨迹检测，效果如图。提示火焰可利用亮度和颜
色,水柱的轨迹需要先用背景差分获得水柱的连通域，然后利用连通域上的像素点
进行曲线的拟合，水枪的位置视为已知，即可以手动活动坐标。
-----------------------------------------------------------------
1.视频差分√
2.二值化√
3.连通域
4.分析点 鼠标响应
5.拟合
6.绘图
---------------------------------------------------------------*/

using namespace cv;
using namespace std;


int main(){
    cout << "Hello,World!" << endl;

    VideoCapture cap("./src.mp4");
    Mat src,src_gray,bgMat,subMat,subMat_b;
    int count = 0;

    while(true){
        
        cap >> src;
        cvtColor(src,src_gray,CV_BGR2GRAY);
        if(count == 0){
            src_gray.copyTo(bgMat);
        }
        else{
            absdiff(src_gray,bgMat,subMat);
            threshold(subMat,subMat_b,100,255,CV_THRESH_BINARY);
            imshow("src",src);
            imshow("sub",subMat);
            imshow("dst",subMat_b);
            waitKey(30);
        }
        count++;


    }


    return 0;
}

