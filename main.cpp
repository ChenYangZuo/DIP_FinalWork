#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>
/*---------------------------------------------------------------
在连续的视频中对火焰及水柱的轨迹检测，效果如图。
提示:火焰可利用亮度和颜色,水柱的轨迹需要先用背景差分获得水柱的连通域，然后利用
连通域上的像素点进行曲线的拟合，水枪的位置视为已知，即可以手动活动坐标。
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
    Mat src,src_gray,bgMat,subMat,subMat_b,src_hsv,mask,mask1,mask2;
    Mat labelsMat;//标签号
    Mat statsMat;//状态矩阵
    Mat centroidsMat;//连通域中心
    vector<vector<Point>> dst;

    int count = 0;
//    int fps = (int)cap.get(CAP_PROP_FPS);
//    int totalFps = (int)cap.get(CAP_PROP_FRAME_COUNT);
//    cout << "FPS:" << fps << endl;
//    cout << "Total FPS:" << totalFps << endl;

    while(true){
        cout << count << endl;

        bool rSucess = cap.read(src);
        if(!rSucess){
            cout << "FINISH" << endl;
            break;
        }

        //---------------火焰识别---------------
        cvtColor(src,src_gray,COLOR_BGR2GRAY);
        cvtColor(src,src_hsv,COLOR_BGR2HSV);
        inRange(src_hsv,Scalar(150,43,130),Scalar(180,255,255),mask1);
        inRange(src_hsv,Scalar(0,43,130),Scalar(10,255,255),mask2);

        mask = mask1 + mask2;

        findContours(mask,dst,RETR_EXTERNAL,CHAIN_APPROX_NONE);

        for(int i=0;i<dst.size();i++) {
            RotatedRect rbox = minAreaRect(dst[i]);
            Point2f vtx[4];

            if (rbox.size.width > 50) {
                rbox.points(vtx);
                for (int j = 0; j < 4; j++) {
                    line(src, vtx[j], vtx[j < 3 ? j + 1 : 0], Scalar(0, 0, 255), 2);
                }
            }
        }

        //---------------水流识别---------------
        //69fps-86fps
        if(count == 0){
            src_gray.copyTo(bgMat);
        }
        else{
            absdiff(src_gray,bgMat,subMat);
            threshold(subMat,subMat_b,50,255,CV_THRESH_BINARY);
            int cnt = connectedComponentsWithStats(subMat_b,labelsMat,statsMat,centroidsMat);
            for(int i=0;i<cnt;i++){
                Rect box;
                box.x = statsMat.at<int>(i,0);
                box.y = statsMat.at<int>(i,1);
                box.width = statsMat.at<int>(i,2);
                box.height = statsMat.at<int>(i,3);
                if(box.area()>500)
                    rectangle(src,box,CV_RGB(0,255,0),1,8,0);
            }



            imshow("src",src);
            imshow("sub",subMat_b);
//            imwrite("../pic/"+to_string(count)+".jpg",subMat_b);
//            imshow("redMask",mask);
            waitKey(40);
        }
        count++;

    }


    return 0;
}

