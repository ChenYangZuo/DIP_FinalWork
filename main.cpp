#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>
/*---------------------------------------------------------------
在连续的视频中对火焰及水柱的轨迹检测，效果如图。
提示:火焰可利用亮度和颜色,水柱的轨迹需要先用背景差分获得水柱的连通域，然
后利用连通域上的像素点进行曲线的拟合，水枪的位置视为已知，即可以手动活动
坐标。
-----------------------------------------------------------------
1.视频差分√
2.二值化√
3.连通域√
4.拟合
5.绘图
---------------------------------------------------------------*/

using namespace cv;
using namespace std;

int main(){
    cout << "Hello,World!" << endl;

    VideoCapture cap("./src.mp4");

    Mat src,src_gray,src_hsv,src_sub_Fire;
    Mat bgMat,subMat,subMat_b,subMat_Fire,subMat_b_Fire;
    Mat mask,mask1,mask2;
    Mat labelsMat,labelsMat2;//标签号
    Mat statsMat,statsMat2;//状态矩阵
    Mat centroidsMat,centroidsMat2;//连通域中心
    Mat lastFrame;
    Mat kernel = getStructuringElement(0,Size(3,3));
    Mat kernel1 = getStructuringElement(0,Size(7,7));

    vector<vector<Point>> dst;

    int count = 0;
    int fps = (int)cap.get(CAP_PROP_FPS);
    int totalFps = (int)cap.get(CAP_PROP_FRAME_COUNT);
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "---------------------------------------------" << endl;
    cout << "FRAME_WIDTH:" << frame_width <<endl;
    cout << "FRAME_HEIGHT:" << frame_height <<endl;
    cout << "FPS:" << fps << endl;
    cout << "Total FPS:" << totalFps << endl;
    cout << "---------------------------------------------" << endl;

    while(true){
        bool rSucess = cap.read(src);
        if(!rSucess){
            cout << "FINISH!" << endl;
            break;
        }

        cvtColor(src,src_gray,COLOR_BGR2GRAY);//为水流识别转换灰度图
        
        if(count == 0){
            src_gray.copyTo(bgMat);
        }
        else{
            absdiff(src_gray,bgMat,subMat);//对视频首帧视频差分
            absdiff(src_gray,lastFrame,subMat_Fire);//对视频前帧视频差分

            //对差分后的图像二值化处理
            threshold(subMat,subMat_b,50,255,CV_THRESH_BINARY);
            threshold(subMat_Fire,subMat_b_Fire,50,255,CV_THRESH_BINARY);

            //二值图膨胀运算
            morphologyEx(subMat_b,subMat_b,1,kernel);
            morphologyEx(subMat_b_Fire,subMat_b_Fire,1,kernel1);

            src.copyTo(src_sub_Fire,subMat_b_Fire);//识别火焰所使用的差分图

            //---------------------火焰识别---------------------
            cvtColor(src_sub_Fire,src_hsv,COLOR_BGR2HSV);//为火焰识别转换HSV图
            //opencv中HSV色彩空间中红色位于两侧，因此分别取mask并合并
            inRange(src_hsv,Scalar(150,43,130),Scalar(180,255,255),mask1);
            inRange(src_hsv,Scalar(0,43,130),Scalar(10,255,255),mask2);
            mask = mask1 + mask2;
            morphologyEx(mask,mask,1,kernel1);//膨胀运算

            int cnt = connectedComponentsWithStats(mask,labelsMat2,statsMat2,centroidsMat2);
            for(int i=0;i<cnt;i++) {
                Rect box;
                box.x = statsMat2.at<int>(i,0);
                box.y = statsMat2.at<int>(i,1);
                box.width = statsMat2.at<int>(i,2);
                box.height = statsMat2.at<int>(i,3);
                if (box.width > 15 && box.width < 200) {
                    rectangle(src,box,CV_RGB(255,0,0),1,8,0);
                }
            }
            //---------------------火焰识别---------------------

            //---------------------水流识别---------------------
            int cnt1 = connectedComponentsWithStats(subMat_b,labelsMat,statsMat,centroidsMat);//取连通域
            for(int i=0;i<cnt1;i++){
                Rect box;
                box.x = statsMat.at<int>(i,0);
                box.y = statsMat.at<int>(i,1);
                box.width = statsMat.at<int>(i,2);
                box.height = statsMat.at<int>(i,3);
                //x=174,y=35为手工标记出水口
                double distance = sqrt(pow((box.x - 180),2)+pow((box.y - 35),2));
                //当欧几里得距离小于10时确定为水流的连通域
                if(distance < 10){
                    box.x = 180;
                    box.y = 35;
                    rectangle(src,box,CV_RGB(0,255,0),1,8,0);
                    double k = box.height / pow(box.width,2);
                    // cout << "height:" <<box.height << endl;
                    // cout << "width:" << box.width << endl;
                    // cout << "k:" << k << endl;
                    // cout << "---------------------"<< endl;
                    for(int x=0;x<frame_width-box.x;x++){
                        int y = box.y + k*pow(x,2);
                        // cout << "Y:" << y << endl;
                        if(y > frame_height){
                            break;
                        }
                        else{
                            circle(src,Point(box.x + x,y),1,Scalar(255, 0, 0),1);
                        }   
                    }
                }
            }
            //---------------------水流识别---------------------

            
            imshow("src",src);
            // imshow("redMask",mask);
            // imshow("test",subMat_Fire);
            waitKey(40);
            imwrite("./pic1/"+to_string(count)+".jpg",src);
        }
        
        src_gray.copyTo(lastFrame);
        count++;
    }

    return 0;
}
