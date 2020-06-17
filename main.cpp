#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

//This function is not original
void polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	int N = key_point.size();

	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++){
		for (int j = 0; j < n + 1; j++){
			for (int k = 0; k < N; k++){
				X.at<double>(i, j) = X.at<double>(i, j) + std::pow(key_point[k].x, i + j);
			}
		}
	}
 
	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++){
		for (int k = 0; k < N; k++){
			Y.at<double>(i, 0) = Y.at<double>(i, 0) + std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}
 
	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
}

int main(){
    cout << "Hello,World!" << endl;

    VideoCapture cap("./src.mp4");

    Mat src,src_gray,src_hsv;
    Mat bgMat,subMat,subMat_b,subMat_Fire,subMat_b_Fire;
    Mat mask,mask1,mask2;
    Mat labelsMat,labelsMat2;//标签号
    Mat statsMat,statsMat2;//状态矩阵
    Mat centroidsMat,centroidsMat2;//连通域中心
    Mat lastFrame;
    Mat kernel = getStructuringElement(0,Size(3,3));
    Mat kernel1 = getStructuringElement(0,Size(7,7));
    Mat A;

    vector<Point> points;

    int count = 0;
    int CNT = 0;
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
            src.copyTo(lastFrame);
        }
        else{
            absdiff(src_gray,bgMat,subMat);//对视频首帧视频差分
            absdiff(src,lastFrame,subMat_Fire);//对视频前帧视频差分
            cvtColor(subMat_Fire,subMat_Fire,COLOR_BGR2GRAY);

            //对差分后的图像二值化处理
            threshold(subMat,subMat_b,50,255,CV_THRESH_BINARY);
            threshold(subMat_Fire,subMat_b_Fire,50,255,CV_THRESH_BINARY);

            //二值图膨胀运算
            morphologyEx(subMat_b,subMat_b,1,kernel);
            morphologyEx(subMat_b_Fire,subMat_b_Fire,1,kernel);

            Mat src_sub_Fire;
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

                    CNT++;
                    if(CNT<160){
                        points.push_back(cv::Point(box.width+180,box.height+35));
                    }
                    
                    polynomial_curve_fit(points, 2, A);
                    vector<Point> points_fitted;
                    for (int x = 180; x < 500; x++)
                    {
                        double y = A.at<double>(0, 0) + A.at<double>(1, 0)*x + A.at<double>(2, 0)*pow(x, 2);
                        points_fitted.push_back(cv::Point(x, y));
                    }
                    polylines(src, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);

                }
            }
            //---------------------水流识别---------------------

            
            imshow("src",src);
            waitKey(40);
        }
        
        count++;
    }

    return 0;
}
