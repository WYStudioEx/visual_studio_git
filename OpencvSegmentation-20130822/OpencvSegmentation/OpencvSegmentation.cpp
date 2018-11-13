/*
这个程序是在移动机器人研发中为了能够对样本图像数据进行标注而写的视频图像切割程序，
主要功能是读取视频文件，编辑自定义的多边形，利用多边形对每一帧的图像进行切割。
最初的程序采用Opencv2.4.5
注意1：Opencv的polygon函数只能支持凸多边形，如果ROI为凹多边形会出现问题
开发人：邱自成
开发时间：2013.8.22
*/

#include <iostream>	// for standard I/O
#include <string>   // for strings

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write

using namespace std;
using namespace cv;


////在用Opencv做视频文件的操作时，最好保持静态库文件类型和程序编译的模式相一致。
////在Debug模式下用Debug版的库，在Release模式下用Release版本的库
////如果出现视频文件读写出错，可以查找：文件名是否正确，读写文件的路径是否有权限，程序是否运行在Debug-Debug_lib或Release-Release_lib模式下
#ifdef _DEBUG
#pragma comment(lib, "opencv_calib3d245d.lib")
#pragma comment(lib, "opencv_contrib245d.lib")
#pragma comment(lib, "opencv_core245d.lib")
#pragma comment(lib, "opencv_features2d245d.lib")
#pragma comment(lib, "opencv_flann245d.lib")
#pragma comment(lib, "opencv_gpu245d.lib")
///#pragma comment(lib, "opencv_haartraining_engined.lib")
#pragma comment(lib, "opencv_highgui245d.lib")
#pragma comment(lib, "opencv_imgproc245d.lib")
#pragma comment(lib, "opencv_legacy245d.lib")
#pragma comment(lib, "opencv_ml245d.lib")
#pragma comment(lib, "opencv_nonfree245d.lib")
#pragma comment(lib, "opencv_objdetect245d.lib")
#pragma comment(lib, "opencv_photo245d.lib")
#pragma comment(lib, "opencv_stitching245d.lib")
#pragma comment(lib, "opencv_superres245d.lib")
#pragma comment(lib, "opencv_ts245d.lib")
#pragma comment(lib, "opencv_video245d.lib")
#pragma comment(lib, "opencv_videostab245d.lib")
#else
#pragma comment(lib, "opencv_calib3d245.lib")
#pragma comment(lib, "opencv_contrib245.lib")
#pragma comment(lib, "opencv_core245.lib")
#pragma comment(lib, "opencv_features2d245.lib")
#pragma comment(lib, "opencv_flann245.lib")
#pragma comment(lib, "opencv_gpu245.lib")
///#pragma comment(lib, "opencv_haartraining_engined.lib")
#pragma comment(lib, "opencv_highgui245.lib")
#pragma comment(lib, "opencv_imgproc245.lib")
#pragma comment(lib, "opencv_legacy245.lib")
#pragma comment(lib, "opencv_ml245.lib")
#pragma comment(lib, "opencv_nonfree245.lib")
#pragma comment(lib, "opencv_objdetect245.lib")
#pragma comment(lib, "opencv_photo245.lib")
#pragma comment(lib, "opencv_stitching245.lib")
#pragma comment(lib, "opencv_superres245.lib")
#pragma comment(lib, "opencv_ts245.lib")
#pragma comment(lib, "opencv_video245.lib")
#pragma comment(lib, "opencv_videostab245.lib")
#endif


#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

////为了消息响应函数使用方便，使用了大量的全局变量
////建议对这些变量进行封装，如设计一个PolygonSegmentation类
////视频操作相关的变量
VideoCapture	q_vdCptr("E:\\qzc.avi");
unsigned long	q_frmNoOfVideo = 0;
unsigned long	q_crrntFrmIndex = 0;

////记录视频文件中总的帧数
string			q_strFrmNo;

/////存放切割后图像的路径
const string	q_cnstImgPth2Sv("E:\\HouseIndor\\Video1\\Wall\\");
const string	q_cnstPartName("WallFrame");
const string	q_cnstSuffix(".bmp");

/////用于记录Shift按键响应
bool shift_on = false; 

/////图像原始数据，切换帧时，数据更新
Mat		qtmpImgSrc;

/////被处理后的数据，其上保留了绘制的多边形
Mat		qtmpImgOprt;

/////数据处理时的一个临时缓冲区，
/////响应鼠标事件时动态刷新鼠标移动而变化的多边形，
/////响应鼠标移动事件，动态刷新整个移动的多边形
Mat		qtmpImgOprtTmp;

/////存放多边形定义的ROI部分被白色填充后的数据
/////原始图像与该图像的差为切割后得到的结果
Mat		qtmpImg2Cut;

/////被保存的图像
Mat		qTmpImg2Save;

/////记录鼠标左键是否被按下
bool	q_isMouseDown = false;

/////响应鼠标移动事件画直线时，记录鼠标移动的起点和终点
Point	qTmpStartPnt;
Point	qTmpEndPnt;

/////键盘控制，记录是否处于绘制多边形的模式
bool	q_isPolygonDrawBegin = false;

/////鼠标右键双击事件响应，记录多边形是否被选中
bool	qTmpIsPolygonSeleted = false;

/////整体移动多边形时，记录每次移动事件的起始点
Point	qTmpOrgPnt4Move = Point(0, 0);

/////整体移动多边形时，记录每次移动的x和y的距离
int		qTmpDeltX = 0;
int		qTmpDeltY = 0;

////多边形顶点坐标的数组
std::vector<cv::Point>	q_pnts4Polygon;

////要改变多边形单个顶点位置时，记录被选中的顶点的Index
int						q_pntIndexSlected = -1;


////重绘多边形
////被选中的点和与选中点相连的边不重绘
void redrawPolygon()
{
	/////多边形的顶点不少于3个
	if (q_pnts4Polygon.size() < 3)
	{
		cout << "less than 3 points, no polygon can be drawn!" << endl;
	}

	qtmpImgOprt = qtmpImgSrc.clone();
	qtmpImgOprtTmp = qtmpImgOprt.clone();

	////鼠标右键双击会选中多边形，被选中的多边形全部画成红色
	if (qTmpIsPolygonSeleted)
	{
		////遍历所有的顶点
		for (unsigned qIndex = 0; qIndex < q_pnts4Polygon.size(); qIndex++)
		{

			////绘制顶点上的矩形
			rectangle(qtmpImgOprt,cv::Rect(q_pnts4Polygon[qIndex].x - 10, q_pnts4Polygon[qIndex].y - 10, 20, 20),Scalar(0,0,255), 3, 8, 0);
	
			////绘制N-1顶点到N顶点的直线
			if (qIndex >= 1)
			{
					line(qtmpImgOprt, q_pnts4Polygon[qIndex - 1], q_pnts4Polygon[qIndex], Scalar(0,0,255), 3, 8, 0);
			}

			/////处理数组第1个点和最后1个点
			if (qIndex == (q_pnts4Polygon.size() - 1))
			{
				line(qtmpImgOprt, q_pnts4Polygon[0], q_pnts4Polygon[qIndex], Scalar(0,0,255), 3, 8, 0);
			}
		}
	}
	else
	{
		////如果多边形中有顶点被选中，需要把原多边形分成两部分来绘制
		for (unsigned qIndex = 0; qIndex < q_pnts4Polygon.size(); qIndex++)
		{
			/////绘制顶点，但是不绘制被选中的顶点
			if (q_pntIndexSlected != qIndex)
			{
				rectangle(qtmpImgOprt,cv::Rect(q_pnts4Polygon[qIndex].x - 10, q_pnts4Polygon[qIndex].y - 10, 20, 20),Scalar(0,0,255), 3, 8, 0);
			}

			///绘制N-1顶点到N顶点的边
			if (qIndex >= 1)
			{
				/////不绘制和被选中的顶点相连的边
				if ((qIndex!= q_pntIndexSlected) && ((qIndex - 1) != q_pntIndexSlected))
				{
					line(qtmpImgOprt, q_pnts4Polygon[qIndex - 1], q_pnts4Polygon[qIndex], Scalar(0,255,0), 3, 8, 0);
				}

			}

			////处理数组中第1个点和最后1个点，同样考虑该顶点是否被选中
			if (qIndex == (q_pnts4Polygon.size() - 1) && (q_pntIndexSlected != 0) && (q_pntIndexSlected != qIndex))
			{
				line(qtmpImgOprt, q_pnts4Polygon[0], q_pnts4Polygon[qIndex], Scalar(0,255,0), 3, 8, 0);
			}
		}
	}

	/////把绘制好的多边形拷贝至临时缓冲区
	qtmpImgOprt.copyTo(qtmpImgOprtTmp);
}

/////为了提高效率，移动多边形的单个点时，
/////只重绘被移动的点和与该点相连的两条边
/////被选中的顶点和该顶点相连的边在临时缓冲区中绘制
void drawSelectedPart(int x, int y)
{
	qtmpImgOprt.copyTo(qtmpImgOprtTmp);

	////在临时缓冲区中绘制被选中的点
	rectangle(qtmpImgOprtTmp,cv::Rect(x - 10, y - 10, 20, 20),Scalar(0,255,255), 3, 8, 0);

	
	if (q_pntIndexSlected == 0 )
	{
		////考虑数组第1个点被选中，连接第1个顶点和最后1个顶点
		qTmpStartPnt = q_pnts4Polygon[q_pnts4Polygon.size() -1];
	}
	else
	{
		////如果被选中的不是第1个顶点，则连接N-1顶点到N顶点
		qTmpStartPnt = q_pnts4Polygon[q_pntIndexSlected -1];
	}			
	line(qtmpImgOprtTmp, qTmpStartPnt, Point(x, y), Scalar(255,255,0), 3, 8, 0);

	if (q_pntIndexSlected == (q_pnts4Polygon.size() -1))
	{
		////考虑数组最后1个顶点被选中，连接第1个顶点和最后一个顶点
		qTmpStartPnt = q_pnts4Polygon[0];
	}
	else
	{
		////如果被选中的不是最后1个顶点，则连接N顶点到N+1顶点
		qTmpStartPnt = q_pnts4Polygon[q_pntIndexSlected +1];
	}			
	line(qtmpImgOprtTmp, qTmpStartPnt, Point(x, y), Scalar(255,255,0), 3, 8, 0);
}


////响应鼠标移动时，移动多边形的位置
void polygonMove(int deltX, int deltY)
{
	for (unsigned qIndex = 0; qIndex < q_pnts4Polygon.size(); qIndex++)
	{
		q_pnts4Polygon[qIndex].x +=deltX;
		q_pnts4Polygon[qIndex].y +=deltY;
	}
}


////判断是否有多边形顶点被选中
int isAnyApexSelected(int x, int y)
{
	for (unsigned qIndex = 0; qIndex < q_pnts4Polygon.size(); qIndex++)
	{
		if ((abs(x - q_pnts4Polygon[qIndex].x) < 10) || (abs(y - q_pnts4Polygon[qIndex].y) < 10))
		{
			q_pntIndexSlected = qIndex;
			return 1;
		}
	}
	return 0;
}


/////响应鼠标事件：移动，左键按下，左键抬起，右键双击
/////动态刷新的图像一律在临时缓冲区qtmpImgOprtTmp中存放
void onMouse(int event,int x,int y,int,void*)
{
	////鼠标左键没有按下
	if (!q_isMouseDown)
	{
		//////如果处于绘制多边形模式，则实时绘制最后一个顶点到鼠标位置的直线
		if (q_isPolygonDrawBegin && q_pnts4Polygon.size() > 0)
		{
			qtmpImgOprt.copyTo(qtmpImgOprtTmp);
			qTmpStartPnt = (Point)q_pnts4Polygon[q_pnts4Polygon.size() - 1];
			line(qtmpImgOprtTmp, qTmpStartPnt, Point(x, y), Scalar(0,255,0), 3, 8, 0);
		}
	}

	/////鼠标左键按下
	if (q_isMouseDown)
	{
		////如果有顶点被选中，要动态重绘被选中的点，以及与该点相连的两条边
		if (-1 !=q_pntIndexSlected)
		{			
			drawSelectedPart(x, y);
		}

		/////如果右键双击后选中了多边形，则此时移动鼠标要动态移动整个多边形的位置
		if (qTmpIsPolygonSeleted)
		{
			qTmpDeltX = x - qTmpOrgPnt4Move.x;
			qTmpDeltY = y - qTmpOrgPnt4Move.y;
			polygonMove(qTmpDeltX, qTmpDeltY);
			redrawPolygon();
			qTmpOrgPnt4Move.x = x;
			qTmpOrgPnt4Move.y = y;
		}

	}

	////响应鼠标左键按下
	if(event==CV_EVENT_LBUTTONDOWN)
	{
		////记录左键的状态
		if (!q_isMouseDown)  q_isMouseDown = true;

		/////如果处于绘制多边形模式
		if (q_isPolygonDrawBegin)
		{
			////绘制第1个顶点时，不需要画边
			if (q_pnts4Polygon.size() > 0)
			{
				qTmpStartPnt = (Point)q_pnts4Polygon[q_pnts4Polygon.size() - 1];
				line(qtmpImgOprt, qTmpStartPnt, Point(x, y), Scalar(0,255,0), 3, 8, 0);
			}

			/////画新的多边形顶点
			rectangle(qtmpImgOprt,cv::Rect(x - 10, y - 10, 20, 20),Scalar(0,0,255), 3, 8, 0);

			////保存新顶点的位置
			q_pnts4Polygon.push_back(Point(x, y));

			////绘制的顶点不属于动态刷新的图像，因此首先在qtmpImgOprt中画，然后拷贝至动态缓冲区qtmpImgOprtTmp
			qtmpImgOprt.copyTo(qtmpImgOprtTmp);

			/////输出新顶点的相关信息
			cout << "Add one apex into the polygon. Total: " << q_pnts4Polygon.size() << " Apexes "  << endl;
		}

		/////绘制多边形已经完成
		if (!q_isPolygonDrawBegin && q_pnts4Polygon.size() > 3)
		{
			////判断是否选中了某个顶点
			if (isAnyApexSelected(x, y))
			{
				redrawPolygon();
				drawSelectedPart(x, y);
			}

			////整体移动多边形时，记录鼠标按下的初始位置
			if (qTmpIsPolygonSeleted)
			{
				qTmpOrgPnt4Move.x = x;
				qTmpOrgPnt4Move.y = y;
			}
		}

	}
	else if(event==CV_EVENT_LBUTTONUP)   /////鼠标左键抬起
	{
		/////更新鼠标左键状态
		if (q_isMouseDown)  q_isMouseDown = false;

		/////如果左键抬起时，处于绘制多边形的状态
		if (q_isPolygonDrawBegin)
		{
			////保证多边形至少有3个顶点
			if (q_pnts4Polygon.size() > 3)
			{
				///如果最后1个顶点和起始点重合，则将两个点合并
				qTmpStartPnt = q_pnts4Polygon[0];
				qTmpEndPnt = q_pnts4Polygon[q_pnts4Polygon.size() - 1];
				if (abs(qTmpStartPnt.x - qTmpEndPnt.x) < 20 && abs(qTmpStartPnt.y - qTmpEndPnt.y) < 20 )
				{
					q_pnts4Polygon.pop_back();
					q_isPolygonDrawBegin = false;
					redrawPolygon();
					cout << "Draw a new polygon End! Total: " << q_pnts4Polygon.size() << " Apexes " << endl;
				}
			}
		}

		//////左键抬起时，有被选中的点
		if ((-1 !=q_pntIndexSlected))
		{
			/////确定被选中，被移动的顶点的位置
			q_pnts4Polygon[q_pntIndexSlected] = Point(x, y);
			q_pntIndexSlected = -1;
			redrawPolygon();
		}

		////在左键抬起时，整个多边形被选中
		if (qTmpIsPolygonSeleted)
		{
			qTmpIsPolygonSeleted = false;
			redrawPolygon();
		}

	}
	else if (event == CV_EVENT_RBUTTONDBLCLK)   ////双击选中多边形
	{
		if (!q_isPolygonDrawBegin)
		{
			///判断双击的点是否在多边形内部
			double qTmpIsPntInPolygon = pointPolygonTest(q_pnts4Polygon, Point(x, y), false);
			if (qTmpIsPntInPolygon > 0)
			{
				qTmpIsPolygonSeleted = true;
				redrawPolygon();
			}
		}
	}
}


////按键响应函数
int onKeyDown()
{
	char key = cvWaitKey(10);  
	switch(key)  
	{  
	case '\t':   /////用Tab键代替Shift键
		cout << "Shift On!" << endl;
		shift_on = !shift_on;
		break;

	case 'b':  ////开始绘制一个新的多边形，如果已经存在多边形，则要删除原来的
		cout << "Draw a new polygon Begin!" << endl;
		if (q_pnts4Polygon.size() > 0)
		{
			q_pnts4Polygon.clear();
		}

		q_pntIndexSlected = -1;
		qtmpImgOprt = qtmpImgSrc.clone();
		qtmpImgOprtTmp = qtmpImgOprt.clone();		
		q_isPolygonDrawBegin = true; 
		break;

	case 'e':  /////绘制多边形结束
		////多边形顶点不少于3个
		if (q_pnts4Polygon.size() < 3)
		{
			cout << "A polygon's apexes no less than 3! Please add more points!" << endl;
			break;
		}
		cout << "Draw a new polygon End! Total: " << q_pnts4Polygon.size() << " Apexes " << endl;
		qTmpStartPnt = q_pnts4Polygon[0];
		qTmpEndPnt = q_pnts4Polygon[q_pnts4Polygon.size() - 1];
		line(qtmpImgOprt, qTmpStartPnt, qTmpEndPnt, Scalar(0,255,0), 3, 8, 0);
		qtmpImgOprt.copyTo(qtmpImgOprtTmp);
		q_isPolygonDrawBegin = false; 
		break;

	case 'c':  ////根据多边形设置的ROI，对原始图像进行切割
		if (!q_isPolygonDrawBegin)
		{
			if (q_pnts4Polygon.size() > 0)
			{
				////得到多边形选中的图像
				qtmpImg2Cut = qtmpImgSrc.clone();
				fillConvexPoly(qtmpImg2Cut, q_pnts4Polygon, Scalar(0,0,0));
				qTmpImg2Save = qtmpImgSrc - qtmpImg2Cut;

				////得到当前帧的索引
				std::stringstream  ss;
				ss << q_crrntFrmIndex;
				q_strFrmNo = ss.str();

				////根据标定的要求，把切割的图像存入合适的位置，文件名包括对象名称图像帧的序号
				imwrite(q_cnstImgPth2Sv + q_cnstPartName + q_strFrmNo + q_cnstSuffix, qTmpImg2Save);

				////读取新的帧,刷新显示，保留原有的多边形
				q_vdCptr >> qtmpImgSrc;
				qtmpImgSrc.copyTo(qtmpImgOprt);
				cout << q_crrntFrmIndex << " frames of " << q_frmNoOfVideo << " Cut "<< endl;
				q_crrntFrmIndex++; 
				redrawPolygon();
			}
		}
		break;

	case 'n':  ////到下一帧
		/////不处理该帧图像
		////读取新的帧
		q_vdCptr >> qtmpImgSrc;
		qtmpImgSrc.copyTo(qtmpImgOprt);
		cout << q_crrntFrmIndex << " frames of " << q_frmNoOfVideo << " Pass without action " << endl;
		q_crrntFrmIndex++;
		redrawPolygon();
		break;

	case 27:
		return -1;
	};  
	return 0;
}

int main(int argc, unsigned char* argv[])
{
	//////////说明信息输出
	cout << "Implement image segmentation!" << endl;
	cout << "Press ESC to exit" << endl;

	cout << "Press b to begin image cut" << endl;
	cout << "Press e to end image cut" << endl;
	cout << "Segmentation end when tail point meets with start point" << endl;

	cout << "Reading video file..." << endl;
	if (!q_vdCptr.isOpened())
	{
		cout << "video file reading failed!" << endl;
		return 1;
	}

	//////视频文件抽取图像，读取第一帧
	q_vdCptr >> qtmpImgSrc;
	q_crrntFrmIndex++;

	////得到视频的总帧数
	q_frmNoOfVideo = q_vdCptr.get(CV_CAP_PROP_FRAME_COUNT);
	cout << q_crrntFrmIndex << " frames of " << q_frmNoOfVideo << endl;

	qtmpImgOprt = qtmpImgSrc.clone();
	const string qtmpWndName = "QImageShowTest";
	

	/////建立窗口
	 namedWindow(qtmpWndName,CV_WINDOW_NORMAL);//显示视频原图像的窗口
	 moveWindow(qtmpWndName, 100, 100);
	 resizeWindow(qtmpWndName, 1000, 1000 * qtmpImgSrc.rows / qtmpImgSrc.cols);

	//////捕捉鼠标
	setMouseCallback(qtmpWndName,onMouse,0);

	while(1)
	{
		/////响应按键
		if (onKeyDown())
		{
			break;
		}
		
		//显示视频图片到窗口
		if (q_isPolygonDrawBegin)
		{
			////绘制多边形时显示，动态情况下用qtmpImgOprtTmp数据
			imshow(qtmpWndName,qtmpImgOprtTmp);
		}
		else
		{
			/////手动调整多边形时显示
			if (q_isMouseDown && (-1 !=q_pntIndexSlected))
			{
				imshow(qtmpWndName,qtmpImgOprtTmp);
			}
			else
			{
				imshow(qtmpWndName,qtmpImgOprt);
			}			
		}
	}
	return 0;
}


