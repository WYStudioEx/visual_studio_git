/*
������������ƶ��������з���Ϊ���ܹ�������ͼ�����ݽ��б�ע��д����Ƶͼ���и����
��Ҫ�����Ƕ�ȡ��Ƶ�ļ����༭�Զ���Ķ���Σ����ö���ζ�ÿһ֡��ͼ������и
����ĳ������Opencv2.4.5
ע��1��Opencv��polygon����ֻ��֧��͹����Σ����ROIΪ������λ��������
�����ˣ����Գ�
����ʱ�䣺2013.8.22
*/

#include <iostream>	// for standard I/O
#include <string>   // for strings

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write

using namespace std;
using namespace cv;


////����Opencv����Ƶ�ļ��Ĳ���ʱ����ñ��־�̬���ļ����ͺͳ�������ģʽ��һ�¡�
////��Debugģʽ����Debug��Ŀ⣬��Releaseģʽ����Release�汾�Ŀ�
////���������Ƶ�ļ���д�������Բ��ң��ļ����Ƿ���ȷ����д�ļ���·���Ƿ���Ȩ�ޣ������Ƿ�������Debug-Debug_lib��Release-Release_libģʽ��
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

////Ϊ����Ϣ��Ӧ����ʹ�÷��㣬ʹ���˴�����ȫ�ֱ���
////�������Щ�������з�װ�������һ��PolygonSegmentation��
////��Ƶ������صı���
VideoCapture	q_vdCptr("E:\\qzc.avi");
unsigned long	q_frmNoOfVideo = 0;
unsigned long	q_crrntFrmIndex = 0;

////��¼��Ƶ�ļ����ܵ�֡��
string			q_strFrmNo;

/////����и��ͼ���·��
const string	q_cnstImgPth2Sv("E:\\HouseIndor\\Video1\\Wall\\");
const string	q_cnstPartName("WallFrame");
const string	q_cnstSuffix(".bmp");

/////���ڼ�¼Shift������Ӧ
bool shift_on = false; 

/////ͼ��ԭʼ���ݣ��л�֡ʱ�����ݸ���
Mat		qtmpImgSrc;

/////�����������ݣ����ϱ����˻��ƵĶ����
Mat		qtmpImgOprt;

/////���ݴ���ʱ��һ����ʱ��������
/////��Ӧ����¼�ʱ��̬ˢ������ƶ����仯�Ķ���Σ�
/////��Ӧ����ƶ��¼�����̬ˢ�������ƶ��Ķ����
Mat		qtmpImgOprtTmp;

/////��Ŷ���ζ����ROI���ֱ���ɫ���������
/////ԭʼͼ�����ͼ��Ĳ�Ϊ�и��õ��Ľ��
Mat		qtmpImg2Cut;

/////�������ͼ��
Mat		qTmpImg2Save;

/////��¼�������Ƿ񱻰���
bool	q_isMouseDown = false;

/////��Ӧ����ƶ��¼���ֱ��ʱ����¼����ƶ��������յ�
Point	qTmpStartPnt;
Point	qTmpEndPnt;

/////���̿��ƣ���¼�Ƿ��ڻ��ƶ���ε�ģʽ
bool	q_isPolygonDrawBegin = false;

/////����Ҽ�˫���¼���Ӧ����¼������Ƿ�ѡ��
bool	qTmpIsPolygonSeleted = false;

/////�����ƶ������ʱ����¼ÿ���ƶ��¼�����ʼ��
Point	qTmpOrgPnt4Move = Point(0, 0);

/////�����ƶ������ʱ����¼ÿ���ƶ���x��y�ľ���
int		qTmpDeltX = 0;
int		qTmpDeltY = 0;

////����ζ������������
std::vector<cv::Point>	q_pnts4Polygon;

////Ҫ�ı����ε�������λ��ʱ����¼��ѡ�еĶ����Index
int						q_pntIndexSlected = -1;


////�ػ�����
////��ѡ�еĵ����ѡ�е������ı߲��ػ�
void redrawPolygon()
{
	/////����εĶ��㲻����3��
	if (q_pnts4Polygon.size() < 3)
	{
		cout << "less than 3 points, no polygon can be drawn!" << endl;
	}

	qtmpImgOprt = qtmpImgSrc.clone();
	qtmpImgOprtTmp = qtmpImgOprt.clone();

	////����Ҽ�˫����ѡ�ж���Σ���ѡ�еĶ����ȫ�����ɺ�ɫ
	if (qTmpIsPolygonSeleted)
	{
		////�������еĶ���
		for (unsigned qIndex = 0; qIndex < q_pnts4Polygon.size(); qIndex++)
		{

			////���ƶ����ϵľ���
			rectangle(qtmpImgOprt,cv::Rect(q_pnts4Polygon[qIndex].x - 10, q_pnts4Polygon[qIndex].y - 10, 20, 20),Scalar(0,0,255), 3, 8, 0);
	
			////����N-1���㵽N�����ֱ��
			if (qIndex >= 1)
			{
					line(qtmpImgOprt, q_pnts4Polygon[qIndex - 1], q_pnts4Polygon[qIndex], Scalar(0,0,255), 3, 8, 0);
			}

			/////���������1��������1����
			if (qIndex == (q_pnts4Polygon.size() - 1))
			{
				line(qtmpImgOprt, q_pnts4Polygon[0], q_pnts4Polygon[qIndex], Scalar(0,0,255), 3, 8, 0);
			}
		}
	}
	else
	{
		////�����������ж��㱻ѡ�У���Ҫ��ԭ����ηֳ�������������
		for (unsigned qIndex = 0; qIndex < q_pnts4Polygon.size(); qIndex++)
		{
			/////���ƶ��㣬���ǲ����Ʊ�ѡ�еĶ���
			if (q_pntIndexSlected != qIndex)
			{
				rectangle(qtmpImgOprt,cv::Rect(q_pnts4Polygon[qIndex].x - 10, q_pnts4Polygon[qIndex].y - 10, 20, 20),Scalar(0,0,255), 3, 8, 0);
			}

			///����N-1���㵽N����ı�
			if (qIndex >= 1)
			{
				/////�����ƺͱ�ѡ�еĶ��������ı�
				if ((qIndex!= q_pntIndexSlected) && ((qIndex - 1) != q_pntIndexSlected))
				{
					line(qtmpImgOprt, q_pnts4Polygon[qIndex - 1], q_pnts4Polygon[qIndex], Scalar(0,255,0), 3, 8, 0);
				}

			}

			////���������е�1��������1���㣬ͬ�����Ǹö����Ƿ�ѡ��
			if (qIndex == (q_pnts4Polygon.size() - 1) && (q_pntIndexSlected != 0) && (q_pntIndexSlected != qIndex))
			{
				line(qtmpImgOprt, q_pnts4Polygon[0], q_pnts4Polygon[qIndex], Scalar(0,255,0), 3, 8, 0);
			}
		}
	}

	/////�ѻ��ƺõĶ���ο�������ʱ������
	qtmpImgOprt.copyTo(qtmpImgOprtTmp);
}

/////Ϊ�����Ч�ʣ��ƶ�����εĵ�����ʱ��
/////ֻ�ػ汻�ƶ��ĵ����õ�������������
/////��ѡ�еĶ���͸ö��������ı�����ʱ�������л���
void drawSelectedPart(int x, int y)
{
	qtmpImgOprt.copyTo(qtmpImgOprtTmp);

	////����ʱ�������л��Ʊ�ѡ�еĵ�
	rectangle(qtmpImgOprtTmp,cv::Rect(x - 10, y - 10, 20, 20),Scalar(0,255,255), 3, 8, 0);

	
	if (q_pntIndexSlected == 0 )
	{
		////���������1���㱻ѡ�У����ӵ�1����������1������
		qTmpStartPnt = q_pnts4Polygon[q_pnts4Polygon.size() -1];
	}
	else
	{
		////�����ѡ�еĲ��ǵ�1�����㣬������N-1���㵽N����
		qTmpStartPnt = q_pnts4Polygon[q_pntIndexSlected -1];
	}			
	line(qtmpImgOprtTmp, qTmpStartPnt, Point(x, y), Scalar(255,255,0), 3, 8, 0);

	if (q_pntIndexSlected == (q_pnts4Polygon.size() -1))
	{
		////�����������1�����㱻ѡ�У����ӵ�1����������һ������
		qTmpStartPnt = q_pnts4Polygon[0];
	}
	else
	{
		////�����ѡ�еĲ������1�����㣬������N���㵽N+1����
		qTmpStartPnt = q_pnts4Polygon[q_pntIndexSlected +1];
	}			
	line(qtmpImgOprtTmp, qTmpStartPnt, Point(x, y), Scalar(255,255,0), 3, 8, 0);
}


////��Ӧ����ƶ�ʱ���ƶ�����ε�λ��
void polygonMove(int deltX, int deltY)
{
	for (unsigned qIndex = 0; qIndex < q_pnts4Polygon.size(); qIndex++)
	{
		q_pnts4Polygon[qIndex].x +=deltX;
		q_pnts4Polygon[qIndex].y +=deltY;
	}
}


////�ж��Ƿ��ж���ζ��㱻ѡ��
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


/////��Ӧ����¼����ƶ���������£����̧���Ҽ�˫��
/////��̬ˢ�µ�ͼ��һ������ʱ������qtmpImgOprtTmp�д��
void onMouse(int event,int x,int y,int,void*)
{
	////������û�а���
	if (!q_isMouseDown)
	{
		//////������ڻ��ƶ����ģʽ����ʵʱ�������һ�����㵽���λ�õ�ֱ��
		if (q_isPolygonDrawBegin && q_pnts4Polygon.size() > 0)
		{
			qtmpImgOprt.copyTo(qtmpImgOprtTmp);
			qTmpStartPnt = (Point)q_pnts4Polygon[q_pnts4Polygon.size() - 1];
			line(qtmpImgOprtTmp, qTmpStartPnt, Point(x, y), Scalar(0,255,0), 3, 8, 0);
		}
	}

	/////����������
	if (q_isMouseDown)
	{
		////����ж��㱻ѡ�У�Ҫ��̬�ػ汻ѡ�еĵ㣬�Լ���õ�������������
		if (-1 !=q_pntIndexSlected)
		{			
			drawSelectedPart(x, y);
		}

		/////����Ҽ�˫����ѡ���˶���Σ����ʱ�ƶ����Ҫ��̬�ƶ���������ε�λ��
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

	////��Ӧ����������
	if(event==CV_EVENT_LBUTTONDOWN)
	{
		////��¼�����״̬
		if (!q_isMouseDown)  q_isMouseDown = true;

		/////������ڻ��ƶ����ģʽ
		if (q_isPolygonDrawBegin)
		{
			////���Ƶ�1������ʱ������Ҫ����
			if (q_pnts4Polygon.size() > 0)
			{
				qTmpStartPnt = (Point)q_pnts4Polygon[q_pnts4Polygon.size() - 1];
				line(qtmpImgOprt, qTmpStartPnt, Point(x, y), Scalar(0,255,0), 3, 8, 0);
			}

			/////���µĶ���ζ���
			rectangle(qtmpImgOprt,cv::Rect(x - 10, y - 10, 20, 20),Scalar(0,0,255), 3, 8, 0);

			////�����¶����λ��
			q_pnts4Polygon.push_back(Point(x, y));

			////���ƵĶ��㲻���ڶ�̬ˢ�µ�ͼ�����������qtmpImgOprt�л���Ȼ�󿽱�����̬������qtmpImgOprtTmp
			qtmpImgOprt.copyTo(qtmpImgOprtTmp);

			/////����¶���������Ϣ
			cout << "Add one apex into the polygon. Total: " << q_pnts4Polygon.size() << " Apexes "  << endl;
		}

		/////���ƶ�����Ѿ����
		if (!q_isPolygonDrawBegin && q_pnts4Polygon.size() > 3)
		{
			////�ж��Ƿ�ѡ����ĳ������
			if (isAnyApexSelected(x, y))
			{
				redrawPolygon();
				drawSelectedPart(x, y);
			}

			////�����ƶ������ʱ����¼��갴�µĳ�ʼλ��
			if (qTmpIsPolygonSeleted)
			{
				qTmpOrgPnt4Move.x = x;
				qTmpOrgPnt4Move.y = y;
			}
		}

	}
	else if(event==CV_EVENT_LBUTTONUP)   /////������̧��
	{
		/////����������״̬
		if (q_isMouseDown)  q_isMouseDown = false;

		/////������̧��ʱ�����ڻ��ƶ���ε�״̬
		if (q_isPolygonDrawBegin)
		{
			////��֤�����������3������
			if (q_pnts4Polygon.size() > 3)
			{
				///������1���������ʼ���غϣ���������ϲ�
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

		//////���̧��ʱ���б�ѡ�еĵ�
		if ((-1 !=q_pntIndexSlected))
		{
			/////ȷ����ѡ�У����ƶ��Ķ����λ��
			q_pnts4Polygon[q_pntIndexSlected] = Point(x, y);
			q_pntIndexSlected = -1;
			redrawPolygon();
		}

		////�����̧��ʱ����������α�ѡ��
		if (qTmpIsPolygonSeleted)
		{
			qTmpIsPolygonSeleted = false;
			redrawPolygon();
		}

	}
	else if (event == CV_EVENT_RBUTTONDBLCLK)   ////˫��ѡ�ж����
	{
		if (!q_isPolygonDrawBegin)
		{
			///�ж�˫���ĵ��Ƿ��ڶ�����ڲ�
			double qTmpIsPntInPolygon = pointPolygonTest(q_pnts4Polygon, Point(x, y), false);
			if (qTmpIsPntInPolygon > 0)
			{
				qTmpIsPolygonSeleted = true;
				redrawPolygon();
			}
		}
	}
}


////������Ӧ����
int onKeyDown()
{
	char key = cvWaitKey(10);  
	switch(key)  
	{  
	case '\t':   /////��Tab������Shift��
		cout << "Shift On!" << endl;
		shift_on = !shift_on;
		break;

	case 'b':  ////��ʼ����һ���µĶ���Σ�����Ѿ����ڶ���Σ���Ҫɾ��ԭ����
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

	case 'e':  /////���ƶ���ν���
		////����ζ��㲻����3��
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

	case 'c':  ////���ݶ�������õ�ROI����ԭʼͼ������и�
		if (!q_isPolygonDrawBegin)
		{
			if (q_pnts4Polygon.size() > 0)
			{
				////�õ������ѡ�е�ͼ��
				qtmpImg2Cut = qtmpImgSrc.clone();
				fillConvexPoly(qtmpImg2Cut, q_pnts4Polygon, Scalar(0,0,0));
				qTmpImg2Save = qtmpImgSrc - qtmpImg2Cut;

				////�õ���ǰ֡������
				std::stringstream  ss;
				ss << q_crrntFrmIndex;
				q_strFrmNo = ss.str();

				////���ݱ궨��Ҫ�󣬰��и��ͼ�������ʵ�λ�ã��ļ���������������ͼ��֡�����
				imwrite(q_cnstImgPth2Sv + q_cnstPartName + q_strFrmNo + q_cnstSuffix, qTmpImg2Save);

				////��ȡ�µ�֡,ˢ����ʾ������ԭ�еĶ����
				q_vdCptr >> qtmpImgSrc;
				qtmpImgSrc.copyTo(qtmpImgOprt);
				cout << q_crrntFrmIndex << " frames of " << q_frmNoOfVideo << " Cut "<< endl;
				q_crrntFrmIndex++; 
				redrawPolygon();
			}
		}
		break;

	case 'n':  ////����һ֡
		/////�������֡ͼ��
		////��ȡ�µ�֡
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
	//////////˵����Ϣ���
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

	//////��Ƶ�ļ���ȡͼ�񣬶�ȡ��һ֡
	q_vdCptr >> qtmpImgSrc;
	q_crrntFrmIndex++;

	////�õ���Ƶ����֡��
	q_frmNoOfVideo = q_vdCptr.get(CV_CAP_PROP_FRAME_COUNT);
	cout << q_crrntFrmIndex << " frames of " << q_frmNoOfVideo << endl;

	qtmpImgOprt = qtmpImgSrc.clone();
	const string qtmpWndName = "QImageShowTest";
	

	/////��������
	 namedWindow(qtmpWndName,CV_WINDOW_NORMAL);//��ʾ��Ƶԭͼ��Ĵ���
	 moveWindow(qtmpWndName, 100, 100);
	 resizeWindow(qtmpWndName, 1000, 1000 * qtmpImgSrc.rows / qtmpImgSrc.cols);

	//////��׽���
	setMouseCallback(qtmpWndName,onMouse,0);

	while(1)
	{
		/////��Ӧ����
		if (onKeyDown())
		{
			break;
		}
		
		//��ʾ��ƵͼƬ������
		if (q_isPolygonDrawBegin)
		{
			////���ƶ����ʱ��ʾ����̬�������qtmpImgOprtTmp����
			imshow(qtmpWndName,qtmpImgOprtTmp);
		}
		else
		{
			/////�ֶ����������ʱ��ʾ
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


