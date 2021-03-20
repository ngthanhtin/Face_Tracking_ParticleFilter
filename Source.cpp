
#include "particleFilterTracker.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <vector>
#include <string>
using namespace std;
using namespace cv;

vector<Rect> detectFace(Mat frame)
{
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "Error loading face cascade" << endl;
		return vector<Rect>();
	};
	

	std::vector<Rect> faces;
	Mat frame_gray;
	frame_gray = frame.clone();

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	return faces;
}

void readImageSequence(string path)
{
	Rect toTrack;
	// initialize particle filter tracker
	ParticleFilterTracker trackor=ParticleFilterTracker();	
	float maxWeight=0;

	vector<Point> tracepoint;
	VideoCapture sequence(path + "\\img\\%04d.jpg");
	if (!sequence.isOpened())
	{
		cerr << "Failed to open Image Sequence!\n" << endl;
		return;
	}

	Mat image;
	namedWindow("Image | q or esc to quit", CV_WINDOW_NORMAL);

	vector<Rect> faces;
	bool isTrack = false;
	for (;;)
	{
		sequence >> image;
		if (image.empty())
		{
			cout << "End of Sequence" << endl;
			break;
		}
		if(isTrack==false)
		{
			faces = detectFace(image);
			if(faces.size()!=0)
			{
				rectangle(image, faces[0], Scalar(255, 0, 0), 4, 8, 0);
				isTrack = true;
				toTrack = faces[0];
				trackor.Initialize(image,toTrack);
			}
		}
		else
		{
			
			int t=trackor.ColorParticleTracking(image,toTrack, maxWeight);
			cout<<t<<"Max weight:  "<<maxWeight<<endl;

			tracepoint.push_back(Point(toTrack.x + toTrack.height, toTrack.y + toTrack.width / 2));
			rectangle(image,toTrack,Scalar(10,10,200),5);
			if(tracepoint.size()==30)
			{
				tracepoint.erase(tracepoint.begin()+2);
			}
			for (int i = 0; i < tracepoint.size() - 1;i++)
			{
				line(image, tracepoint[i], tracepoint[i+1], Scalar(255,0, 0), 2, CV_AA);
			}

		}
		
		imshow("image | q or esc to quit", image);

		char key = (char)waitKey(500);
		if (key == 'q' || key == 'Q' || key == 27)
			break;
	}
	destroyAllWindows();
}
void cameraTracking()
{
VideoCapture capture(0);
	Mat frame;

	//detect a face 
	vector<Rect> faces;
	while (1)
	{
		capture >> frame;
		faces = detectFace(frame);
		if(faces.size() == 0)
		{
			continue;
		}

		//draw faces on the picture
		rectangle(frame, faces[0], Scalar(255, 0, 0), 4, 8, 0);
		//show
		imshow("detect face",frame);
		
		char c= waitKey(100);
			if(c=='i')
				break;
	}

	destroyAllWindows();

	// tracking a face
	//just choose one face to track
	Rect toTrack = faces[0];

	// initialize particle filter tracker
	ParticleFilterTracker trackor=ParticleFilterTracker();	
	trackor.Initialize(frame,toTrack);
	
	

	//Begin to track
	float maxWeight=0;
	while (1)
	{
		capture>>frame;
		int t=trackor.ColorParticleTracking(frame,toTrack, maxWeight);
		cout<<t<<"Max weight:  "<<maxWeight<<endl;
		
		rectangle(frame,toTrack,Scalar(10,10,200),5);
		imshow("track",frame);
		
		char c = waitKey(100);
		if(c =='q')
			break;
	}


	destroyAllWindows();
}
int main(int argc, char *argv[])
{
	if(argc==3)
	{
		if(strcmp(argv[1], "sequence") == 0)
		{
			string path = argv[2];
			readImageSequence(path);
		}
		else
		{
			cout << "The parameters are not right\n";
		}
	}
	else if (argc == 2)
	{
		cout << argv[0] << " " << argv[1] << endl;
		if(strcmp(argv[1], "camera") == 0)
		{
			cameraTracking();
		}
		else
		{
			cout << "The parameters are not right 2\n";
		}
	}
	else
	{
		cout << "The parameters are not right 3\n";
	}
	

	return 0;
}

