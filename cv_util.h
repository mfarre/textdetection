#ifndef CV_UTIL_H
#define CV_UTIL_H

#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include "pseudocolor.h"

void writePseudoImage(IplImage *img, char* filename)
{
	Pseudocolor psdata;
	//Init psdata
	psdata.p[0]=1.0;
	psdata.p[1]=2.0;
	psdata.p[2]=1.0;
	psdata.t[0]=0.80;
	psdata.t[1]=0.70;
	psdata.t[2]=0.40;

	//Create image result
	IplImage *img_result= cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 3);
	//Generate basic table and create pseudo image

	generate_pseudocolorTable(&psdata);
	pseudocolor(img, img_result, &psdata);

	cvSaveImage(filename,img_result);
	cvReleaseImage(&img_result);
}	


#endif
