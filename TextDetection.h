/*
*Copyright 2011 Miquel Angel Farre Guiu
*
*This file is part of my personal textDetector implementation based in the Microsoft Research Publication 
*"Detecting Text in Natural Scenes with Stroke Width Transform".
*
*textDetector is free software: you can redistribute it and/or modify it under the terms of the GNU General
*Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option)
*any later *version.
*
*textDetector is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied 
*warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
*You should have received a copy of the GNU General Public License along with Foobar. If not, see http://www.gnu.org/licenses/.
*
*To compile, run: gcc `pkg-config --cflags --libs opencv` -o textDetector src/TextDetection.cpp src/pseudocolor.c
*/

#ifndef TEXTDETECTION_H
#define TEXTDETECTION_H

#include <opencv/cv.h>

struct Point2d {
    int x;
    int y;
    float SWT;
};

struct Point2dFloat {
    float x;
    float y;
};

struct Ray {
        Point2d p;
        Point2d q;
        std::vector<Point2d> points;
};

struct Point3dFloat {
    float x;
    float y;
    float z;
};


struct Chain {
    int p;
    int q;
    float dist;
    bool merged;
    Point2dFloat direction;
    std::vector<int> components;
};

bool Point2dSort (Point2d const & lhs,
                  Point2d const & rhs);

IplImage * loadByteImage ( const char * name );

std::vector < std::pair < CvPoint, CvPoint > >
textDetection (IplImage * in, bool dark_on_light,
	       IplImage * grayImage, IplImage * edgeImage,
	       IplImage * gradientX, IplImage * gradientY,
	       float chain_strictness_pi,
	       float denom_pi_swt_acceptation_angle, float max_color_dist);

void strokeWidthTransform (IplImage * edgeImage,
                           IplImage * gradientX,
                           IplImage * gradientY,
                           bool dark_on_light,
                           IplImage * SWTImage,
                           std::vector<Ray> & rays,
			   float denom_pi_acceptation_angle);

void SWTMedianFilter (IplImage * SWTImage,
                     std::vector<Ray> & rays);

std::vector< std::vector<Point2d> >
findLegallyConnectedComponents (IplImage * SWTImage,
				IplImage * colorImage,
                                std::vector<Ray> & rays);

void componentStats(IplImage * SWTImage,
                                        std::vector<Point2d> & component,
                                        float & mean, float & variance, float & median,
                                        int & minx, int & miny, int & maxx, int & maxy);

void filterComponents(IplImage * SWTImage,
                      std::vector<std::vector<Point2d> > & components,
                      std::vector<std::vector<Point2d> > & validComponents,
                      std::vector<Point2dFloat> & compCenters,
                      std::vector<float> & compMedians,
                      std::vector<Point2d> & compDimensions,
                      std::vector<std::pair<Point2d,Point2d> > & compBB );

std::vector<Chain> makeChains( IplImage * colorImage,
                 std::vector<std::vector<Point2d> > & components,
                 std::vector<Point2dFloat> & compCenters,
                 std::vector<float> & compMedians,
                 std::vector<Point2d> & compDimensions,
                 std::vector<std::pair<Point2d,Point2d> > & compBB,
                 float strictness_pi, float max_color_dist);


#endif // TEXTDETECTION_H

