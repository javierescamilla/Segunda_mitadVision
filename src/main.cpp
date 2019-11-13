#include <chrono>
#include <ctime>
#include <thread>
#include "BebopController/BebopDrone.hpp"

#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <queue>
#include <string>
#include <stdlib.h>

#define FILTER_SAMPLE 15
#define TOLERANCE 45

#define MINIMUM_AREA 30000

#define BLACK cv::Vec3b(0, 0, 0)
#define BLUE cv::Vec3b(0, 255, 0)
#define GREEN cv::Vec3b(255, 0, 0)
#define MAX_VALUE 255
#define RANDOM_COLOR_MAX 200
#define RANDOM_COLOR_MIN 50
#define RED cv::Vec3b(0, 0, 255)
#define THRESHOLD 127
#define WHITE cv::Vec3b(255, 255, 255)

#define RING_PH1_AVG 0.00107054
#define RING_PH2_AVG 2.11875E-08
#define RING_PH1_VAR 5.81799999999999E-05 *2
#define RING_PH2_VAR 6.5761E-09 *2
#define TIE_PH1_AVG 0.00155887
#define TIE_PH2_AVG 1.780055E-06
#define TIE_PH1_VAR 8.27999999999998E-05 *2
#define TIE_PH2_VAR 2.40715E-07 *2
#define PANTS_PH1_AVG 0.001297275
#define PANTS_PH2_AVG 4.32789E-07
#define PANTS_PH1_VAR 0.000131785 *2
#define PANTS_PH2_VAR 9.62E-08 *2
#define SHIRT_PH1_AVG 0.0007930105
#define SHIRT_PH2_AVG 6.54703E-08
#define SHIRT_PH1_VAR 1.04975E-05 *2 
#define SHIRT_PH2_VAR 1.10927E-08 *2

#define DRONE_SPEED 50
#define TIME_MOVE 2000000

using namespace std;
using namespace cv;
using namespace std::chrono;

const int KEY_DELAY_MS = 50;
const int DRONE_OUTPUT = 70;

const String WINDOW_ORIGINAL_NAME = "Original";
const String WINDOW_FLIPPED_NAME = "Flipped";

void flipImageBasic(const Mat &sourceImage, Mat &destinationImage);

std::vector<cv::Vec3b> pixels;

void mouseEvent(int event, int x, int y, int flags, void* param) {
  cv::Mat* image = (cv::Mat*)param;

  switch (event)
  {
      case CV_EVENT_LBUTTONDOWN:
          pixels.push_back(image->at<cv::Vec3b>(y,x));
          std::cout << x << "\t" << y << std::endl;
          break;
      case CV_EVENT_MOUSEMOVE:
          break;
      case CV_EVENT_LBUTTONUP:
          break;
  }
}


std::vector<int> filterHSV(const cv::Mat& hsvImage) {
  std::vector<int> limits {0, 0, 0, 0, 0, 0};
  int data[3] = {0, 0, 0};
  cv::Mat filteredImage;
  
  while (cv::waitKey(1) != 'x') {  
    if (pixels.size() >= FILTER_SAMPLE) {
      for (int i = 1; i <= FILTER_SAMPLE; i++) {
        data[0] += pixels[pixels.size() - i][0];
        data[1] += pixels[pixels.size() - i][1];
        data[2] += pixels[pixels.size() - i][2];
      }
      data[0] /= FILTER_SAMPLE;
      data[1] /= FILTER_SAMPLE;
      data[2] /= FILTER_SAMPLE;
      int limit = 0;
      for (int i = 0; i < 3; i++) {
        limits[limit] = data[i] - TOLERANCE;
        limits[limit + 1] = data[i] + TOLERANCE;
        limit += 2;
      }
      cv::inRange(hsvImage, cv::Scalar(limits[0], limits[2], limits[4]),
      cv::Scalar(limits[1], limits[3], limits[5]), filteredImage);
      cv::imshow("Image", filteredImage);

    }

  }
  cv::destroyWindow("Image");
  return limits;
}


cv::Vec3b getRandomColor(int minValue, int maxValue) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type>  randomNum(minValue, maxValue);
  return cv::Vec3b(randomNum(rng), randomNum(rng), randomNum(rng));
}

cv::Moments expandColor(const int& row, const int& col, cv::Mat& image,
 const cv::Vec3b& rightColor, const cv::Vec3b& wrongColor) {
  cv::Mat auxImage(image.rows, image.cols, CV_8UC3, cv::Scalar(0,0,0));
  std::queue<int> queue;
  queue.push(row);
  queue.push(col);

  while(!queue.empty()) {
    int row = queue.front();
    queue.pop();
    int col = queue.front();
    queue.pop();

    auxImage.at<cv::Vec3b>(row, col) = WHITE;
    image.at<cv::Vec3b>(row, col) = rightColor;

    if (row - 1 >= 0) {
      if (image.at<cv::Vec3b>(row - 1, col) == wrongColor) {
        image.at<cv::Vec3b>(row - 1, col) = rightColor;
        queue.push(row - 1);
        queue.push(col);
      }
    }
    if (col + 1 < image.cols) {
      if (image.at<cv::Vec3b>(row, col + 1) == wrongColor) {
        image.at<cv::Vec3b>(row, col + 1) = rightColor;
        queue.push(row);
        queue.push(col + 1);
      }
    }
    if (col - 1 >= 0) {
      if (image.at<cv::Vec3b>(row, col - 1) == wrongColor) {
        image.at<cv::Vec3b>(row, col - 1) = rightColor;
        queue.push(row);
        queue.push(col - 1);
      }
    }
    if (row + 1 < image.rows) {
      if (image.at<cv::Vec3b>(row + 1, col) == wrongColor) {
        image.at<cv::Vec3b>(row + 1, col) = rightColor;
        queue.push(row + 1);
        queue.push(col);
      }
    }
  }
  cv::cvtColor(auxImage, auxImage, cv::COLOR_RGB2GRAY);
  return cv::moments(auxImage, false);
}

std::vector<cv::Moments> selectiveSegmentation(cv::Mat& image, const cv::Vec3b& objectColor) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> randomCol(0, image.cols - 1);
  std::uniform_int_distribution<std::mt19937::result_type> randomRow(0, image.rows - 1);

  int row = randomRow(rng);
  int col = randomCol(rng);

  std::vector<cv::Moments> moments;

  int numTriesToDetectObject = 0;
  while(numTriesToDetectObject < image.cols * image.rows) {
    row = randomRow(rng);
    col = randomCol(rng);
    numTriesToDetectObject++;
    if (image.at<cv::Vec3b>(row, col) == objectColor) {
      cv::Vec3b color = getRandomColor(RANDOM_COLOR_MIN, RANDOM_COLOR_MAX);
      moments.push_back(expandColor(row, col, image, color, objectColor));
      numTriesToDetectObject = 0;
    }
  }

  return moments;
}

std::vector<std::vector<double>> draw(const std::vector<cv::Moments>& moments, cv::Mat& image) {
  std::vector<std::vector<double>> huMoments;
  for (auto moment : moments) {
    huMoments.push_back(std::vector<double>());
    cv::HuMoments(moment, huMoments.back());
    double angle = 0.5*atan2(2*moment.mu11, moment.mu20-moment.mu02);
    (huMoments.back()).push_back(angle);
    double xCenter = moment.m10/moment.m00;
    (huMoments.back()).push_back(xCenter);
    double yCenter = moment.m01/moment.m00;
    (huMoments.back()).push_back(yCenter);

    cv::circle(image, cv::Point(xCenter,yCenter), 3, (0,255,0), -1);

    int length = 100;
    
    int x =  xCenter + length * cos(angle);
    int y =  yCenter + length * sin(angle);
    int x2 =  xCenter + length * cos(angle +  M_PI*0.5);
    int y2 =  yCenter + length * sin(angle +  M_PI*0.5);

    cv::line(image, cv::Point(xCenter, yCenter), cv::Point(x, y), cv::Scalar(0,0,255));
    cv::line(image, cv::Point(xCenter, yCenter), cv::Point(x2, y2), cv::Scalar(255,0,0));
    
  }
  return huMoments;
}

std::vector<std::vector<double>> filterAndSegment(cv::Mat &image){
  cv::setMouseCallback("Image", mouseEvent, &image);
  pixels.clear();
  cv::Mat filteredImage;
  vector<int> limits {0, 0, 0, 0, 0, 0};

  GaussianBlur(image, image, Size(5, 5), 0);

  limits = filterHSV(image);

  cv::inRange(image, cv::Scalar(limits[0], limits[2], limits[4]),
  cv::Scalar(limits[1], limits[3], limits[5]), filteredImage);

  cv::erode(filteredImage, filteredImage, cv::Mat());
  cv::erode(filteredImage, filteredImage, cv::Mat());
  cv::dilate(filteredImage, filteredImage, cv::Mat());
  //cv::erode(filteredImage, filteredImage, cv::Mat());

  cv::cvtColor(filteredImage, filteredImage, cv::COLOR_GRAY2BGR);
  
  cv::imshow("Filtered Image", filteredImage);

  auto moments = selectiveSegmentation(filteredImage, WHITE);
  auto huMoments = draw(moments, filteredImage);
  imshow("Selective segmentation", filteredImage);
  waitKey(0);

  return huMoments;
}


vector<bool> caracterize(std::vector<std::vector<double>> huMoments){
  float phi1,phi2,angle,x,y;
  vector<bool> steps (3);
  for(auto hu : huMoments){
    phi1 = hu[0];
    phi2 = hu[1];
    angle = hu[7];
    x = hu[8];
    y = hu[9];
    cout << phi1 << " " << phi2 << " " << angle << " (" << (int)x << "," << (int)y << ")" << endl;
    if(phi1 < RING_PH1_AVG + RING_PH1_VAR && phi1 > RING_PH1_AVG - RING_PH1_VAR){
      if(phi2 < RING_PH2_AVG + RING_PH2_VAR && phi2 > RING_PH2_AVG - RING_PH2_VAR){
        cout << "ANILLO" << endl;
        steps[2] = false;//Atras
      } 
    }
    if(phi1 < TIE_PH1_AVG + TIE_PH1_VAR && phi1 > TIE_PH1_AVG - TIE_PH1_VAR){
      if(phi2 < TIE_PH2_AVG + TIE_PH2_VAR && phi2 > TIE_PH2_AVG - TIE_PH2_VAR){
        cout << "CORBATA" << endl;
        steps[0] = false;//Izquierda
        steps[1] = angle <= 0 && angle >= -1.5707963267949;
      } 
    }
    if(phi1 < PANTS_PH1_AVG + PANTS_PH1_VAR && phi1 > PANTS_PH1_AVG - PANTS_PH1_VAR){
      if(phi2 < PANTS_PH2_AVG + PANTS_PH2_VAR && phi2 > PANTS_PH2_AVG - PANTS_PH2_VAR){
        cout << "PANTALON" << endl;
        steps[0] = true;//Derecha
        if(angle >= -1.5707963267949 && angle <=0){
          steps[1] = true;
        }
        else if(angle <= 1.5707963267949 && angle >0){
          steps[1] = false;
        }
      } 
    }
    if(phi1 < SHIRT_PH1_AVG + SHIRT_PH1_VAR && phi1 > SHIRT_PH1_AVG - SHIRT_PH1_VAR){
      if(phi2 < SHIRT_PH2_AVG + SHIRT_PH2_VAR && phi2 > SHIRT_PH2_AVG - SHIRT_PH2_VAR){
        cout << "PLAYERA" << endl;
        steps[2] = true;//Adelante
      } 
    }
  }
  return steps;
}

void doRoutine(vector<bool> steps, BebopDrone &drone){
  if(steps[0]){
    cout << "right\n";
    drone.setRoll(DRONE_SPEED);
    usleep(TIME_MOVE);
    drone.hover();
    usleep(2000000);
  }
  else{
    cout << "left\n";
    drone.setRoll(-DRONE_SPEED);
    usleep(TIME_MOVE);
    drone.hover();
    usleep(2000000);
  }
  if(steps[1]){
    cout << "arriba\n";
    drone.setVerticalSpeed(DRONE_SPEED);
    usleep(TIME_MOVE*0.5);
    drone.hover();
    usleep(2000000);
  }
  else{
    cout << "abajo\n";
    drone.setVerticalSpeed(-DRONE_SPEED);
    usleep(TIME_MOVE*0.5);
    drone.hover();
    usleep(2000000);
  }
  if(steps[2]){
    cout << "adelante\n";
    drone.setPitch(DRONE_SPEED);
    usleep(TIME_MOVE);
    drone.hover();
    usleep(2000000);
  }
  else{
    cout << "atras\n";
    drone.setPitch(-DRONE_SPEED);
    usleep(TIME_MOVE);
    drone.hover();
    usleep(2000000);
  }
}

int main(int argc, char *argv[])
{
  /* Create images where captured and transformed frames are going to be stored */
  cv::Mat image;
  string space = "RGB";
  vector<bool> steps;
  bool freeze = false;
  char key;
  int umbral = 127;

  BebopDrone &drone = BebopDrone::getInstance();

  high_resolution_clock::time_point currentTime = high_resolution_clock::now();
  high_resolution_clock::time_point lastKeyPress = high_resolution_clock::now();
  Mat currentImage, flippedImage, droneImage;
  
  drone.takeoff();
  usleep(3500000);
  cout << "takeoff" << endl;

  drone.hover();

  while (true){
    
    currentTime = high_resolution_clock::now();
    /* Obtain a new frame from camera */
    droneImage = drone.getFrameAsMat();
    if(!freeze) 
      currentImage = droneImage.clone();
    if(space == "HSV"){
      cv::cvtColor(currentImage, image, CV_HSV2BGR);
       
      imshow("Image", image);

      std::vector<std::vector<double>> huMoments;
      huMoments = filterAndSegment(image);
      steps = caracterize(huMoments);

      space = "RGB";
    }
    else{
      image = currentImage.clone();
    }


    imshow("Image", image);

    key = waitKeyEx(3);
    if (key == 'x'){
      break;
    }
    else if (key == 'f'){ 
      freeze = !freeze;
    }
    else if (key == 'b'){ 
      doRoutine(steps, drone);
    }
    else if (key == 'r'){
      space = "RGB";
    }
    else if (key == 'h'){
      space = "HSV";
    }
  }
  drone.land();
}