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
#include <utility>
#include <stdlib.h>
#include <climits>

#define RESTRICT_MOVEMENT 1 //Debe ser 1 para la presentación

#define FILTER_SAMPLE 15
#define TOLERANCE 45

#define MINIMUM_AREA 30000

#define BLACK cv::Vec3b(0, 0, 0)
#define BLUE cv::Vec3b(255, 0, 0)
#define GREEN cv::Vec3b(0, 255, 0)
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
#define SHIRT_PH1_AVG 0.000776667
#define SHIRT_PH2_AVG 6.54703E-08
#define SHIRT_PH1_VAR 2.6841E-05 *2 
#define SHIRT_PH2_VAR 1.10927E-08 *2

#define xMinRing 130
#define xMaxRing 249
#define yMinRing 0
#define yMaxRing 3
#define xMinTie 354
#define xMaxTie 512
#define yMinTie 206
#define yMaxTie 361
#define xMinPants 170
#define xMaxPants 440
#define yMinPants 36
#define yMaxPants 96
#define xMinShirt 11
#define xMaxShirt 66
#define yMinShirt 4
#define yMaxShirt 11
#define xMeanRing 189
#define xMeanTie 439
#define xMeanPants 305
#define xMeanShirt 39
#define yMeanRing 1
#define yMeanTie 284
#define yMeanPants 67
#define yMeanShirt 8

#define DRONE_SPEED 50
#define TIME_MOVE 2000000
#define TIME_IN_CM 200000

using namespace std;
using namespace cv;
using namespace std::chrono;

const int KEY_DELAY_MS = 50;
const int DRONE_OUTPUT = 70;

const char KEY_CALIBRATE = 'c';
const char KEY_BEGIN = 'b';
const char KEY_STOP_PROGRAM = 'x';
const char KEY_TAKEOFF = 'o';
const char KEY_LAND = 'p';
const char KEY_MOVE_FORWARD = 'w';
const char KEY_MOVE_BACK = 's';
const char KEY_MOVE_LEFT = 'a';
const char KEY_MOVE_RIGHT = 'd';
const char KEY_TURN_LEFT = 'q';
const char KEY_TURN_RIGHT = 'e';
const char KEY_EMERGENCY_STOP = 'f';
const char KEY_HOVER = 'r';
const char KEY_SEQUENCE = 'k';

const String WINDOW_ORIGINAL_NAME = "Original";
const String WINDOW_FLIPPED_NAME = "Flipped";

void flipImageBasic(const Mat &sourceImage, Mat &destinationImage);

std::vector<cv::Vec3b> pixels;

std::vector<cv::Vec3b> listColor;

vector<int> limits = {32,122,132,222,75,165};

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
  int data[3] = {0, 0, 0};
  cv::Mat image = hsvImage;
  cv::setMouseCallback("Image", mouseEvent, &image);
  cv::Mat filteredImage;
  pixels.clear();
  GaussianBlur(hsvImage, hsvImage, Size(5, 5), 0);
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
  listColor.clear();
  while(numTriesToDetectObject < image.cols * image.rows) {
    row = randomRow(rng);
    col = randomCol(rng);
    numTriesToDetectObject++;
    if (image.at<cv::Vec3b>(row, col) == objectColor) {
      cv::Vec3b color = getRandomColor(RANDOM_COLOR_MIN, RANDOM_COLOR_MAX);
      listColor.push_back(color);
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

int getPosPhi1(double x)
{
  int pos = (int)((x - 0.0007) * 512.0 / (0.0017- 0.0007));
  if (pos > 512) pos = 512;
  if (pos < 0) pos = 0;
  return pos;
}

int getPosPhi2(double x)
{
  return (int)((x - 1.4E-8) * 400.0 / (2.5E-6 - 1.4E-8));
}

void drawSquare(cv::Mat &image, int xMin, int xAv, int xMax, int yMin, int yAv, int yMax, cv::Scalar color){
  cv::line(image, cv::Point(xAv, yMin), cv::Point(xMax, yAv), color);
  cv::line(image, cv::Point(xMax, yAv), cv::Point(xAv, yMax), color);
  cv::line(image, cv::Point(xAv, yMax), cv::Point(xMin, yAv), color);
  cv::line(image, cv::Point(xMin, yAv), cv::Point(xAv, yMin), color);
}

void flipImageBasic(const Mat &sourceImage, Mat &destinationImage)
{
  if (destinationImage.empty())
    destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

  for (int x = 0; x < sourceImage.cols; ++x)
    for (int y = 0; y < sourceImage.rows / 2; ++y)
      for (int i = 0; i < sourceImage.channels(); ++i)
      {
        destinationImage.at<Vec3b>(y, x)[i] = sourceImage.at<Vec3b>(sourceImage.rows-1-y, x)[i];
        destinationImage.at<Vec3b>(sourceImage.rows-1-y, x)[i] = sourceImage.at<Vec3b>(y, x)[i];
      }
}

void drawGraph(std::vector<std::vector<double>> hM){
  int graph_w = 512; int graph_h = 400;
  
  Mat graph( graph_h, graph_w, CV_8UC3, Scalar( 0,0,0) );
  Mat flipped = graph.clone();
  
  drawSquare(graph, xMinRing, xMeanRing, xMaxRing, yMinRing, yMeanRing, yMaxRing, cv::Scalar(0,255,255));
  drawSquare(graph, xMinTie, xMeanTie, xMaxTie, yMinTie, yMeanTie, yMaxTie, cv::Scalar(255,255,0));
  drawSquare(graph, xMinPants, xMeanPants, xMaxPants, yMinPants, yMeanPants, yMaxPants, cv::Scalar(0,0,255));
  drawSquare(graph, xMinShirt, xMeanShirt, xMaxShirt, yMinShirt, yMeanShirt, yMaxShirt, cv::Scalar(200,200,200));
  
  for( int i = 0; i < hM.size(); i++ ){
    int xPos = getPosPhi1(hM[i][0]);
    int yPos = getPosPhi2(hM[i][1]);
    if (xPos <= 512 && xPos >= 0){
      if (yPos <= 400 && yPos >= 0){
        circle(graph, cv::Point(xPos, yPos), 3, listColor[i], CV_FILLED);
      }
    }
  }
    
  flipImageBasic(graph, flipped);

  namedWindow("Clasificacion", CV_WINDOW_AUTOSIZE );
  imshow("Clasificacion", flipped );
}

std::vector<std::vector<double>> getMoments(cv::Mat &image){
  cv::Mat filteredImage;

  GaussianBlur(image, image, Size(5, 5), 0);

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
  
  drawGraph(huMoments);

  return huMoments;
}

vector<bool> caracterize(std::vector<std::vector<double>> huMoments){
  float phi1,phi2,angle,x,y;
  vector<bool> steps (3);
  cout << "------------------------------------------------------------------------------------\n"; 
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
        steps[2] = false;//F2
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
        steps[2] = true;//F1
      } 
    }
  }

  return steps;
}

void moveInCm(string direction, BebopDrone &drone, cv::Point &position, cv::Mat &droneMap){
  cv::Point lastPosition(position.x, position.y);
  if(direction == "left"){
    drone.setRoll(-DRONE_SPEED);
    position.x += 10;
  }
  else if(direction == "right"){
    drone.setRoll(DRONE_SPEED);
    position.x -= 10;
  }
  else if(direction == "back"){
    drone.setPitch(-DRONE_SPEED);
    position.y -= 10;
  }
  else if(direction == "forward"){
    drone.setPitch(DRONE_SPEED);
    position.y += 10;
  }
  else{
    cout << "Incorrect direction.\n";
    return;
  }
  usleep(TIME_IN_CM);
  drone.hover();
  cv::line(droneMap, lastPosition, position, RED);
  imshow("Drone map",droneMap);
  waitKey(200);
}

void drawMaps(cv::Mat &objectMap, cv::Mat &droneMap, cv::Mat &NF1Map, vector<bool> steps){
  bool isWayFree[5][3];
  cv::Mat temporal( 410, 410, CV_8UC3, Scalar( 255,255,255) );
  objectMap = temporal.clone();
  cv::circle(objectMap, cv::Point(205, 123), 13, BLACK, CV_FILLED);
  cv::circle(objectMap, cv::Point(205, 287), 13, BLACK, CV_FILLED);
  droneMap = objectMap.clone();
  cv::circle(droneMap, cv::Point(205, 123), 22, BLACK, CV_FILLED);
  cv::circle(droneMap, cv::Point(205, 287), 22, BLACK, CV_FILLED);
  NF1Map = droneMap.clone();
  cv::line(NF1Map, cv::Point(0, 82), cv::Point(410, 82), BLACK);
  cv::line(NF1Map, cv::Point(0, 164), cv::Point(410, 164), BLACK);
  cv::line(NF1Map, cv::Point(0, 246), cv::Point(410, 246), BLACK);
  cv::line(NF1Map, cv::Point(0, 328), cv::Point(410, 328), BLACK);
  cv::line(NF1Map, cv::Point(137, 0), cv::Point(137, 410), BLACK);
  cv::line(NF1Map, cv::Point(274, 0), cv::Point(274, 410), BLACK);
  for(int i = 0; i < 5; i++){
    for(int j = 0; j < 3; j++){
      if(NF1Map.at<Vec3b>(i*82+41, j*137+68)[0] == 0){
        rectangle(NF1Map, cv::Point(j*137-1, i*82-1), cv::Point((j+1)*137-1, (i+1)*82-1), cv::Scalar(0,0,0), CV_FILLED);
        isWayFree[i][j] = false;
      }
      else{
        isWayFree[i][j] = true;
      }
    } 
  }
  if(!steps[0]){
    rectangle(NF1Map, cv::Point(1,409), cv::Point(136,329), cv::Scalar(127,127,127), CV_FILLED);
    rectangle(NF1Map, cv::Point(1,327), cv::Point(136,247), cv::Scalar(127,127,127), CV_FILLED);
    rectangle(NF1Map, cv::Point(1,245), cv::Point(136,165), cv::Scalar(127,127,127), CV_FILLED);
    if(steps[2]){
      rectangle(NF1Map, cv::Point(138,245), cv::Point(273,165), cv::Scalar(127,127,127), CV_FILLED);
    }
    else{
      rectangle(NF1Map, cv::Point(1,163), cv::Point(136,83), cv::Scalar(127,127,127), CV_FILLED);
      rectangle(NF1Map, cv::Point(1,81), cv::Point(136,1), cv::Scalar(127,127,127), CV_FILLED);
      rectangle(NF1Map, cv::Point(138,81), cv::Point(273,1), cv::Scalar(127,127,127), CV_FILLED);
    }
  }
  else{
    rectangle(NF1Map, cv::Point(275,409), cv::Point(409,329), cv::Scalar(127,127,127), CV_FILLED);
    rectangle(NF1Map, cv::Point(275,327), cv::Point(409,247), cv::Scalar(127,127,127), CV_FILLED);
    rectangle(NF1Map, cv::Point(275,245), cv::Point(409,165), cv::Scalar(127,127,127), CV_FILLED);
    if(steps[2]){
      rectangle(NF1Map, cv::Point(138,245), cv::Point(273,165), cv::Scalar(127,127,127), CV_FILLED);
    }
    else{
      rectangle(NF1Map, cv::Point(275,163), cv::Point(409,83), cv::Scalar(127,127,127), CV_FILLED);
      rectangle(NF1Map, cv::Point(275,81), cv::Point(409,1), cv::Scalar(127,127,127), CV_FILLED);
      rectangle(NF1Map, cv::Point(138,81), cv::Point(273,1), cv::Scalar(127,127,127), CV_FILLED);
    }
  }
  imshow("Object map", objectMap);
  imshow("Drone map", droneMap);
  imshow("NF! map", NF1Map);
}

void doRoutine(vector<bool> steps, BebopDrone &drone, cv::Mat &droneMap){
  cv::Point position(205, 369);

  //Se mueve a la izquierda o derecha
  if(steps[0]){
    for(int i = 0; i < 12*RESTRICT_MOVEMENT; i++){
      moveInCm("left", drone, position, droneMap);
    }
  }
  else{
    for(int i = 0; i < 12*RESTRICT_MOVEMENT; i++){
      moveInCm("right", drone, position, droneMap);
    }
  }

  usleep(2000000);

  //Sube o baja
  if(steps[1])
    drone.setVerticalSpeed(DRONE_SPEED);
  else
    drone.setVerticalSpeed(-DRONE_SPEED);
  usleep(TIME_MOVE*0.5);
  drone.hover();

  usleep(2000000);

  //Va a F1 o F2
  if(steps[2]){
    for(int i = 0; i < 16*RESTRICT_MOVEMENT; i++){
      moveInCm("back", drone, position, droneMap);
    }
  }
  else{
    for(int i = 0; i < 33*RESTRICT_MOVEMENT; i++){
      moveInCm("back", drone, position, droneMap);
    }
  }

  usleep(2000000);

  //Entra a la posicion final
  if(steps[0]){
    for(int i = 0; i < 12*RESTRICT_MOVEMENT; i++){
      moveInCm("right", drone, position, droneMap);
    }
  }
  else{
    for(int i = 0; i < 12*RESTRICT_MOVEMENT; i++){
      moveInCm("left", drone, position, droneMap);
    }
  }
}

int main(int argc, char *argv[])
{
  /* Create images where captured and transformed frames are going to be stored */
  cv::Mat image;
  cv::Mat objectMap; 
  cv::Mat droneMap; 
  cv::Mat NF1Map;
  string space = "RGB";
  vector<pair<string,int>> path;
  vector<bool> steps;
  char key;
  int umbral = 127;

  VideoCapture camera = VideoCapture(0);

  BebopDrone &drone = BebopDrone::getInstance();

  high_resolution_clock::time_point currentTime = high_resolution_clock::now();
  high_resolution_clock::time_point lastKeyPress = high_resolution_clock::now();
  Mat currentImage, flippedImage, droneImage;
  
  cout << "Battery Level: " << drone.getBatteryLevel() << endl;
  int count = 0;
  bool run = false;
  bool stop = false;
  bool calib = false;
  while (!stop){
    
    currentTime = high_resolution_clock::now();
    /* Obtain a new frame from camera */
    droneImage = drone.getFrameAsMat();
    currentImage = droneImage.clone();
    imshow("Drone", droneImage);
    key = tolower(cv::waitKey(3));

    cv::cvtColor(currentImage, image, CV_HSV2BGR);

    if(calib){
      imshow("Image", image);
      limits = filterHSV(image);
      for(auto limit : limits){
        cout << limit << " ";
      }
      cout << endl;
      calib = false;
    }

    count++;
    
    std::vector<std::vector<double>> huMoments;
    huMoments = getMoments(image);

    if (key == KEY_STOP_PROGRAM) {
      break;
    }
    if (key != -1) {
      lastKeyPress = high_resolution_clock::now();
    }
    
    if(run){
      cout << "-----------------------------Run Caracterize------------------------\n\n";
      steps = caracterize(huMoments);
      usleep(2000000);
      path.clear();
      drawMaps(objectMap, droneMap, NF1Map, steps);
      waitKey(0);
      doRoutine(steps, drone, droneMap);
      drone.land();
      run = false;
      //break;
    }
    else{
      caracterize(huMoments);
    }
    
    switch (key) {
      case KEY_CALIBRATE:
        calib = !calib;
        break;
      case KEY_BEGIN:
        run = !run;
        break;
      case KEY_TAKEOFF:
        drone.takeoff();
        break;
      case KEY_LAND:
        drone.land();
        break;
      case KEY_MOVE_FORWARD:
        drone.setPitch(DRONE_OUTPUT);
        break;
      case KEY_MOVE_BACK:
        drone.setPitch(-DRONE_OUTPUT);
        break;
      case KEY_MOVE_LEFT:
        drone.setRoll(-DRONE_OUTPUT);
        break;
      case KEY_MOVE_RIGHT:
        drone.setRoll(DRONE_OUTPUT);
        break;
      case KEY_TURN_LEFT:
        drone.setYaw(-DRONE_OUTPUT);
        break;
      case KEY_TURN_RIGHT:
        drone.setYaw(DRONE_OUTPUT);
        break;
      case KEY_EMERGENCY_STOP:
        drone.emergencyStop();
        break;
      case KEY_HOVER:
        drone.hover();
        break;
      default:
        // If key delay has passed and no new keys have been pressed stop
        // drone
        duration<double, std::milli> time_span = currentTime - lastKeyPress;
        if (time_span.count() > KEY_DELAY_MS) {
          drone.hover();
        }
        break;
    }
  }
  cout << "Battery Level: " << drone.getBatteryLevel() << endl;
}