//
//  main.cpp
//  MyOpenCV
//
//  Created by xietian on 13-9-10.
//  Copyright (c) 2013年 tiange. All rights reserved.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

/** Function Headers */
void detectAndDisplay( Mat frameIn, Ptr<CLAHE> calhe);

/** Global variables */
String face_cascade_name = "/Users/apple/Downloads/opencv-2.4.6/data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/Users/apple/Downloads/opencv-2.4.6/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);


int firstTest()
{
    Mat img = imread("/Users/apple/Pictures/IMG_0783.JPG"); //Change the image path here.
    if (img.data == 0) {
        cerr << "Image not found!" << endl;
        return -1;
    }
    namedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", img);
    waitKey();
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
    
}

void rotateImage(const Mat& src,Mat& dest, double angle)
{
    
    //int height = srcImage->height;
    //int width = srcImage->width;
    //IplImage *rotatedImg = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U,srcImage->nChannels);
    CvPoint2D32f center;
    center.x = src.rows/2.0;
    center.y = src.cols/2.0f;
    //CvMat *mapMatrix = cvCreateMat( 2, 3, CV_32FC1 );
    
    cout << "x:" << center.x << ",y:" << center.y << endl;
    
    Mat mapMatrix = getRotationMatrix2D(center, angle, 1.0);
    cout << "rotate matrix: rows:" << mapMatrix.rows << ",col:" << mapMatrix.cols << endl;
    printf("matrix value:");
    double* data = (double*)mapMatrix.data;
    for(int i = 0; i < mapMatrix.rows; i ++){
        for(int col = 0; col < mapMatrix.cols; col ++){
            printf("%.5f ", data[i * mapMatrix.rows + col]);
        }
        cout << endl;
    }
    warpAffine(src, dest, mapMatrix, dest.size());
}

//It seems all the input data are treat as matrix.
//Take time to really understand it.
//Because I will work on this for the rest of your life.
//Just one method call to achieve the normalization of the matrix, which really really make the
//The whole machine learning process like a breeze.
//I love this game more and more.
static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

//What I have learnt from this method call?
//I could return & as parameter, what's the behavior of this method?
//The compile knew that the method should return the things used by the caller.
//So it will do something with the caller I guess, mean, it basically turned in this style.
//In the end, this return value style is just a syntax candy.
//but enough syntax candy to get you a happy interface though.
void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    //return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

static int getLabel(const std::string& label)
{
    std::vector<std::string> elems;
    split(label, '/', elems);
    std::cout << "Elements is:" << elems[4] << "," << elems[5] << endl;
    return atoi(elems[5].substr(1).c_str());
    
}

static void matToImage(Mat& mat, const std::string& fileName);

static void read_files(const string& filename, vector<Mat>& images, vector<int>& labels)
{
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        //stringstream liness(line);
        //getline(liness, path, separator);
        //getline(liness, classlabel);
        std::string path = line;
        //std::vector<std::string> elems;
        //split(line, '/', elems);
        int curLabel = getLabel(line);
        if(!path.empty()) {
            Mat imgMat = imread(path, 0);
            images.push_back(imgMat);
            matToImage(imgMat, path);
            labels.push_back(curLabel);
        }
    }
    
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

//Make it simple and stupid.
const char* changePostfix(const std::string& path,const std::string& oldFix, const std::string& postFix)
{
    std::string res = path.substr(0, path.size()-oldFix.size());
    //cout << "removed old fix:" << res;
    return res.append(postFix).c_str();
}
//This method will do the eigne face test in this method.
//This is my first shot. Enjoy it man.
//This is a very specific implementation.
static void matToImage(Mat& mat, const std::string& fileName)
{
    string changedName = changePostfix(fileName, "pgm", "png");
    imwrite(changedName, mat);
}


static void predict(Ptr<FaceRecognizer> fr, const string& fileName)
{
    double confidence;
    int label;
    Mat imgMat = imread(fileName, 0);
    //cout << "before predict:" << fileName << endl;
    fr->predict(imgMat, label, confidence);
    string msg = format(",confident:%.5f", confidence);
    cout << "fileName:" << fileName << ",label:" << label << msg << endl;
}

Ptr<FaceRecognizer> trainModel(string& fn_csv, string& output_folder, bool isFisher)
{
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_files(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    //Great, have turn the image to matrix, so that you are free to do whatever you like to the image now.
    //I hate to do the things out skirt, I like to strike the stem of the evil and turn myself into NEO
    //I am, every one of us is NEO, we are actually limited by the cone created by our generic being
    //And our upbringing. The only way to break away from it and gain the freedom is keep typing code and keep learning.
    //Nothing could be more firmly move you toward the right direction.
    //My feeling tell me, only by typing the code and genuiely understood what you are doing.
    
    //int height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    //Mat testSample = images[images.size() - 1];
    //int testLabel = labels[labels.size() - 1];
    //images.pop_back();
    //labels.pop_back();
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    if(isFisher){
        model = createFisherFaceRecognizer();
    }
    model->train(images, labels);
    return model;
}

void engineFace(string& fn_csv, string& output_folder, bool isFisher)
{
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    // Get the path to your CSV.
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_files(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    // The following lines create an Eigenfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // This here is a full PCA, if you just want to keep
    // 10 principal components (read Eigenfaces), then call
    // the factory method like this:
    //
    //      cv::createEigenFaceRecognizer(10);
    //
    // If you want to create a FaceRecognizer with a
    // confidence threshold (e.g. 123.0), call it with:
    //
    //      cv::createEigenFaceRecognizer(10, 123.0);
    //
    // If you want to use _all_ Eigenfaces and have a threshold,
    // then call the method like this:
    //
    //      cv::createEigenFaceRecognizer(0, 123.0);
    //
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    if(isFisher){
        model = createFisherFaceRecognizer();
    }
    model->train(images, labels);
    // The following line predicts the label of a given
    // test image:
    
    int predictedLabel = model->predict(testSample);
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;
    // Here is how to get the eigenvalues of this Eigenfaces model:
    Mat eigenvalues = model->getMat("eigenvalues");
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getMat("eigenvectors");
    // Get the sample mean from the training data
    Mat mean = model->getMat("mean");
    // Display or save:
    //if(argc == 2) {
    //    imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
    //} else {
    imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
    //}
    // Display or save the Eigenfaces:
    for (int i = 0; i < min(10, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Jet colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        // Display or save:
        //if(argc == 2) {
        //    imshow(format("eigenface_%d", i), cgrayscale);
        //} else {
        imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(grayscale));
        //}
    }
    
    // Display or save the image reconstruction at some predefined steps:
    for(int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components+=15) {
        // slice the eigenvectors from the model
        Mat evs = Mat(W, Range::all(), Range(0, num_components));
        Mat projection = subspaceProject(evs, mean, images[0].reshape(1,1));
        Mat reconstruction = subspaceReconstruct(evs, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        // Display or save:
        //if(argc == 2) {
        //    imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
        //} else {
        imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
        //}
    }
    // Display if we are not writing to an output folder:
    //if(argc == 2) {
    //    waitKey(0);
    //}
    //return 0;
}

int startCapture();

int main(int argc, const char * argv[])
{
    //int label = getLabel("/Users/apple/Downloads/orl_faces/s1/1.pgm");
    //cout << "Label is:"<< label << endl;
    string fn_csv = "/Users/apple/Downloads/fisher_faces/faces.txt";
    string output_folder = "/Users/apple/Downloads/fisher_faces/outputs";
    //engineFace(fn_csv, output_folder, true);
    //string res = changePostfix("Tiange.pgm", "pgm", "png");
    //cout << "replaced:" << res << endl;
    //Ptr<FaceRecognizer> model = trainModel(fn_csv, output_folder, true);
    //for(int i = 1; i < 41; i++){
    //    string fileName = format("/Users/apple/Downloads/fisher_faces/s%i/7.pgm", i);
    //    predict(model, fileName);
    //}
    startCapture();
}


int startCapture()
{
    CvCapture* capture;
    Mat frame;
    
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading: %s\n", face_cascade_name.c_str()); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading: %s\n", eyes_cascade_name.c_str()); return -1; };
    
    //-- 2. Read the video stream
    capture = cvCaptureFromCAM( -1 );
    Ptr<CLAHE> calhe =  createCLAHE();
    
    if( capture )
    {
        while( true )
        {
            frame = cvQueryFrame( capture );
            
            //-- 3. Apply the classifier to the frame
            if( !frame.empty() )
            { detectAndDisplay( frame, calhe ); }
            else
            { printf(" --(!) No captured frame -- Break!"); break; }
            
            int c = waitKey(20000);
            if( (char)c == 'c' ) {
                printf("Capture next");
            }
        }
    }
    return 0;
}

void rotateMat(Mat frame, Mat& res, double angle)
{
 
    double x = frame.rows;
    double y = frame.cols;
    Point2f center(x/2.0,y/2.0);
    
    Mat rotateMat = cv::getRotationMatrix2D(center, angle, 1);
    cout << format("center is: %f, %f", x/2.0, y/2.0);
    //Mat res;
    cv::transform(frame, res, rotateMat);
    //return res;
}

void detectAndDisplay( Mat frameIn, Ptr<CLAHE> calhe)
{
    std::vector<Rect> faces;
    std::vector<Rect> oldFaces;
    Mat frame_gray;
    Mat frame_old;
    cvtColor( frameIn, frame_gray, CV_BGR2GRAY );
    cvtColor(frameIn, frame_old, CV_BGR2GRAY);
    //equalizeHist( frame_gray, frame_gray );
    calhe->apply(frame_gray, frame_gray);
    equalizeHist(frame_old, frame_old);
    Mat frame = frameIn;
    Mat screenFrame = frame_old;
    //-- Detect faces
    
    face_cascade.detectMultiScale( frame_old, oldFaces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    for(int angleCount = 1; angleCount < 12; angleCount ++){
    
        face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
       
        
        cout << "Old face:" << oldFaces.size() << ", new face:" << faces.size() << endl;
        
        if(faces.size() > oldFaces.size()){
            cout << "Beat old face" << endl;
        }else if(faces.size() < oldFaces.size()){
            cout << "Defeat by old face" << endl;
        }

        for( int i = 0; i < faces.size(); i++ )
        {
            Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        
            ellipse( screenFrame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        
            Mat faceROI = frame_gray( faces[i] );
            std::vector<Rect> eyes;
        
            //-- In each face, detect eyes
            eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        
            for( int j = 0; j < eyes.size(); j++ )
            {
                Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                circle( screenFrame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
            }
        }
        if(faces.size() > 0){
            break;
        }
        double curAngle = angleCount * 30.0;
        cout << format("Will rotate to angle %f, see what's going on\n", curAngle);
        Mat frameTemp;
        rotateImage(frameIn, frameTemp, curAngle);
        cvtColor(frameTemp, frame_gray, CV_BGR2GRAY);
        calhe->apply(frame_gray, frame_gray);
        screenFrame = frame_gray;
       // cvtColor(frame, frame_gray, CV_BGR2GRAY );
       // equalizeHist( frame_gray, frame_gray );
    }
    //-- Show what you got
    imshow( window_name, screenFrame);
}

