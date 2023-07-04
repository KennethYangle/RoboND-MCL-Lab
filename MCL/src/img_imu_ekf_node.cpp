#include <iostream>
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/console.h>
#include <tf/tf.h>

#include <Eigen/Geometry>
#include <eigen_conversions/eigen_msg.h>
#include <deque>

#include "img_imu_ekf.h"
#include "mcl_filter.h"

#include <algorithm>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/UInt64.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>

#include "sensor_msgs/image_encodings.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/imgcodecs.hpp>
#include <cv_bridge/cv_bridge.h>

cv::Mat imgCallback;

using namespace std;
using namespace Eigen;

// roslaunch ekf_pkg img_ekf.launch | tee -a `roscd ekf_pkg/log/ && pwd`/`date +%Y%m%d_%H%M%S_fly.log`
img_imu_ekf img_imu;

double pos_ref_time = 0;
double vel_ref_time = 0;
double imu_ref_time = 0;
double imu_pre_time = 0; //这个为系统时钟
double last_time;    //这个为系统时钟
double img_ref_time = 0;
bool time_flag = false;
bool cnt_flag = false;

Vector3d acc;
Vector3d gyro;
Vector2d mav_img_pre;
Vector2d mav_img;
Vector2d img0(640, 360);//图像中心
double img_f = 370;
double img_width, img_height;
Vector3d mav_vel;
Vector3d mav_pos;
Vector3d mav_pos_prev;
Vector3d target_pos;
geometry_msgs::Quaternion mav_quad;
Vector4d mav_q;
Vector4d mav_qq;
double mav_yaw;
double mav_yaw_prev;
double mav_roll;
double mav_pitch;
Eigen::Matrix3d mav_R;
Eigen::Quaterniond mav_quad_eigen;
Eigen::Matrix3d R_cb;
ros::Time timestamp_begin;
ros::Time timestamp;
ros::Time timestamp_prev;

int loop_cnt = 2;
int img_cnt = 0;
bool is_img_come = false;
bool get_img = false;
bool get_pose = false;
bool get_vel = false;
bool get_sensor = false;

MatrixXd Phi[50];
VectorXd ZZ[50];
MatrixXd GG[50];
Vector3d ACC[50];
MatrixXd PP[50]; //有Img后，第一次使用的P
MatrixXd KK;
// MatrixXd GG;
double alpha_1 = 0.0;   // uncertainty weight
double alpha_2 = 0.0;
double alpha_3 = 0.0;
double alpha_4 = 0.0;
double init_pos_conv = 1.0;
double init_orient_conv = 0.1;
double sigma_p = 1.0;

// Create a set of particles
int particle_num = 1000;
vector<Robot> p;

ros::Publisher pub_img_pos, pub_particle, pub_target_marker, pub_sphere_marker;

void resampling_particles();
void publishParticles();
void publishTargetMarkers();
void publishSphereMarkers(Vector3d p_s);

//predict(MatrixXd Phi, MatrixXd X, MatrixXd P, MatrixXd G, MatrixXd Q)
//void update(MatrixXd P, MatrixXd H, MatrixXd R, MatrixXd X, MatrixXd K, MatrixXd Phi, VectorXd Z);


//Img来了之后进行循环,传入的参数为当前循环周期
void loop_img(int loop)
{
    // MatrixXd GG; //定义单位矩阵
    
    //img来了之后，迭代n次
    //第一次迭代只有更新步骤，没有预测步骤
    //PP为第一次img没来，imu产生的协方差预测
    int cnt;
    cnt = img_cnt - loop_cnt + loop + 1;
    if (cnt < 0)
    {
        cnt += 50;
    }

    if(loop == 0)
    {
        KK = PP[cnt] * img_imu.kf.H.transpose() * (img_imu.kf.H * PP[cnt] * img_imu.kf.H.transpose() + img_imu.R).inverse();
        img_imu.kf.update(PP[cnt], img_imu.kf.H, img_imu.Q, img_imu.kf.x, KK, ZZ[img_cnt]); //状态观测为图像来临时的观测
        // PP[img_cnt] = img_imu.kf.P;
    }
    else
    {
        img_imu.kf.predict(Phi[cnt], img_imu.kf.x, img_imu.kf.P, GG[cnt], img_imu.Q);
        if(loop == loop_cnt - 1)
        {
            PP[img_cnt] = img_imu.kf.P;
        }
    }
}


void mav_pose_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    mav_pos(0) = msg->pose.position.x;
    mav_pos(1) = msg->pose.position.y;
    mav_pos(2) = msg->pose.position.z;
    mav_quad = msg->pose.orientation;
    tf::Quaternion RQ2;
    tf::quaternionMsgToTF(mav_quad, RQ2);
    tf::Matrix3x3(RQ2).getRPY(mav_roll, mav_pitch, mav_yaw);
    tf::quaternionMsgToEigen(mav_quad, mav_quad_eigen);
    mav_R = mav_quad_eigen.toRotationMatrix();
    timestamp = msg->header.stamp;

    ROS_DEBUG_STREAM("timestamp: " << (timestamp - timestamp_begin).toSec());
    ROS_DEBUG_STREAM("mav_pos: " << mav_pos);
    if (!get_pose)
    {
      mav_pos_prev = mav_pos;
      mav_yaw_prev = mav_yaw;
      timestamp_prev = timestamp;
      timestamp_begin = timestamp;

      p.clear();
      for (int i = 0; i < particle_num; i++) {
        p.push_back( Robot(mav_pos(0)-target_pos(0), mav_pos(1)-target_pos(1), mav_pos(2)-target_pos(2), mav_yaw, 
                           alpha_1, alpha_2, alpha_3, alpha_4, init_pos_conv, init_orient_conv, sigma_p) );
        ROS_DEBUG_STREAM("particle[" << i << "]_init_pos: " << p[i].x << ", " << p[i].y << ", " << p[i].z);
      }
      get_pose = true;
      return;
    }
    const float dt = (timestamp - timestamp_prev).toSec();
    if (dt < 0.0 || dt > 5.0)
    {
      ROS_WARN("Detected time jump in odometry. Resetting.");
      get_pose = false;
      return;
    }
    else if (dt > 0.02)
    {
      // Simulate a robot motion for each of these particles
      for (int i = 0; i < particle_num; i++) {
        p[i].set_odoms(mav_pos, mav_pos_prev, mav_yaw, mav_yaw_prev, dt);
        p[i].sample_motion_model_simple();
        ROS_DEBUG_STREAM("particle[" << i << "]_pos: " << p[i].x << ", " << p[i].y << ", " << p[i].z);
      }
      mav_pos_prev = mav_pos;
      mav_yaw_prev = mav_yaw;
      timestamp_prev = timestamp;

      // Visualization
      publishParticles();
      publishTargetMarkers();
    }
}

void mav_vel_cb(const geometry_msgs::TwistStamped::ConstPtr &msg)
{
    get_vel = true;
    vel_ref_time = msg->header.stamp.toSec();
    mav_vel(0) = msg->twist.linear.x;
    mav_vel(1) = msg->twist.linear.y;
    mav_vel(2) = msg->twist.linear.z;
}

void mav_imu_cb(const sensor_msgs::Imu::ConstPtr &msg)
{
    img_cnt = (img_cnt + 1)%50;

    if(get_img && get_pose && get_vel)
    {
        get_sensor = true;
    }
    Matrix3d I3 = Matrix3d::Identity(); //定义单位矩阵

    imu_ref_time = ros::Time::now().toSec();

    if (time_flag == false)
    {
        last_time = imu_ref_time;
        imu_pre_time = imu_ref_time;
        time_flag = true;
    }

    acc(0) = msg->linear_acceleration.x;
    acc(1) = msg->linear_acceleration.y;
    acc(2) = msg->linear_acceleration.z;
    gyro(0) = msg->angular_velocity.x;
    gyro(1) = msg->angular_velocity.y;
    gyro(2) = msg->angular_velocity.z;
    mav_q(0) = msg->orientation.w;
    mav_q(1) = msg->orientation.x;
    mav_q(2) = msg->orientation.y;
    mav_q(3) = msg->orientation.z;

    if (img_imu.is_init_done == false && get_sensor == true)
    {
        //初始化的时候应该更新状态X:img_imu.kf.x
        img_imu.sensor_init(mav_q, mav_pos - target_pos, mav_vel, mav_img);
        //初始化的时候应该更新下Phi:img_imu.Phi
        img_imu.update_Phi(gyro, mav_vel, acc, imu_ref_time - imu_pre_time);
        is_img_come = false;//防止未初始化时，Img先于Imu来临。
        img_cnt = 0;
        Phi[img_cnt] = img_imu.Phi;
        GG[img_cnt] = img_imu.G;
        ZZ[img_cnt] = VectorXd::Ones(2);
        ZZ[img_cnt].segment(0, 2) = mav_img;
        // PP[img_cnt] = MatrixXd::Zero(18, 18);
        PP[img_cnt] = img_imu.kf.P;
    }

    if (img_cnt <= loop_cnt && cnt_flag == false)
    {
        is_img_come = false;
        cnt_flag == true;
    }

    if (img_imu.is_init_done == true && get_sensor == true)
    {
        //执行矫正过程
        if (is_img_come)
        {
            // loop_cnt++;
            img_imu.update_Phi(gyro, mav_vel, acc, imu_ref_time - imu_pre_time);
            // PP[img_cnt] = img_imu.kf.P;
            // img_imu.kf.predict(img_imu.Phi, img_imu.kf.x, img_imu.kf.P, img_imu.G, img_imu.Q);
            Phi[img_cnt] = img_imu.Phi;
            GG[img_cnt] = img_imu.G;
            // PP[img_cnt] = img_imu.kf.P;
            ZZ[img_cnt] = VectorXd::Ones(2);
            ZZ[img_cnt].segment(0, 2) = mav_img;
            for (int circle = 0; circle < loop_cnt; circle++)
            {
                loop_img(circle);
            }
            is_img_come = false;
        }
        else
        {
            // loop_cnt++;
            // img_imu.kf.predict(img_imu.Phi, img_imu.kf.x, img_imu.kf.P, img_imu.G, img_imu.Q);  Test
            //Phi是上一时刻获得的，因此要先进行预测步骤，在更新Phi，同时还需要保存本次更新的Phi
            img_imu.update_Phi(gyro, mav_vel, acc, imu_ref_time - imu_pre_time);
            // PP[img_cnt] = img_imu.kf.P;
            img_imu.kf.predict(img_imu.Phi, img_imu.kf.x, img_imu.kf.P, img_imu.G, img_imu.Q);
            //保存这一次的预测协方差阵P,当图像来临时，第一次预测更新用这个协方差阵进行计算
            // if(loop_cnt == 1)
            //     PP = img_imu.kf.P;
            PP[img_cnt] = img_imu.kf.P;
            Phi[img_cnt] = img_imu.Phi;
            GG[img_cnt] = img_imu.G;
        }
        imu_pre_time = imu_ref_time;
        std_msgs::Float32MultiArray img_predict;
        img_predict.data.push_back(img_imu.kf.x(10) * img_f + img0(0));
        img_predict.data.push_back(img_imu.kf.x(11) * img_f + img0(1));
        pub_img_pos.publish(img_predict);
    }
    
}

void mav_img_cb(const std_msgs::Float32MultiArray::ConstPtr &msg)
{
    if (!get_pose) return;

    mav_img_pre = mav_img;
    is_img_come = true;
    get_img = true;

    // Measurement
    Vector3d p_s_c(msg->data[0] - img0(0), msg->data[1] - img0(1), img_f);
    if (msg->data[0] < 0) {
        p_s_c = Vector3d(0., 0., 0.);
    }
    else {
        p_s_c.normalize();
    }
    Vector3d p_s = mav_R * R_cb * p_s_c;

    //Generate particle weights depending on robot's measurement
    for (int i = 0; i < p.size(); i++) {
        p[i].landmark_model_likelyhood_simple(p_s);
    }

    //Resample the particles with a sample probability proportional to the importance weight
    resampling_particles();

    // Visualization
    publishSphereMarkers(p_s);
}

void img_show_cb(const sensor_msgs::CompressedImage::ConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr_compressed = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    imgCallback = cv_ptr_compressed->image;
    if (get_sensor)
    {
        cv::circle(imgCallback, cv::Point(img_imu.kf.x(10) * img_f + img0(0), img_imu.kf.x(11) * img_f + img0(1)), 5, cv::Scalar(255, 0, 0), 2);
        cv::circle(imgCallback, cv::Point(mav_img(0) * img_f + img0(0), mav_img(1) * img_f + img0(1)), 5, cv::Scalar(255, 255, 0), 1);
    }
    // cv::circle(imgCallback, cv::Point(360, 210), 5, cv::Scalar(255, 0, 0), 2);

    cv::imshow("imgCallback", imgCallback);
    cv::waitKey(1);
}

void ekf_state_cb(const std_msgs::UInt64::ConstPtr &msg)
{
    if (msg->data == 0)
    {
        img_imu.is_init_done = false;
    }
}

void resampling_particles()
{
    vector<Robot> p2;
    int index = gen_real_random() * particle_num;
    double beta = 0.0;
    double mq = 0;
    for (int i = 0; i < particle_num; i++) {
        if (mq < p[i].q) {
            mq = p[i].q;
        }
    }

    for (int i = 0; i < particle_num; i++) {
        beta += gen_real_random() * 2.0 * mq;
        while (beta > p[index].q) {
            beta -= p[index].q;
            index = mod((index + 1), particle_num);
        }
        p2.push_back(p[index]);
    }
    p.assign(p2.begin(), p2.end());
}

// Draw arrows
void publishParticles()
{
    geometry_msgs::PoseArray pa;
    pa.header.stamp = ros::Time::now();
    pa.header.frame_id = "map";
    for (size_t i = 0; i < particle_num; i++)
    {
        geometry_msgs::Pose pm;
        pm.position.x = 0;
        pm.position.y = 0;
        pm.position.z = 0;
        pm.orientation = tf::createQuaternionMsgFromRollPitchYaw(mav_roll, mav_pitch, p[i].orient);

        pa.poses.push_back(pm);
    }
    pub_particle.publish(pa);
}

// Draw target particles
void publishTargetMarkers()
{
    visualization_msgs::MarkerArray markers;//定义MarkerArray对象
	for(int i = 0; i < particle_num; i++)
	{
		visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
		marker.type = visualization_msgs::Marker::SPHERE;
		marker.ns = "Target points";
		marker.id = i; //用来标记同一帧不同的对象，如果后面的帧的对象少于前面帧的对象，那么少的id将在rviz中残留，所以需要后续的实时更新程序
        marker.pose.position.x = -p[i].x;
        marker.pose.position.y = -p[i].y;
        marker.pose.position.z = -p[i].z;
		marker.pose.orientation.w = 1.0;
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2;
		marker.color.a = 1.0;
		marker.color.r = 1.0;
		marker.color.g = 0.0;
		marker.color.b = 1.0;
        marker.lifetime = ros::Duration(0.2);
        markers.markers.push_back(marker);
	}
    pub_target_marker.publish(markers);
}

// Draw target particles on sphere
void publishSphereMarkers(Vector3d p_s)
{
    visualization_msgs::MarkerArray markers;//定义MarkerArray对象
    // p_s_hat
	for(int i = 0; i < particle_num; i++)
	{
		visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
		marker.type = visualization_msgs::Marker::SPHERE;
		marker.ns = "Sphere estimate points";
		marker.id = i;
        marker.pose.position.x = p[i].p_s_hat(0);
        marker.pose.position.y = p[i].p_s_hat(1);
        marker.pose.position.z = p[i].p_s_hat(2);
		marker.pose.orientation.w = 1.0;
        marker.scale.x = marker.scale.y = marker.scale.z = max(p[i].q/2, 0.1);
		marker.color.a = 1.0;
		marker.color.r = 0.0;
		marker.color.g = 1.0;
		marker.color.b = 0.0;
        marker.lifetime = ros::Duration(0.2);
        markers.markers.push_back(marker);
	}

    // p_s
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.ns = "Sphere points";
    marker.id = particle_num;
    marker.pose.position.x = p_s(0);
    marker.pose.position.y = p_s(1);
    marker.pose.position.z = p_s(2);
    marker.pose.orientation.w = 1.0;
    marker.scale.x = marker.scale.y = marker.scale.z = 0.2;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.lifetime = ros::Duration(0.2);
    markers.markers.push_back(marker);

    // Barycente
    visualization_msgs::Marker marker_arrow;
    marker_arrow.header.frame_id = "map";
    marker_arrow.header.stamp = ros::Time::now();
    marker_arrow.type = visualization_msgs::Marker::ARROW;
    marker_arrow.ns = "Barycente arrow";
    marker_arrow.id = particle_num + 1;
    marker_arrow.scale.x = 0.1;
    marker_arrow.scale.y = 0.16;
    marker_arrow.scale.z = 0.2;
    marker_arrow.color.a = 1.0;
    marker_arrow.color.r = 0.3;
    marker_arrow.color.g = 1.0;
    marker_arrow.color.b = 0.3;

    geometry_msgs::Point pt1, pt2;
    pt1.x = 0.;   // start point
    pt1.y = 0.;
    pt1.z = 0.;
    // end point
	for(int i = 0; i < particle_num; i++)
    {
        pt2.x += p[i].p_s_hat(0) * p[i].q;
        pt2.y += p[i].p_s_hat(1) * p[i].q;
        pt2.z += p[i].p_s_hat(2) * p[i].q;
    }
    double pt2_norm = sqrt(pt2.x*pt2.x + pt2.y*pt2.y + pt2.z*pt2.z);
    pt2.x = pt2.x / pt2_norm * 1.5;
    pt2.y = pt2.y / pt2_norm * 1.5;
    pt2.z = pt2.z / pt2_norm * 1.5;

    marker_arrow.points.push_back(pt1);
    marker_arrow.points.push_back(pt2);
    marker_arrow.lifetime = ros::Duration(0.2);
    markers.markers.push_back(marker_arrow);

    pub_sphere_marker.publish(markers);
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "mcl_node");
    ros::NodeHandle nh;

    ros::Subscriber pos_sub = nh.subscribe<geometry_msgs::PoseStamped>
                ("mavros/local_position/pose", 10, mav_pose_cb);   //50hz
    ros::Subscriber vel_sub = nh.subscribe<geometry_msgs::TwistStamped>
                ("mavros/local_position/velocity_local", 10, mav_vel_cb);//50hz
    ros::Subscriber imu_sub = nh.subscribe<sensor_msgs::Imu>
                ("/mavros/imu/data", 10, mav_imu_cb); //50hz
    ros::Subscriber img_sub = nh.subscribe<std_msgs::Float32MultiArray>
                ("tracker/pos_image", 10, mav_img_cb);          //23hz
    ros::Subscriber img_show_sub = nh.subscribe<sensor_msgs::CompressedImage>
                ("/image_raw/compressed", 10, img_show_cb); //21hz
    ros::Subscriber ekf_state_sub = nh.subscribe<std_msgs::UInt64>
                ("ekf/state", 1, ekf_state_cb); //21hz

    pub_particle = nh.advertise<geometry_msgs::PoseArray>("particles", 1, true);
    pub_target_marker = nh.advertise<visualization_msgs::MarkerArray>("target_marker", 1, true);
    pub_sphere_marker = nh.advertise<visualization_msgs::MarkerArray>("sphere_marker", 1, true);
    pub_img_pos = nh.advertise<std_msgs::Float32MultiArray>
                ("tracker/pos_image_ekf", 10);

    nh.param<int>("/particle_num", particle_num, 100);
    ROS_INFO_STREAM_ONCE("particle_num: " << particle_num);
    nh.param<double>("/MCL_params/alpha_1", alpha_1, 0.0);
    nh.param<double>("/MCL_params/alpha_2", alpha_2, 0.0);
    nh.param<double>("/MCL_params/alpha_3", alpha_3, 0.0);
    nh.param<double>("/MCL_params/alpha_4", alpha_4, 0.0);
    ROS_INFO_STREAM_ONCE("alpha_1: " << alpha_1 << ", alpha_2: " << alpha_2 << ", alpha_3: " << alpha_3 << ", alpha_4: " << alpha_4);
    nh.param<double>("/MCL_params/init_pos_conv", init_pos_conv, 1.0);
    nh.param<double>("/MCL_params/init_orient_conv", init_orient_conv, 0.1);
    nh.param<double>("/MCL_params/sigma_p", sigma_p, 1.0);
    ROS_INFO_STREAM_ONCE("init_pos_conv: " << init_pos_conv << ", init_orient_conv: " << init_orient_conv << ", sigma_p: " << sigma_p);
    nh.param<double>("/target_pos/target_x", target_pos(0), 0.0);
    nh.param<double>("/target_pos/target_y", target_pos(1), 0.0);
    nh.param<double>("/target_pos/target_z", target_pos(2), 0.0);
    ROS_INFO_STREAM_ONCE("target_pos: " << target_pos);
    nh.param<double>("/camera/img_width", img_width, 0.0);
    nh.param<double>("/camera/img_height", img_height, 0.0);
    img0 = Vector2d(img_width/2, img_height/2);
    nh.param<double>("/camera/img_f", img_f, 0.0);
    ROS_INFO_STREAM_ONCE("img_width: " << img_width << ", img_height: " << img_height << ", img_f: " << img_f);
    R_cb << 0, 0, 1, -1, 0, 0, 0, -1, 0;


    // ros::spin();
    ros::AsyncSpinner spinner(4);
    spinner.start();
    ros::waitForShutdown();

    return 0;
}
