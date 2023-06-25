#ifndef __IMG_IMU_EKF_H
#define __IMG_IMU_EKF_H
#include "eskf.h"
#include "sensor_config.h"
#include <iostream>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

class img_imu_ekf
{
public:
    //state vector[p, v, q, ba, bw]  ---------------15dim????
    //state vector[q, pos, v, img, b_gyr, b_acc]  --------------17dim
    Quaterniond q;
    Vector3d pos;
    Vector3d vel;
    Vector2d img;
    Vector3d acc_bias;
    Vector3d gyro_bias;
    Vector3d w;
    Vector3d acc;

    const int dim = 18;
    double imu_freq = 50;
    double img_freq = 100;
    double img_f = 360; //相机焦距
    double timestamp;

    const Vector3d gravity = Vector3d(0, 0, -9.8);
    const Vector3d magnetic = Vector3d(1, 0, 0);
    const double gyro_bias_noise = GYRO_BIAS_WALK;
    const double gyro_noise = GYRO_NOISE_DENSITY;
    const double acc_noise = ACC_BIAS_WALK;
    const double acc_bias_noise = ACC_RANDOM_WALK;
    const double mag_noise = MAG_NOISE_DENSITY;
    const double gps_pos_noise = GPS_POS_NOISE_DENSITY;
    const double gps_vel_noise = GPS_VEL_NOISE_DENSITY;
    const double img_bias_noise = IMG_RANDOM_WALK;
    const double img_noise = IMG_NOISE_DENSITY;

    MatrixXd Q;
    MatrixXd R;
    MatrixXd G;
    MatrixXd Phi;
    // MatrixXd Phi_k1;
    // MatrixXd Phi_k2;
    // MatrixXd Phi_k3;

    eskf kf;
    bool is_atti_init_done;
    bool is_init_done;
    bool is_img_init_done;

    Vector3d delta_pos;
    Vector3d delta_vel;
    VectorXd delta_img;
    Vector3d delta_theta_acc;
    Vector3d delta_theta_mag;

    img_imu_ekf();
    ~img_imu_ekf();

    void sensor_init(Vector4d q, Vector3d pos, Vector3d vel, Vector2d img);
    void img_init(Vector3d acc, Vector3d mag);

    // void predict(Vector3d acc, Vector3d omega_raw, double dt);
    void update_Phi(Vector3d w, Vector3d vel, Vector3d acc, double dt);
    void predict_img(Vector3d w, Vector3d vel, double dt);

    //无Img时进行迭代
    // void no_img(Vector3d Z, Vector3d vel, int D);

    //有Img时进行迭代,目前只进行两次迭代
    void Img_Itera(MatrixXd P21, MatrixXd Phi_2, MatrixXd Q2, MatrixXd G1,
                   MatrixXd R, MatrixXd Z1, MatrixXd Z2, double dt);

    void update_matrix(Vector3d w, Vector3d vel, double dt);

    void set_imu_freq(double freq)
    {
        this->imu_freq = freq;
    }
    void set_img_freq(double freq) { this->img_freq = freq; }
    void set_Q_matrix(double dt);
    void set_P_matrix();
    void update_state();
    void print_state();
    Vector3d get_atti_euler();
    void set_timestatmp(double t) { timestamp = t; }
};

#endif