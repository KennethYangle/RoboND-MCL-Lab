#ifndef __ESKF_H
#define __ESKF_H
#include <iostream>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

class eskf
{

public:
    int dim;
    VectorXd dx;
    VectorXd x;
    MatrixXd P;
    VectorXd z;
    MatrixXd H;

    eskf();
    eskf(int dim);
    ~eskf();
    void eskf_init_P(VectorXd p);
    //状态预测:Phi:k-1时刻到k时刻的状态转移矩阵，X：k-1时刻到k-1时刻的状态估计更新
    //P:k-1时刻到k-1时刻误差协方差状态更新 G和Q是噪声方差矩阵
    void predict(MatrixXd Phi, MatrixXd X, MatrixXd P, MatrixXd G, MatrixXd Q);

    void update(VectorXd v, MatrixXd H, MatrixXd R);
    //状态更新：P：k-1时刻到k时刻误差协方差阵预测，H是k时刻观测方程，R是k时刻观测方差阵
    //X：k-1时刻到k时刻状态估计预测，K是k时刻卡尔曼增益阵，Phi是k-1时刻状态转移矩阵
    //Z：k-1时刻到k时刻状态观测值
    void update(MatrixXd P, MatrixXd H, MatrixXd R, MatrixXd X, MatrixXd K, VectorXd Z);

    MatrixXd get_Covariance() { return P; };     //获取误差协方差矩阵 赋值方式：this -> P = P;
    MatrixXd get_State_Estimate() { return x; }; //获取状态估计
    // MatrixXd get_Trans_Matrix() { return Phi; }; //获取状态转移矩阵Phi
    VectorXd get_state() { return dx; };
};

#endif