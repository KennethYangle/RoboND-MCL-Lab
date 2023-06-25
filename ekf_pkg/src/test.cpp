#include <iostream>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

int main()
{
    // VectorXd Q[3];// = VectorXd::Ones(18);
    // Q[0] = VectorXd::Ones(18);
    // Q[1] = VectorXd::Ones(18);
    // Q[2] = VectorXd::Ones(18);
    // Vector4d a1(1,2,3,4);
    // Vector3d a2(5,6,7);
    // Vector3d a3(8, 9, 10);
    // Vector2d a4(11, 12);
    // Vector3d a5(0, 0, 0);
    // Vector3d a6(1, 1, 1);
    // Q[0].segment(0, 4) = a1;
    // Q[0].segment(4, 3) = a2;
    // Q[0].segment(7, 3) = a3;
    // Q[0].segment(10, 2) = a4;
    // Q[0].segment(12, 3) = a5;
    // Q[0].segment(15, 3) = a6;
    // // Q[0] = Vector3d(1,2,3);
    // cout << "Q0:" << Q[0] << endl;

    // // Q[1] = Vector3d(3, 4, 5);
    // // Q[1] = 2*Q[1];
    // Q[1].segment(0, 4) = Vector4d::Ones();
    // Q[2].segment(4, 3) = Vector3d::Ones() * 4.4;
    // Q[1].segment(7, 3) = Vector3d::Ones() * 5.5;
    // Q[1].segment(10, 2) = Vector2d::Ones() * 6.6;
    // Q[1].segment(12, 3) = Vector3d::Ones() * 0.02 * 0.02;
    // Q[1].segment(15, 3) = Vector3d::Ones() * 0.02 * 0.02;
    // cout << "Q1:" << Q[1] << endl;

    // // Q[2] = Vector3d(6, 7, 3);
    // Q[2].segment(0, 4) = Vector4d::Ones();
    // Q[2].segment(4, 3) = Vector3d::Ones() * 7.7;
    // Q[2].segment(7, 3) = Vector3d::Ones() * 8.8;
    // Q[2].segment(10, 2) = Vector2d::Ones() * 9.9;
    // Q[2].segment(12, 3) = Vector3d::Ones() * 0.03 * 0.03;
    // Q[2].segment(15, 3) = Vector3d::Ones() * 0.03 * 0.03;
    // // Q[2] = 5*Q[2];
    // cout << "Q2:" << Q[2] << endl;
    // Vector4d q(1,2,3,4);
    // Vector3d pos(5,6,7);
    // Vector3d vel(5, 6, 7);
    // Vector2d img(5, 6);
    // VectorXd x;
    // x = VectorXd::Ones(18);
    // x.segment(0, 4) = q;
    // x.segment(4, 3) = pos;
    // x.segment(7, 3) = vel;
    // x.segment(10, 2) = img;
    // x.segment(12, 3) = Vector3d::Ones() * 0.01 * 0.01;
    // x.segment(15, 3) = Vector3d::Ones() * 0.01 * 0.01;
    // cout << "XX:" << x << endl;
    MatrixXd Q;
    // Q = MatrixXd::Identity(18, 18);
    Q = MatrixXd::Zero(3, 3);
    Q << 0,0,1,
        0,1,0,
        1,0,0;
    // Matrix2d I2 = Matrix2d::Identity(); //定义单位矩阵
    // Matrix3d I3 = Matrix3d::Identity(); //定义单位矩阵
    // Matrix4d I4 = Matrix4d::Identity(); //定义单位矩阵
    //状态的顺序为：q,p,v,img,b_gyr,b_acc
    //其中，四元数用欧拉角表示
    //目前这个是随意设置的，不一定正确,四元数直接从mavros中读取
    // Q.block(0, 0, 4, 4) = I4;             //q
    // Q.block(4, 4, 3, 3) = I3;       //pos
    // Q.block(7, 7, 3, 3) = I3;               //v
    // Q.block(10, 10, 2, 2) = I2;             //img
    // Q.block(12, 12, 3, 3) = I3; //b_gyr
    // Q.block(15, 15, 3, 3) = I3;   //b_acc
    Q = 5 * Q;
    cout << "Q:" << Q << endl;
    Q = Q.reverse();
    cout << "Q____:" << Q << endl;

    return 0;
}
