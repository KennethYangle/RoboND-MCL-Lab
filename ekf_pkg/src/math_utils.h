#ifndef __MATH_UTILS_H
#define __MATH_UTILS_H

#include <Eigen/Eigen>

using namespace Eigen;

Vector3d quat2euler(Quaterniond qq);
Quaterniond euler2quat(Vector3d euler);
Matrix3d skew_symmetric(Vector3d v);

#endif