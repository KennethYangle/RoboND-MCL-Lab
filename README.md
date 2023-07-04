[![Rfly](https://rfly.buaa.edu.cn/img/logo-banner.af7a6ec6.png)](https://rfly.buaa.edu.cn/)

# Active Search
Multicopter active search and second attack.


## Compiling the Program
```sh
# g++ main.cpp -o app -std=c++11 -I/usr/include/python2.7 -lpython2.7
catkin_make
```

## Running the Program
```sh
roslaunch MCL img_ekf.launch
```

## Datas
All test data in the `dataset` folder.

## Parameters
Config files are in the `MCL/config` folder.

## Reference
[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

https://github.com/udacity/RoboND-MCL-Lab
