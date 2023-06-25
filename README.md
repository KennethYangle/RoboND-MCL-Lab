[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# RoboND-MCL-Lab
You will be able to observe the `MCL` in action through the generated images. 

### Editing the Program
Enter the code in the designated section:
```C++
//####   DON'T MODIFY ANYTHING ABOVE HERE! ENTER CODE BELOW ####
		
//TODO: Graph the position of the robot and the particles at each step 
```

### Compiling the Program
```sh
# g++ main.cpp -o app -std=c++11 -I/usr/include/python2.7 -lpython2.7
catkin_make
```


### Running the Program
Before you run the program, make sure the `Images` folder is empty!
```sh
rosrun MCL mcl_node
```
Wait for the program to iterate `50` times.

### Generated Images
After running the program, `50` images will be generated in the `Images` folder.

### Datas
All test data in the `dataset` folder.