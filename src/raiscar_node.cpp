#include "ros/ros.h"
#include "std_msgs/UInt16.h"
#include "raiscar/MotorController.h"
#include <sensor_msgs/Joy.h>
#include <sstream>
class SubscribeAndPublish
{
public:
  SubscribeAndPublish()
  {
    pub_ = n_.advertise<raiscar::MotorController>("controls",1000);
    sub_ = n_.subscribe("joy",1000, &SubscribeAndPublish::controllerCallback, this);
  }

  //---------------------------------------------
  float map_acceleration(float acc){
  // takes input between [-1,1]
  // maps to [0,1]
  return 1-((1-acc)/2);
  }

  //---------------------------------------------
  float map_steering(float st_left, float st_right)
  {
  // output: steering value between 0 and 1
  // output: 0->steering left
  // output: 1->steering right
  // input: left and right from controller within range [-1;1] each
  
  // mapping to [0,0.5]
  float right_mapped = (1-st_right)/4;
  float left_mapped = (1-st_left)/4;
  float steering = (1+st_right)/2;
  return steering;
  }

  //---------------------------------------------
  void controllerCallback(const sensor_msgs::Joy& msg)
  {
  //ROS_INFO("I heard: [%f]", msg.axes[3]);
  //ROS_INFO("I heard: [%f]", msg.axes[0]);
  //ROS_INFO("I heard: [%f]", msg.axes[2]);
  //msg.axes[3];//forward or backward
  //msg.axes[0];//steering right
  //msg.axes[2];//steering left

  _angle = map_steering(msg.axes[2],msg.axes[0]);
  _speed = map_acceleration(msg.axes[3]);
  raiscar::MotorController control_msg;
  control_msg.speed = _speed;
  control_msg.angle = _angle;
  ROS_INFO("I heard: %f]", _angle);

  pub_.publish(control_msg);
  }



private:
  ros::NodeHandle n_; 
  ros::Publisher pub_;
  ros::Subscriber sub_;
  float _angle = 0;
  float _speed = 0;

};//End of class SubscribeAndPublish
  


int main(int argc, char **argv)
{
  ros::init(argc, argv, "raiscar_node");

  //ros::Publisher chatter_pub = n.advertise<std_msgs::UInt16>("servo_steering", 1000);
  SubscribeAndPublish my_obj;

  ros::spin();
  //ros::Rate loop_rate(1);
  return 0;
}
