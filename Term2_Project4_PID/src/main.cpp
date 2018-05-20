#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>
#include <vector>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main(int argc, char *argv[])
{
  uWS::Hub h;

  PID pid;
  // TODO: Initialize the pid variable.
  double init_Kp = atof(argv[1]);
  double init_Ki = atof(argv[2]);
  double init_Kd = atof(argv[3]);
  double d_Kp=1.0;
  double d_Ki=1.0;
  double d_Kd=1.0;
  double best_err;
  int count=0;
  int counter_param=0;
  int loop=1;
  double K;
  double prev_K;
  double d_K;
  double prev_d_K;
  int prev_count;
  double tol=0.01;
  double error=0;
  double sum=0;

  // At command line, put: ./pid.Init(0.11, 0.011, 2.99324)
  pid.Init(init_Kp,init_Ki,init_Kd);
  //pid.Init(0.1,0.01,3);
  h.onMessage([&pid, &d_Kp, &d_Ki, &d_Kd, &best_err, &count, &loop, &error, &sum, &counter_param, &K, &prev_K, &d_K, &prev_d_K, &prev_count, &tol](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data).substr(0, length));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<std::string>());
          double speed = std::stod(j[1]["speed"].get<std::string>());
          double angle = std::stod(j[1]["steering_angle"].get<std::string>());
          double steer_value;
          /*
          * TODO: Calcuate steering value here, remember the steering value is
          * [-1, 1].
          * NOTE: Feel free to play around with the throttle and speed. Maybe use
          * another PID controller to control the speed!
          */

//          if (d_Kp+d_Ki+d_Kd > tol){
//			  std::cout << "d_Kp+d_Ki+d_Kd= " << d_Kp+d_Ki+d_Kd << std::endl;
//			  if (counter_param%3==0){
//				  K=pid.Kp;
//				  d_K=d_Kp;
//			  }
//			  if (counter_param%3==1){
//				  K=pid.Ki;
//				  d_K=d_Ki;
//			  }
//			  if (counter_param%3==2){
//				  K=pid.Kd;
//				  d_K=d_Kd;
//			  }
//
//			  K+=d_K;
//
//			  if (count>0){
//				  if ((cte>=best_err) && (count==prev_count+1) && (abs(K-(prev_K-2*d_K+d_K))<0.0001)){ //since in above d_K was added
//					  // if (last loop enters else{})
//					  //K+=d_K; //not add again since the same was added in line above if (count>0)
//					  d_K*=0.9;
//					  std::cout<<"2nd level check else entered. " << "updated 0/1/2 K " << counter_param%3 << " " << K << std::endl;
//					  goto skip_else;
//				  }
//				  if (cte<best_err){
//					  if ((count==prev_count+1) && (abs(K-(prev_K-2*d_K+d_K))<0.0001)){ //since in above d_K was added
//						  // if (last loop enters else{})
//						  K-=d_K; //need to subtract the same that was added in line above if (count>0)
//						  std::cout<<"2nd level check cte<best_err entered. " << "updated 0/1/2 K " << counter_param%3 << " " << K << std::endl;
//					  }
//					  best_err=cte;
//					  prev_d_K=d_K;
//					  d_K*=1.1;
//					  std::cout<<"1st level check cte<best_err entered. " << "updated 0/1/2 K " << counter_param%3 << " " << K << std::endl;
//				  }else{
//					  prev_K=K;
//					  prev_count=count;
//					  K-=2*d_K;
//					  std::cout<<"1st level check else entered. " << "updated 0/1/2 K " << counter_param%3 << " " << K << std::endl;
//				  }
//skip_else: ;
//				  count+=1;
//			  }
//
//			  //initialize best_err
//			  if (count==0) {
//					  best_err=cte;
//					  count+=1;
//			  }
//
//			  if (counter_param%3==0){
//				  pid.Kp=K;
//				  d_Kp=d_K;
//			  }
//			  if (counter_param%3==1){
//				  pid.Ki=K;
//				  d_Ki=d_K;
//			  }
//			  if (counter_param%3==2){
//				  pid.Kd=K;
//				  d_Kd=d_K;
//			  }
//
//			  if ((count==prev_count+2) || (abs(d_K-1.1*prev_d_K)<0.0001)){
//					  // if (1st route: d_K*1.1; 2nd route: K-2*d_K followed by conditions: a. d_K*1.1; b. K+d_K & d_K*0.9
//					  counter_param+=1;
//					  std::cout<<"continue to tune next parameter. counter_param= " << counter_param << std::endl;
//			  }
//          }

          pid.UpdateError(cte);
        	  pid.Twiddle(cte);

          steer_value = pid.TotalError();

		  if (steer_value > 1.0) {
			  steer_value = 1.0;
		  } else if (steer_value < -1.0) {
			  steer_value = -1.0;
		  }
          // DEBUG
          std::cout << "CTE: " << cte << " Steering Value: " << steer_value << std::endl;

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = 0.3;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
