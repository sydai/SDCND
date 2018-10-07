# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program

# Reflection:

### The WebSocket requires way points to be sent in the format of vectors next_x_vals and next_y_vals to navigate throught the highway track.

Part of the code takes reference from project Q & A video. The key points are:

1. Construct evenly spaced way points of fitted spline (lane keep) by dividing the spline using ref_vel so that point #4 is satisfied
2. Keep lane and drive at the fastest speed but not violating the speed limit and avoid collision
3. When behind a slower moving car, take action to change lane when the adjacent lane is clear
4. Constrain the acceleration and jerk to a certain threshold for the comfort of passengers
5. Keep smooth transition from "previous_path_x(y)" to current loop newly generated way points

#### 1. Construct evenly spaced way points of fitted spline by dividing the spline using ref_vel

* Prepare 5 points for spline fitting:
   * First 2 points: 
      * case 1 cold start: use {car_x, car_y} and previous position calculated from car_x, car_y, car_yaw
      * case 2 previous path exist: use last two elements from previous_path_x(y)
   * Next 3 points:
      * calculate from getXY() given car_s + 30m / 60m / 90m
      
```
ln384:   double ref_x = car_x;
         double ref_y = car_y;
         double ref_yaw = deg2rad(car_yaw);
ln388:   // if previous size is almost empty, use car current position as starting reference
         if (prev_size < 2) {
           // use two points that make the path tangent to the car
           double prev_car_x = car_x - cos(car_yaw);
           double prev_car_y = car_y - sin(car_yaw);

           ptsx.push_back(prev_car_x);
           ptsy.push_back(prev_car_y);

           ptsx.push_back(car_x);
           ptsy.push_back(car_y);
         }
ln400:   // use previous path end point as starting reference
         else {
           // redefine reference state as previous path end point
           ref_x = previous_path_x[prev_size-1];
           ref_y = previous_path_y[prev_size-1];

           double ref_x_prev = previous_path_x[prev_size-2];
           double ref_y_prev = previous_path_y[prev_size-2];

           ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

           // use two points that make the path tangent to the previous path's end point
           ptsx.push_back(ref_x_prev);
           ptsy.push_back(ref_y_prev);

           ptsx.push_back(ref_x);
           ptsy.push_back(ref_y);

         } // 2 points in ptsx, ptsy
         
ln420:   // In Frenet add evenly 30m spaced points ahead of the starting reference, using "lane" variable here
         vector<double> next_wp0 = getXY(car_s + 30, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
         vector<double> next_wp1 = getXY(car_s + 60, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
         vector<double> next_wp2 = getXY(car_s + 90, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

         ptsx.push_back(next_wp0[0]);
         ptsy.push_back(next_wp0[1]);

         ptsx.push_back(next_wp1[0]);
         ptsy.push_back(next_wp1[1]);

         ptsx.push_back(next_wp2[0]);
         ptsy.push_back(next_wp2[1]); // 5 points in ptsx, ptsy
```

* Transform the 5 points w.r.t the ego car frame given car position {ref_x, ref_y} and ref_yaw to avoid vertical case of spline fitting (code skipped)

* Spline fitting:
```
ln445:   // create a spline
         tk::spline s;

         // set (x, y) points to the spline
         s.set_points(ptsx, ptsy);
```
* Divide the target distance of the spline into evenly space segments by using ref_vel then project into the x position and get y position from spline function; Convert back to global frame; Append to previous_path_x(y) until 50 points are reached:
```
ln468:   for (int i = 0; i <= 50-previous_path_x.size(); i++) {
           // divide dist into N segments
           double N = (target_dist/(.02*ref_vel/2.24)); // 2.24 is to convert vel from mph to m/s
           double x_point = x_add_on + (target_x)/N;
           double y_point = s(x_point);

            x_add_on = x_point;

            double x_ref = x_point;
            double y_ref = y_point;

            // transform back to global frame after transforming it earlier to local frame
            x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
            y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }
```
#### 2. Keep lane and drive at the fastest speed but not violating the speed limit (50mph)
```
ln252:   double ref_vel = 0.0; 
ln362:   else if (ref_vel < 49.5) {
            //gradually acc from cold start corresponding init ref_vel = 0.0;
            ref_vel += .224; // .224/1.6093/0.02 = 7 m/s^2 < 10 m/s^2 (acc limit)
         }
```
#### 3. When behind a slower moving car, take action to change lane when the adjacent lane is clear

* Loop all traffic cars from "sensor_fusion" variable; Check if there's car close in front to avoid collision by mark flag "too_close", then check if front car is too slow for ego car to make a lane change using flags "change_left" & "change_right":
```
ln314:   for (int i=0; i < sensor_fusion.size(); i++){
              
           float d = sensor_fusion[i][6];
           if (d < (2+4*lane+2) && d > (2+4*lane-2)) { //traffic car is in ego car lane
             double vx = sensor_fusion[i][3];
             double vy = sensor_fusion[i][4];
             double check_speed = sqrt(vx*vx + vy*vy);
             double check_car_s = sensor_fusion[i][5];

             check_car_s += ((double)prev_size * .02 * check_speed); // if using prev points can project s value out
             double dist_car_front = check_car_s - car_s;

             // when about to hit a suddenly cut in car
             if ((check_car_s > car_s) && (dist_car_front < 7)) {
               emergency_stop = true;
               follow_speed = check_speed;
             }
             // when following a slower moving car 
             if ((check_car_s > car_s) && (dist_car_front < 30)) { // check s value greater than ego car and s gap > 30
               // do some logic here, lower ref_vel so as to not crash into front car;
               // or do flag to try to change lanes
               //ref_vel = 29.5; // mph
               too_close = true;

               // implement lane change rule:
               if (lane==0){ // could change to the right lane
                 change_right = lane_change (sensor_fusion, prev_size, car_s, dist_car_front, lane, 1);
               } 
               else if (lane==1){ // could change to the left or right lane
                 change_left = lane_change (sensor_fusion, prev_size, car_s, dist_car_front, lane, -1);
                 change_right = lane_change (sensor_fusion, prev_size, car_s, dist_car_front, lane, 1);
               }
               else if (lane==2){ // could change to the left lane 
                 change_left = lane_change (sensor_fusion, prev_size, car_s, dist_car_front, lane, -1);
               }
             }
           }
         }
```
* Define lane change rule. Check adjacent lane for front car and rear car: if adjacent lane front car is further away than current lane front car && adjacent lane rear car has a distance of >15 meters away from ego car:
```
ln167:   bool lane_change (const vector<vector<double>> &sensor_fusion, int prev_size, double car_s, double dist_car_front,            int lane, int lr) { 
           // lr = -1: left; lr = 1: right

           double dist_car_adjacent_lane_front;
           double dist_car_adjacent_lane_rear;
           double min_front_dist=10000;
           double min_rear_dist=10000;
           for (int i=0; i < sensor_fusion.size(); i++){

             float d = sensor_fusion[i][6];
             if (d < (2+4*(lane+lr)+2) && d > (2+4*(lane+lr)-2)) { // check if left/right lane has car nearby
               double vx = sensor_fusion[i][3];
               double vy = sensor_fusion[i][4];
               double check_speed = sqrt(vx*vx + vy*vy);
               double check_car_s = sensor_fusion[i][5];
               check_car_s += ((double)prev_size * .02 * check_speed);

               if ((check_car_s > car_s)) {
                   dist_car_adjacent_lane_front = check_car_s - car_s;
                 if (dist_car_adjacent_lane_front < min_front_dist){
                   min_front_dist = dist_car_adjacent_lane_front;
                 }
                 dist_car_adjacent_lane_front = min_front_dist;
               }

               if ((check_car_s < car_s)) {
                   dist_car_adjacent_lane_rear = car_s - check_car_s;
                 if (dist_car_adjacent_lane_rear < min_rear_dist){
                   min_rear_dist = dist_car_adjacent_lane_rear;
                 }
                 dist_car_adjacent_lane_rear = min_rear_dist;
               }
             }
           }
           // when an adjacent lane is clear of other traffic
           if ((dist_car_adjacent_lane_front > dist_car_front) && (dist_car_adjacent_lane_rear > 15)) {
             return true;
           }
           else {
             return false;
           }
         }
```

* Define the action for flags "too_close", "change_left" and "change_right":
```
ln358:   // when too close, decrease with an acc 7m/s^2; when enough front space, increase speed with same acc until reach              speed limit
         if (too_close) {
           ref_vel -= .224; // in mph; .224/1.6093/0.02 = 7 m/s^2 < 10 m/s^2 acc limit
         }
ln366:   // when change_left flag is true no matter change_right flag value, treat change_left high priority
         if (change_left) {
           lane-=1;
         }
         if (change_right) {
           if (!change_left) { // when change_right is the only one flag that is true
             lane+=1;
           }
         }
```
#### 4. Constrain the acceleration and jerk to a certain threshold for the comfort of passengers
Refer to point #1 step #4

#### 5. Keep smooth transition from "previous_path_x(y)" to current loop newly generated way points
Refer to point #1 step #4


### Future improvement:

* Lane change algorithm could use Finite State Machine ("KL", "LCL", "LCR") selection based on lowest cost given the  predictions of traffic cars (for a certain horizon)


### Simulator.
You can download the Term3 Simulator which contains the Path Planning Project from the [releases tab (https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).

### Goals
In this project your goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. You will be provided the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

#### The map of the highway is in data/highway_map.txt
Each waypoint in the list contains  [x,y,s,dx,dy] values. x and y are the waypoint's map coordinate position, the s value is the distance along the road to get to that waypoint in meters, the dx and dy values define the unit normal vector pointing outward of the highway loop.

The highway's waypoints loop around so the frenet s value, distance along the road, goes from 0 to 6945.554.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

Here is the data provided from the Simulator to the C++ Program

#### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

#### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along
the path has processed since last time. 

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

#### Previous path's end s and d values 

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

#### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

## Details

1. The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.

## Tips

A really helpful resource for doing this project and creating smooth trajectories was using http://kluge.in-chemnitz.de/opensource/spline/, the spline function is in a single hearder file is really easy to use.

---

## Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!


## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

