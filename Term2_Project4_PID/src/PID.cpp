#include "PID.h"
#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	this->Kp = Kp;
	this->Ki = Ki;
	this->Kd = Kd;

	p_error = 0;
	i_error = 0;
	d_error = 0;

	tw=false; // already tuned to final parameter value {0.11, 0.011, 2.99324}
	dK={0.1*Kp,0.1*Ki,0.1*Kd};
    loop=0;
    index_param=2; //first round surely enter total_err<best_err, so param tuning from 2nd round
    loop_to_stable=50;
    loop_to_verify=50;
    total_err=0.0;
    best_err=std::numeric_limits<double>::max();
    flag_inc=false;
    flag_dec=false;
}

void PID::UpdateError(double cte) {
	d_error = cte - p_error;
	p_error = cte;
	i_error += cte;


	if (fabs(dK[0]+dK[1]+dK[2]) < 0.01){
		tw=false; //stop twiddle if sum of dK is less than a very small number
		std::cout<<"dKp, dKi, dKd: "<<dK[0]<<" "<<dK[1]<< " " <<dK[2]<<std::endl;
		std::cout<<"absolute sum of dK: "<<fabs(dK[0]+dK[1]+dK[2])<<std::endl;
		std::cout<<"twiddle completed!"<<std::endl;
	}
	std::cout<<"Kp, Ki, Kd: "<< Kp <<" "<<Ki<<" "<<Kd<<std::endl;
	std::cout<<"dK["<<index_param<<"]= "<<dK[index_param]<<std::endl;

}

void PID::Twiddle(double cte){
	loop+=1;
	std::cout <<"loop: " << loop%(loop_to_stable+loop_to_verify)<<std::endl;
	if (loop%(loop_to_stable+loop_to_verify)>loop_to_stable){
		total_err+=pow(cte,2);
	}
	if (tw==true && loop%(loop_to_stable+loop_to_verify)==0){
		if (total_err<best_err){
			std::cout<<"total error is: "<<total_err<<std::endl;
			best_err=total_err;

			if (loop!=loop_to_stable+loop_to_verify){//1st time don't enlarge dK;
				dK[index_param]*=1.1;
			}

			// continue to next K:
			index_param=(index_param+1)%3;
			flag_inc=false;
			flag_dec=false;

			std::cout<<"best_err improved. enlarge dK. continue to tune next K. "<<std::endl;
		}

		if (flag_inc==false && flag_dec==false){
			// increment K by dp:
			TuneParam(index_param, dK[index_param]);
			flag_inc=true;
		}
		else if (flag_inc==true && flag_dec==false){
			// decrement K by 2dp:
			TuneParam(index_param, -2*dK[index_param]);
			flag_dec=true;
		}
		else {
			// revert back to original K (after +dK and -2dK):
			TuneParam(index_param, dK[index_param]);
			dK[index_param]*=0.9;

			// continue to next param:
			index_param=(index_param+1)%3;
			flag_inc=false;
			flag_dec=false;

			std::cout<<"shrink dK. continue to tune next K. "<<std::endl;
		}
		total_err=0;
	}
}

void PID::TuneParam(double index, double change){
	if (index==0) Kp+=change;
	if (index==1) Ki+=change;
	if (index==2) Kd+=change;
}

double PID::TotalError() {
	return -Kp*p_error - Ki*i_error - Kd*d_error;
}

