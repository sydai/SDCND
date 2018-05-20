#ifndef PID_H
#define PID_H
#include <vector>

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;

  /*
  * Coefficients
  */ 
  double Kp;
  double Ki;
  double Kd;

  // twiddle
  bool tw;
  int index_param;
  std::vector<double> dK;
  int loop, loop_to_stable,loop_to_verify;
  double total_err, best_err;
  bool flag_inc, flag_dec;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  void Twiddle(double cte);
  void TuneParam(double index, double change);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
};

#endif /* PID_H */
