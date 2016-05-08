#include <time.h>
#include <sstream>
#include <iostream>
class Stopwatch {
private:
  struct timespec begin;

  long long time_diff( timespec start, timespec end){
    return ((end.tv_sec - start.tv_sec) * 1000000000LL + 
	    (end.tv_nsec - start.tv_nsec));
  }

public:
  Stopwatch() {}
  void start() {
    clock_gettime(CLOCK_REALTIME, &begin);
  }

  double stop() {
    timespec end;
    clock_gettime(CLOCK_REALTIME, &end);
    double difft = (double)time_diff(begin ,end);
    double ret = (difft/1000000000.0);
    return ret;
  }
};

