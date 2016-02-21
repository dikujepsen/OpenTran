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

void CheckNull(unsigned * ptr, const char* val) {
      if (ptr == NULL) {
	std::cout << "Command line option saved to null pointer. Exiting...\n";
	exit(-1);
      } else {
	unsigned temp;
	stringstream toInt(val);
	if ( !(toInt >> temp) ) {
	  std::cout << "Command line option: invalid number: " << val << std::endl;
	  exit(-1);
	}	  
	*ptr = temp;
      }
}

void CheckType(std::string * ptr, const char* val) {
      if (ptr == NULL) {
	std::cout << "Command line option saved to null pointer. Exiting...\n";
	exit(-1);
      } else {
	std::string temp;
	stringstream toStr(val);
	if ( !(toStr >> temp) ) {
	  std::cout << "Command line option: invalid number: " << val << std::endl;
	  exit(-1);
	}	  
	*ptr = temp;
      }
}


void ParseCommandLine(int argc, char** argv,
		      unsigned * val1,
		      unsigned * val2,
		      unsigned * val3,
		      std::string * ocl_type) {
  int len = 1;
  if (val1 != NULL) {
    len += 2;
  }
  if (val2 != NULL) {
    len += 2;
  }
  if (val3 != NULL) {
    len += 2;
  }
  if (ocl_type != NULL) {
    len += 2;
  }
  
  if (((argc-1) % 2 != 0) || argc != len || argc < 3) {
    std::cout << "Please set -n <x> -m <y> -k <z>\n -t <w>";
    exit(-1);
  }
  
  for (int i = 1; i < argc; i+=2) {
    if (std::string(argv[i]) == "-n") {
      CheckNull(val1, argv[i + 1]);
    } else if (std::string(argv[i]) == "-m") {
      CheckNull(val2, argv[i + 1]);
    } else if (std::string(argv[i]) == "-k") {
      CheckNull(val3, argv[i + 1]);
    } else if (std::string(argv[i]) == "-t") {
      CheckType(ocl_type, argv[i + 1]);
    } else {
      std::cout << "Unrecognized command line option\n";
      exit(-1);
    }      
  }
}
