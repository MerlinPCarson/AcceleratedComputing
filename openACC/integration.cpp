/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp integration_seq.cpp -o integration_seq
 ============================================================================
 */
// Sequential integration of testf
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <chrono>

using namespace std;

//---------------------------------------
#pragma acc routine seq
double testf(double x);

inline double testf (double x)
{
  return x * x + 2 * sin (x);
}

//---------------------------------------
double integrateCPU (double st, double en, int div, double (*f) (double))
{
  double localRes = 0;
  double step = (en - st) / div;
  double x;
  x = st;
  localRes = f (st) + f (en);
  localRes /= 2;
  for (int i = 1; i < div; i++)
    {
      x += step;
      localRes += f (x);
    }
  localRes *= step;

  return localRes;
}

//---------------------------------------
double integrateOpenACC (double st, double en, int div, double (*f) (double))
{
  double localRes = 0;
  double step = (en - st) / div;
  double x;
  x = st;
  localRes = f (st) + f (en);
  localRes /= 2;
#pragma acc data copyin(step, st) create(x) copy(localRes)
{
  #pragma acc parallel loop reduction(+:localRes)
  for (int i = 1; i < div; i++)
    {
      x = i * step + st;
      localRes += testf(x);
    }
}
  localRes *= step;

  return localRes;
}
//---------------------------------------
int main (int argc, char *argv[])
{

  if (argc == 1)
    {
      cerr << "Usage " << argv[0] << " start end divisions\n";
      exit (1);
    }
  
  std::chrono::duration<double> elapsedTime;;
  auto start_t = std::chrono::steady_clock::now();

  double start, end, finalRes;
  int divisions;
  start = atof (argv[1]);
  end = atof (argv[2]);
  divisions = atoi (argv[3]);

  start_t = std::chrono::steady_clock::now();
  finalRes = integrateCPU (start, end, divisions, testf);

  elapsedTime = (chrono::steady_clock::now() - start_t);

  cout << endl << "Answer: " << finalRes << endl;
  cout << "Execution time on CPU: " <<  elapsedTime.count() << " seconds\n" << endl;

  start_t = std::chrono::steady_clock::now();
  finalRes = integrateOpenACC (start, end, divisions, testf);

  elapsedTime = (chrono::steady_clock::now() - start_t);

  cout << endl << "Answer: " << finalRes << endl;
  cout << "Execution time using OpenACC: " <<  elapsedTime.count() << " seconds\n" << endl;

  return 0;
}
