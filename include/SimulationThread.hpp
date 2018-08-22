#pragma once

#include <mutex>
#include <iostream>

#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <osg/Vec3i>
#include <osg/ref_ptr>

#include "DomainData.hpp"

// class MyThreadClass
// {
// public:
//   MyThreadClass()
//   { /* empty */
//   }
//   virtual ~MyThreadClass()
//   { /* empty */
//   }

//   /** Returns true if the thread was successfully started, false if there was an error starting the thread */
//   bool StartInternalThread()
//   {
//     return (pthread_create(&_thread, NULL, InternalThreadEntryFunc, this) == 0);
//   }

//   /** Will not return until the internal thread has exited. */
//   void WaitForInternalThreadToExit()
//   {
//     (void)pthread_join(_thread, NULL);
//   }

// protected:
//   /** Implement this method in your subclass with the code you want your thread to run. */
//   virtual void InternalThreadEntry() = 0;

// private:
//   static void *InternalThreadEntryFunc(void *This)
//   {
//     ((MyThreadClass *)This)->InternalThreadEntry();
//     return NULL;
//   }
//   pthread_t _thread;
// };

class SimulationThread : public OpenThreads::Thread
{
private:
  // Quantity to be visualised on plot
  DisplayQuantity::Enum m_visQ;
  // Triple mutex for prioritized access
  std::mutex m_l, m_m, m_n;
  // Buffer for OpenGL plot, copied when drawing is requested
  thrust::device_vector<real> m_plot;
  // Counts the number of simulation updates
  unsigned int m_time;
  // Signals exit of simulation loop
  volatile bool m_exit;
  // Signals simulation pause
  volatile bool m_paused;

  DomainData *m_domainData;
  osg::ref_ptr<VoxelMesh> m_mesh;

public:
  SimulationThread();

  osg::ref_ptr<VoxelMesh> getVoxelMesh();

  // Upload new boundary conditions
  void uploadBCs();

  // Reset the averaging array
  void resetAverages();

  // Reset the simulation
  void resetDfs();

  void pause(bool state);
  bool isPaused();
  void draw(real *plot, DisplayQuantity::Enum visQ);

  virtual void run();

  virtual int cancel();
};
