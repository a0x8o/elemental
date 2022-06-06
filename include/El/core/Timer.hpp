/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_TIMER_HPP
#define EL_TIMER_HPP

#include <chrono>

namespace El {

using std::chrono::duration;
using std::chrono::duration_cast;

using Clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
                               std::chrono::high_resolution_clock,
                               std::chrono::steady_clock>::type;

class Timer
{
public:
    Timer( const string& name="[blank]" );

    const string& Name() const;

    void Start();
    double Stop();
    double Partial() const; // time since last start
    double Total() const; // total elapsed time

    void Reset( const string& name="[blank]" );
private:
    bool running_ = false;
    string name_ = "[blank]";
    double totalTime_=0, lastPartialTime_=0;
    Clock::time_point lastTime_;
};

} // namespace El

#endif // ifndef EL_TIMER_HPP
