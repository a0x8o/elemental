#pragma once
#ifndef EL_IMPORTS_MPI_COMM_HPP_
#define EL_IMPORTS_MPI_COMM_HPP_

#include <El/config.h>
#include <El/core/imports/mpi/plain_comm.hpp>
#ifdef HYDROGEN_HAVE_ALUMINUM
#include <El/core/imports/mpi/aluminum_comm.hpp>
#endif // HYDROGEN_HAVE_ALUMINUM

#include <type_traits>

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
using Comm = AluminumComm;
#else
using Comm = PlainComm;
#endif // HYDROGEN_HAVE_ALUMINUM

/** @brief Create a Comm that takes ownership of the given MPI_Comm
 *
 *  The input communicator is not duplicated and the resource
 *  management passes to the newly created Comm object.
 *
 *  @param comm The MPI_Comm object over which this will take control.
 *
 *  @return A new Comm object controlling the input MPI_Comm.
 */
inline Comm MakeControllingComm(MPI_Comm comm) EL_NO_RELEASE_EXCEPT
{
    Comm controlling_comm;
    controlling_comm.Control(comm);
    return controlling_comm;
}


}// namespace mpi
}// namespace El
#endif /* EL_IMPORTS_MPI_COMM_HPP_ */
