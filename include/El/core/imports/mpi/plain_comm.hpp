#pragma once
#ifndef EL_IMPORTS_MPI_PLAIN_COMM_HPP_
#define EL_IMPORTS_MPI_PLAIN_COMM_HPP_

#include <El/core/imports/mpi/comm_impl.hpp>

namespace El
{
namespace mpi
{

/** @class PlainComm
 *  @brief Plain ol' MPI-based communicator.
 */
class PlainComm : public CommImpl<PlainComm>
{
public:
    PlainComm() EL_NO_EXCEPT = default;
    PlainComm(MPI_Comm comm) EL_NO_EXCEPT
        : CommImpl<PlainComm>{comm}
    {}

    void DoReset() const EL_NO_EXCEPT {};
};// class PlainComm

}// namespace mpi
}// namespace El
#endif /* EL_IMPORTS_MPI_PLAIN_COMM_HPP */
