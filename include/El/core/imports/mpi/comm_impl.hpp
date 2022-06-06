#pragma once
#ifndef EL_IMPORTS_MPI_COMM_IMPL_HPP_
#define EL_IMPORTS_MPI_COMM_IMPL_HPP_

#include <El/config.h>
#include <El/core/imports/mpi/error.hpp>

#include <mpi.h>

#include <exception>

namespace El
{
namespace mpi
{

/** @class CommImpl
 *  @brief A resource-owning Communicator wrapper.
 *
 *  This imposes strong "resource ownership" semantics on an MPI_Comm
 *  object. The underlying MPI_Comm is externally immutable (except by
 *  means hereby defined as pathological) and is not shared with other
 *  objects.
 *
 *  Any underlying resources are freed when this object is destroyed.
 *
 *  This class implements the CRTP for static polymorphism. The goal
 *  is encapsulation of implementation details of orthogonal
 *  communication backends that are "naturally MPI-like" (i.e.,
 *  Aluminum).
 *
 *  @tparam SpecificCommImpl A derived class name.
 */
template <typename SpecificCommImpl>
class CommImpl
{
public:

    /** @name Constructors */
    ///@{

    /** @brief Initialize a new communicator.
     *
     *  Initializes to MPI_COMM_NULL
     */
    CommImpl() = default;

    /** @brief Initialize a new communicator from an MPI_Comm
     *
     *  This duplicates the MPI_Comm object so the resource is "clean".
     *
     *  @param The MPI communicator handle with the same group as the
     *         new communicator.
     */
    CommImpl(MPI_Comm mpi_comm);

    /** @brief Move-construct from an other communicator.
     *
     *  The source is left in a default-constructed-like state.
     */
    CommImpl(CommImpl<SpecificCommImpl>&&) EL_NO_EXCEPT;

    /** @brief Move-assign from an other communicator.
     *
     *  The source is left in a default-constructed-like state.
     */
    CommImpl<SpecificCommImpl>& operator=(CommImpl<SpecificCommImpl>&&);

    ///@}
    /** @name Deleted functions */
    ///@{

    CommImpl(CommImpl<SpecificCommImpl> const&) = delete;
    CommImpl<SpecificCommImpl>& operator=(
        CommImpl<SpecificCommImpl> const&) = delete;

    ///@}
    /** @name Query metadata */
    ///@{

    /** @brief Get the process rank in the communicator. */
    int Rank() const;

    /** @brief Get the number of processes in the communicator. */
    int Size() const;

    ///@}
    /** @name Resource access */
    ///@{

    /** @brief Get the raw MPI handle.
     *
     *  Users must not call any MPI function that modifies the MPI
     *  resource, e.g., @c MPI_Comm_free.
     *
     *  @return A "reference" to the internal MPI_Comm.
     */
    MPI_Comm GetMPIComm() const EL_NO_EXCEPT { return comm_; };

    ///@}
    /** @name Modifiers */
    ///@{

    /** @brief Take ownership of an existing communicator.
     *
     *  This function behaves similarly to @c Reset(MPI_Comm). Any
     *  existing resources are freed and this class manages the given
     *  @c MPI_Comm object.
     *
     *  @param comm The MPI_Comm to be managed.
     */
    void Control(MPI_Comm comm);

    /** @brief Clear all the state of the communicator.
     *
     *  Internal resource handles are all freed and @c this is left in
     *  a default-constructed-like state.
     */
    void Reset();

    /** @brief Clear current state and recreate based on new comm.
     *
     *  The currently-held resources, if any, are freed and replaced
     *  with a duplicate of @c comm. See @c Control for a version of
     *  this function that assumes ownership of an @ c MPI_Comm object
     *  instead of duplicating the input @c MPI_Comm.
     *
     *  @param comm The MPI communicator to duplicate.
     */
    void Reset(MPI_Comm comm);

    /** @brief Relinquish ownership of the underlying resource.
     *
     *  After this call, @c this is left in default-constructed-like
     *  state.
     *
     *  @return The underlying MPI_Comm object, which is no longer
     *          managed by this class.
     */
    MPI_Comm Release();

    /** @brief Swap the interal state with another.
     *
     *  @param other The source with which to swap internals.
     */
    void Swap(CommImpl<SpecificCommImpl>& other) EL_NO_EXCEPT;

    ///@}

protected:

    /** @brief Delete the object; frees all underlying resources. */
    ~CommImpl() EL_NO_EXCEPT;

private:

    /** @brief Handle MPI_Comm duplication in an error-safe way.
     *
     *  @param src_comm The source @c MPI_Comm, i.e., the one that
     *         will be duplicated.
     */
    void SafeDuplicateMPIComm_(MPI_Comm src_comm);

    /** @brief Handle freeing the internal MPI_Comm object.
     *
     *  After this call, the internal @c comm_ will be MPI_COMM_NULL
     *  and any resources that had been previously managed will have
     *  been freed.
    */
    void FreeAndResetInternalComm_();

    /** @brief The local component of Reset().
     *
     *  This is for correct behavior of a CRTP interface.
     */
    void LocalReset_();

private:

    /** @brief The raw MPI handle. */
    MPI_Comm comm_ = MPI_COMM_NULL;

};// class CommImpl


// Public Interface implementation


template <typename SpecificCommImpl>
CommImpl<SpecificCommImpl>::CommImpl(MPI_Comm mpi_comm)
{
    // This just sets the data that this manages, which is only the
    // internal MPI_Comm handle.
    SafeDuplicateMPIComm_(mpi_comm);
}


template <typename SpecificCommImpl>
CommImpl<SpecificCommImpl>::CommImpl(
    CommImpl<SpecificCommImpl>&& other) EL_NO_EXCEPT
{
    // This just needs to handle its own data; it should not touch
    // derived type data at all.
    comm_ = other.comm_;
    other.comm_ = MPI_COMM_NULL;
}


template <typename SpecificCommImpl>
CommImpl<SpecificCommImpl>&
CommImpl<SpecificCommImpl>::operator=(
    CommImpl<SpecificCommImpl>&& other)
{
    // Delete the state for which this is responsible
    FreeAndResetInternalComm_();

    // Take the new MPI_Comm and reset other to null
    comm_ = other.comm_;
    other.comm_ = MPI_COMM_NULL;
}


template <typename SpecificCommImpl>
CommImpl<SpecificCommImpl>::~CommImpl() EL_NO_EXCEPT
{
    try
    {
        FreeAndResetInternalComm_();
    }
    catch (std::exception const& e)
    {
        std::cerr << "Detected MPI error in ~CommImpl.\n"
                  << "Exceptions detected in destructors are considered fatal.\n"
                  << "The caught exception is:\n\n"
                  << "   e.what(): " << e.what()
                  << "Now calling \"std::terminate()\"."
                  << std::endl;
        std::terminate();
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception in ~CommImpl.\n"
                  << "Exceptions detected in destructors are considered fatal.\n"
                  << "Now calling \"std::terminate()\"."
                  << std::endl;
        std::terminate();
    }
}


template <typename SpecificCommImpl>
int CommImpl<SpecificCommImpl>::Rank() const
{
    if (comm_ == MPI_COMM_NULL)
        return MPI_UNDEFINED;

    int rank;
    EL_CHECK_MPI_CALL(MPI_Comm_rank(comm_, &rank));
    return rank;
}


template <typename SpecificCommImpl>
int CommImpl<SpecificCommImpl>::Size() const
{
    if (comm_ == MPI_COMM_NULL)
        return MPI_UNDEFINED;

    int size;
    EL_CHECK_MPI_CALL(MPI_Comm_size(comm_, &size));
    return size;
}


template <typename SpecificCommImpl>
void CommImpl<SpecificCommImpl>::Control(MPI_Comm comm)
{
    Reset();
    comm_ = comm;
}


template <typename SpecificCommImpl>
void CommImpl<SpecificCommImpl>::Reset()
{
    static_cast<SpecificCommImpl*>(this)->DoReset();
    LocalReset_();
}


template <typename SpecificCommImpl>
void CommImpl<SpecificCommImpl>::Reset(MPI_Comm comm)
{
    Reset();
    SafeDuplicateMPIComm_(comm);
}


template <typename SpecificCommImpl>
MPI_Comm CommImpl<SpecificCommImpl>::Release()
{
    MPI_Comm ret = comm_;
    comm_ = MPI_COMM_NULL;
    Reset();
    return ret;
}


template <typename SpecificCommImpl>
void CommImpl<SpecificCommImpl>::Swap(
    CommImpl<SpecificCommImpl>& other) EL_NO_EXCEPT
{
    static_cast<SpecificCommImpl*>(this)->DoSwap(
        static_cast<SpecificCommImpl&>(other));
    std::swap(comm_, other.comm_);
}


// Private interface implementation


template <typename SpecificCommImpl>
void CommImpl<SpecificCommImpl>::SafeDuplicateMPIComm_(MPI_Comm src_comm)
{
    if (src_comm == MPI_COMM_NULL)
        return;

    try
    {
        EL_CHECK_MPI_CALL(MPI_Comm_dup(src_comm, &comm_));
    }
    catch (std::exception const& x)
    {
        // Reset the communicator to the only sensible thing.
        FreeAndResetInternalComm_();
        throw x;
    }
}


template <typename SpecificCommImpl>
void CommImpl<SpecificCommImpl>::FreeAndResetInternalComm_()
{
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized
        && (comm_ != MPI_COMM_WORLD)
        && (comm_ != MPI_COMM_NULL)
        && (comm_ != MPI_COMM_SELF))
    {
        EL_CHECK_MPI_CALL(
            MPI_Comm_free(&comm_));
    }
    comm_ = MPI_COMM_NULL;
}


template <typename SpecificCommImpl>
void CommImpl<SpecificCommImpl>::LocalReset_()
{
    FreeAndResetInternalComm_();
}

}// namespace mpi
}// namespace El
#endif /* EL_IMPORTS_MPI_COMM_IMPL_HPP_ */
