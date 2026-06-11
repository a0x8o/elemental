#ifndef HYDROGEN_SYNCHRONIZEAPI_HPP_
#define HYDROGEN_SYNCHRONIZEAPI_HPP_

#include "SyncInfo.hpp"

namespace hydrogen
{

// This synchronizes the additional SyncInfos to the "master". That
// is, the execution streams described by the "others" will wait
// for the "master" stream.
template <Device D, Device D2, Device... Ds>
void AddSynchronizationPoint(SyncInfo<D> const &master,
                             SyncInfo<D2> const &other,
                             SyncInfo<Ds> const &...others)
{
#ifdef HYDROGEN_HAVE_GPU
    if constexpr (D == Device::GPU && D == D2) {
        // When the streams are the same, there is no need to create
        // synchronization points. Skip "other" call recursively with the rest.
        if (master.Stream() == other.Stream())
        {
<<<<<<< HEAD
<<<<<<< HEAD
            if constexpr (sizeof...(others) > 0UL)
                AddSynchronizationPoint(master, others...);
=======
            AddSynchronizationPoint(master, others...);
>>>>>>> 315bc6461 (Reduce the number of event record calls (#153))
=======
            if constexpr (sizeof...(others) > 0UL)
                AddSynchronizationPoint(master, others...);
>>>>>>> a0d3bbdd2 (Fix a subtle parameter expansion issue in sync info (#155))
            return;
        }
    }
#endif // HYDROGEN_HAVE_GPU

    AddSynchronizationPoint(master);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    int dummy[] = {(details::AddSyncPoint(master, other), 0),
                   (details::AddSyncPoint(master, others), 0)...};
    (void)dummy;
=======
    int dummy[] = {(details::AddSyncPoint(master, b),
=======
    int dummy[] = {(details::AddSyncPoint(master, other),
>>>>>>> 12f0180e2 (Fix naming bug (#154))
                    details::AddSyncPoint(master, others), 0)...};
    (void)dummy;
}

// Specialization of the above function for two arguments
template <Device D, Device D2>
void AddSynchronizationPoint(SyncInfo<D> const &master,
                             SyncInfo<D2> const &other)
{
#ifdef HYDROGEN_HAVE_GPU
    if constexpr (D == Device::GPU && D == D2)
    {
        // When the two streams are the same, there is no need to create
        // synchronization points.
        if (master.Stream() == other.Stream())
        {
            return;
        }
    }
#endif // HYDROGEN_HAVE_GPU

    AddSynchronizationPoint(master);
>>>>>>> 315bc6461 (Reduce the number of event record calls (#153))
}

=======
    int dummy[] = {(details::AddSyncPoint(master, other), 0),
                   (details::AddSyncPoint(master, others), 0)...};
    (void)dummy;
}

>>>>>>> a0d3bbdd2 (Fix a subtle parameter expansion issue in sync info (#155))
template <Device D, Device... Ds>
void AllWaitOnMaster(SyncInfo<D> const &master, SyncInfo<Ds> const &...others)
{
    AddSynchronizationPoint(master, others...);
}

template <Device D, Device... Ds>
void MasterWaitOnAll(SyncInfo<D> const &master, SyncInfo<Ds> const &...others)
{
    int dummy[] = {(AddSynchronizationPoint(others, master), 0)...};
    (void)dummy;
}

} // namespace hydrogen
#endif // HYDROGEN_SYNCHRONIZEAPI_HPP_
