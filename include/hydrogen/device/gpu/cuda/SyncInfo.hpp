#ifndef HYDROGEN_DEVICE_GPU_CUDA_SYNCINFO_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_SYNCINFO_HPP_

#include <cuda_runtime_api.h>

#include <hydrogen/SyncInfo.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>

#include "CUDAError.hpp"
#include "CUDAManagement.hpp"

namespace hydrogen
{

template <>
class SyncInfo<Device::GPU>
{
public:
    SyncInfo()
        : SyncInfo{cuda::GetDefaultStream(), cuda::GetDefaultEvent()}
    {}

    SyncInfo(cudaStream_t stream, cudaEvent_t event)
        : stream_{stream}, event_{event}
    {}

    void Merge(SyncInfo<Device::GPU> const& si) noexcept
    {
        if (si.stream_)
            stream_ = si.stream_;
        if (si.event_)
            event_ = si.event_;
    }

    cudaStream_t Stream() const noexcept { return stream_; }
    cudaEvent_t Event() const noexcept { return event_; }
private:
    friend void DestroySyncInfo(SyncInfo<Device::GPU>&);
    cudaStream_t stream_;
    cudaEvent_t event_;
};// struct SyncInfo<Device::GPU>

inline void AddSynchronizationPoint(SyncInfo<Device::GPU> const& syncInfo)
{
    H_CHECK_CUDA(cudaEventRecord(syncInfo.Event(), syncInfo.Stream()));
}


namespace details
{
inline void AddSyncPoint(
    SyncInfo<Device::CPU> const& master,
    SyncInfo<Device::GPU> const& dependent)
{
}

inline void AddSyncPoint(
    SyncInfo<Device::GPU> const& master,
    SyncInfo<Device::CPU> const& dependent)
{
    // The CPU must wait for the GPU to catch up.
    Synchronize(master); // wait for "master"
}

// This captures the work done on A and forces "others" to wait for
// completion.
template <typename... Ts>
inline
void AddSyncPoint(
    SyncInfo<Device::GPU> const& master, SyncInfo<Device::GPU> const& other)
{
    if (master.Stream() != other.Stream())
        H_CHECK_CUDA(
            cudaStreamWaitEvent(other.Stream(), master.Event(), 0));
}
}// namespace details

inline void Synchronize(SyncInfo<Device::GPU> const& syncInfo)
{
    H_CHECK_CUDA(cudaStreamSynchronize(syncInfo.Stream()));
}

}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_SYNCINFO_HPP_
