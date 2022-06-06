#ifndef HYDROGEN_META_TYPE_TRAITS_HPP_
#define HYDROGEN_META_TYPE_TRAITS_HPP_

#include <string>
#include <typeinfo>

namespace hydrogen
{

template <typename T>
struct TypeTraits
{
    static T One() noexcept { return T{1}; }
    static T Zero() noexcept { return T{0}; }
    static std::string Name() { return typeid(T).name(); }
};// struct TypeTraits

}// namespace hydrogen
#endif // HYDROGEN_META_TYPE_TRAITS_HPP_
