/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */
#pragma once
#include "DTypes.h"
#include "Meta.h"

namespace fl {

namespace detail {
    template<class T>
    struct FL_API dtype_traits_base {
        static constexpr dtype fl_type = [] {
            auto group = dtype_group_from_type<T>();

            for(size_t i = begin_of(group); i < end_of(group); ++i)
                if(auto type = static_cast<dtype>(i); size_of(type) == sizeof(T))
                    return type;

            throw std::logic_error{"unknown type size requested"};
        }();

        using base_type = T;
    };
}

template<class T>
struct dtype_traits;


#define FL_TYPE_TRAIT(T)                                               \
    template<>                                                         \
    struct FL_API dtype_traits<T> : detail::dtype_traits_base<T> {     \
        static constexpr std::string_view name() {                     \
            return #T;                                                 \
        }                                                              \
        /* deprecated, use @ref name() instead */                      \
        static constexpr const char* getName() {                       \
            return #T;                                                 \
        }                                                              \
    };

// using fundamental types instead of fixed to avoid missing templates when multiple fundamentals are equal size

FL_TYPE_TRAIT(float);
FL_TYPE_TRAIT(double);
FL_TYPE_TRAIT(int);
FL_TYPE_TRAIT(unsigned int);
FL_TYPE_TRAIT(char);
FL_TYPE_TRAIT(unsigned char);
FL_TYPE_TRAIT(long);
FL_TYPE_TRAIT(unsigned long);
FL_TYPE_TRAIT(long long);
FL_TYPE_TRAIT(unsigned long long);
FL_TYPE_TRAIT(bool);
FL_TYPE_TRAIT(short);
FL_TYPE_TRAIT(unsigned short);

namespace detail {
    //TODO add c++23 float16_t once version is bumped
    using fundamental_types = std::tuple<
        float,
        double,
        bool,
        char,
        unsigned char,
        short,
        unsigned short,
        int,
        unsigned int,
        long,
        unsigned long,
        long long,
        unsigned long long>;
}


} // namespace fl

namespace fl {
/**
 * @brief Checks if T is any of @ref fl::fundamental_types
 * @tparam T type to check
 */
template<class T>
concept fundamental_type = std::apply(
    []<class... Ts>(Ts...) { return dev::is_any_of<T, Ts...>; },
    detail::fundamental_types{}
);


/**
 * @brief Accepts if the type would resolve to any @ref fl::fundamental_types in overload resolution.
 * 
 * Let `f(X x)` be instantiated functions for `X` in @ref fl::fundamental_types .
 * The concept is satisfied if and only if `f(std::declval<T>())` is well-formed and unambiguous.
 * 
 * @tparam T type to check resolution for
 */
template<class T>
concept fundamental_type_compatible = fundamental_type<T> || std::apply(
    []<class... Ts>(Ts...) { return resolves_to_any_of<T, Ts...>; },
    detail::fundamental_types{}
);



}


namespace fl {
//TODO not really happy with this, return type deduction should be possible

/**
 * @brief Runtime matches dtype with a type from the list and calls the templated function
 * @tparam R function return type
 * @tparam TypeList type list to apply
 * @tparam Func templated function type
 * @param type dtype to runtime dispatch
 * @param func templated function to call
 * @return result of func<T>() where T corresponds to the dtype passed
 */
template<class R = void, class TypeList = detail::fundamental_types, class Func>
R dispatch_dtype(fl::dtype type, Func&& func) {
    std::conditional_t<std::same_as<R, void>, int, R> result{};
    bool found = false;

    auto try_dispatch = [&found, &result, type, &func]<class Type>() {
        if(!found && fl::dtype_traits<Type>::fl_type == type) {
            if constexpr(std::is_void_v<R>)
                func.template operator()<Type>();
            else
                result = func.template operator()<Type>();
            found = true;
        }
    };

    [&]<class... Ts>(std::tuple<Ts...>) { (try_dispatch.template operator()<Ts>(), ...); }(
        TypeList{}
    );

    if(!found)
        throw std::invalid_argument("Unsupported dtype for dispatch");

    // C++17 feature: only return if R isn't void
    if constexpr(!std::is_void_v<R>)
        return result;
    else
        return;
}
}
