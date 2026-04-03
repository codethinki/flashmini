/*
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */
#pragma once
#include <tuple>

namespace fl {
namespace dev {

    template<class T, class... Ts>
    concept is_any_of = (std::same_as<T, Ts> || ...);


    template<class Tuple, class... Ts>
    struct unique_tuple_impl;

    template<class Tuple, class T, class... Ts> requires(is_any_of<T, Ts...>)
    struct unique_tuple_impl<Tuple, T, Ts...> : unique_tuple_impl<Tuple, Ts...> {};

    template<class... TupleTs, class T, class... Ts> requires(!is_any_of<T, Ts...>)
    struct unique_tuple_impl<std::tuple<TupleTs...>, T, Ts...> : unique_tuple_impl<std::tuple<TupleTs..., T>, Ts...> {};

    template<class Tuple>
    struct unique_tuple_impl<Tuple> {
        using types = Tuple;
    };
}



template<class... Ts>
struct unique_tuple : dev::unique_tuple_impl<std::tuple<>, Ts...> {};

template<class... Ts>
using unique_tuple_t = unique_tuple<Ts...>::types;
}

namespace fl {
namespace dev {
    template<class T>
    struct resolution_node {
        T operator()(T);
    };

    template<class... Ts>
    struct overload_set : resolution_node<Ts>... {
        using resolution_node<Ts>::operator()...;
    };
}

/**
 * Resolves the correct function overload type for T from Ts
 * @tparam T resolve target
 * @tparam Ts options to resolve from
 */
template<class T, class... Ts>
struct resolve_overload_from : std::invoke_result<dev::overload_set<Ts...>, T> {};


/**
 * shorthand for @ref resolve_overload_from
 * @tparam T resolve target
 * @tparam Ts options to resolve from
 */
template<class T, class... Ts>
using resolve_overload_from_t = resolve_overload_from<T, Ts...>::type;

/**
 * checks if T resolves to any of Ts via function overload resolution
 * @tparam T resolve target
 * @tparam Ts options to resolve from
 */
template<class T, class... Ts>
concept resolves_to_any_of = std::same_as<unique_tuple_t<Ts...>, std::tuple<Ts...>> && requires {
    typename resolve_overload_from<T, Ts...>::type;
};
}
