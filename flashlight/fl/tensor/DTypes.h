/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */
#pragma once
#include "flashlight/fl/common/Defines.h"

#include <array>
#include <concepts>
#include <format>
#include <optional>
#include <ostream>
#include <string>

namespace fl {


/**
 * Enumeration of all supported types
 */
enum class dtype {
    f16, // 16-bit float
    f32, // 32-bit float
    f64, // 64-bit float
    b8, // 8-bit boolean
    s16, // 16-bit signed integer
    s32, // 32-bit signed integer
    s64, // 64-bit signed integer
    u8, // 8-bit unsigned integer
    u16, // 16-bit unsigned integer
    u32, // 32-bit unsigned integer
    u64, // 64-bit unsigned integer

    DTYPES_SIZE
};
/**
 * Enumeration of the different type groups in @ref dtype
 */
enum class dtype_group {
    FLOAT,
    BOOL,
    SIGNED,
    UNSIGNED,

    DTYPE_GROUPS_SIZE
};



[[nodiscard]] constexpr std::string_view to_string(dtype e) {
    switch(e) {
        case dtype::f16: return "f16";
        case dtype::f32: return "f32";
        case dtype::f64: return "f64";
        case dtype::b8: return "b8";
        case dtype::s16: return "s16";
        case dtype::s32: return "s32";
        case dtype::s64: return "s64";
        case dtype::u8: return "u8";
        case dtype::u16: return "u16";
        case dtype::u32: return "u32";
        case dtype::u64: return "u64";
        default: return "unknown";
    }
}



[[nodiscard]] constexpr auto to_index(dtype d) { return static_cast<std::underlying_type_t<dtype>>(d); }

[[nodiscard]] constexpr auto to_index(dtype_group d) { return static_cast<std::underlying_type_t<dtype_group>>(d); }


/**
 * Library details, may change
 */
namespace detail {
    constexpr size_t DTYPES_SIZE = to_index(dtype::DTYPES_SIZE);

    /**
     * Array of dtype byte sizes
     */
    constexpr auto dtype_sizes = [] {
        std::array<size_t, DTYPES_SIZE> sizes{};
        sizes[to_index(dtype::f16)] = 2;
        sizes[to_index(dtype::f32)] = 4;
        sizes[to_index(dtype::f64)] = 8;
        sizes[to_index(dtype::b8)] = 1;
        sizes[to_index(dtype::s16)] = 2;
        sizes[to_index(dtype::s32)] = 4;
        sizes[to_index(dtype::s64)] = 8;
        sizes[to_index(dtype::u8)] = 1;
        sizes[to_index(dtype::u16)] = 2;
        sizes[to_index(dtype::u32)] = 4;
        sizes[to_index(dtype::u64)] = 8;
        return sizes;
    }();

    constexpr size_t DTYPE_GROUPS = to_index(dtype_group::DTYPE_GROUPS_SIZE);

    /**
     * Gets the dtype group for a c++ standard type
     * @tparam T to get group for
     * @return dtype group
     */
    template<class T>
    constexpr dtype_group dtype_group_from_type() {
        if constexpr(std::is_floating_point_v<T>) return dtype_group::FLOAT;
        else if constexpr(std::same_as<T, bool> || std::same_as<T, char>) return dtype_group::BOOL;
        else if constexpr(std::is_signed_v<T>) return dtype_group::SIGNED;
        else if constexpr(std::is_unsigned_v<T>) return dtype_group::UNSIGNED;
        else
            static_assert(DTYPE_GROUPS != 4, "unknown type group");
        return dtype_group{0};
    }

    constexpr auto dtype_group_begins = [] {
        std::array<dtype, detail::DTYPE_GROUPS> begins{};
        begins[to_index(dtype_group::FLOAT)] = dtype::f16;
        begins[to_index(dtype_group::BOOL)] = dtype::b8;
        begins[to_index(dtype_group::SIGNED)] = dtype::s16;
        begins[to_index(dtype_group::UNSIGNED)] = dtype::u8;
        return begins;
    }();

    constexpr auto dtype_group_lasts = [] {
        std::array<dtype, detail::DTYPE_GROUPS> lasts{};
        lasts[to_index(dtype_group::FLOAT)] = dtype::f64;
        lasts[to_index(dtype_group::BOOL)] = dtype::b8;
        lasts[to_index(dtype_group::SIGNED)] = dtype::s64;
        lasts[to_index(dtype_group::UNSIGNED)] = dtype::u64;
        return lasts;
    }();
}



/**
 * Gets the dtypes size in bytes
 * @param[in] type to get size of
 */
[[nodiscard]] FL_API constexpr size_t size_of(dtype type) { return detail::dtype_sizes[to_index(type)]; }

/**
 * Gets the dtype groups first dtype enum index
 * @param[in] group dtype group
 */
[[nodiscard]] FL_API constexpr size_t begin_of(dtype_group group) {
    return to_index(detail::dtype_group_begins[to_index(group)]);
}
/**
 * Gets the groups dtype enum end index (exclusive)
 */
[[nodiscard]] FL_API constexpr size_t end_of(dtype_group group) {
    return to_index(detail::dtype_group_lasts[to_index(group)]) + 1;
}
/**
 * Gets the size of the dtype group in the dtype enum
 */
[[nodiscard]] FL_API constexpr size_t size_of(dtype_group group) { return end_of(group) - begin_of(group); }


/**
 * Returns the size of the type in bytes.
 *
 * @param[in] type the input type to query.
 * @deprecated use @ref size_of(dtype) instead
 */
FL_API inline size_t getTypeSize(dtype type) { return size_of(type); }

/**
 * Convert a dtype to its string representation.
 * @deprecated use @ref to_string(fl::dtype) instead
 */
FL_API inline std::string dtypeToString(dtype type) { return std::string{to_string(type)}; }

/**
 * Tries to parse dtype from string
 * @param str type name
 * @return dtype or empty if not found
 */
FL_API std::optional<fl::dtype> dtype_from_string(std::string_view str);

/**
 * Converts string to a Flashlight dtype
 *
 * @param[in] string type name as a string.
 *
 * @return returns the corresponding Flashlight dtype
 * @deprecated use @dtype_from_string(std::string_view) instead
 */
FL_API inline fl::dtype stringToDtype(std::string const& string) { return *dtype_from_string(string); }


/**
 * Write a type's string representation to an output stream.
 */
FL_API inline std::ostream& operator<<(std::ostream& ostream, dtype const& s) {
    ostream << to_string(s);
    return ostream;
}

}

template<>
struct std::formatter<fl::dtype> {
    [[nodiscard]] constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
    template<class FormatContext> [[nodiscard]] constexpr auto format(fl::dtype const& obj, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "fl::dtype[{}]", to_string(obj));
    }
};
