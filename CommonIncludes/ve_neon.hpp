#pragma once

#include <arm_neon.h>

namespace ve {
    constexpr int32_t vector_size = 4;

    struct int_vector;
    
    template<typename tag_type>
    struct tagged_vector;
    
    struct fp_vector;

    struct mask_vector;

    template<typename T>
    struct ve_identity {
        using type = T;
    };

    struct vbitfield_type {
        using storage = uint8_t;

        uint8_t v;
    };

	RELEASE_INLINE vbitfield_type operator&(vbitfield_type a, vbitfield_type b) {
		return vbitfield_type{ uint8_t(a.v & b.v) };
	}
	RELEASE_INLINE vbitfield_type operator|(vbitfield_type a, vbitfield_type b) {
		return vbitfield_type{ uint8_t(a.v | b.v) };
	}
	RELEASE_INLINE vbitfield_type operator^(vbitfield_type a, vbitfield_type b) {
		return vbitfield_type{ uint8_t(a.v ^ b.v) };
	}
	RELEASE_INLINE vbitfield_type operator~(vbitfield_type a) {
		return vbitfield_type{ uint8_t(~a.v) };
	}
	RELEASE_INLINE vbitfield_type operator!(vbitfield_type a) {
		return vbitfield_type{ uint8_t(~a.v) };
	}
	RELEASE_INLINE vbitfield_type and_not(vbitfield_type a, vbitfield_type b) {
		return vbitfield_type{ uint8_t(a.v & (~b.v)) };
	}
	RELEASE_INLINE vbitfield_type operator!=(vbitfield_type a, vbitfield_type b) {
		return vbitfield_type{ uint8_t(a.v ^ b.v) };
	}
	RELEASE_INLINE vbitfield_type operator==(vbitfield_type a, vbitfield_type b) {
		return vbitfield_type{ uint8_t(~(a.v ^ b.v)) };
	}

    struct alignas(16) mask_vector {
        using wrapped_value = bool;
        uint32x4_t value;

        RELEASE_INLINE mask_vector() : value(vdupq_n_u32(0)) {}
        RELEASE_INLINE mask_vector(bool b) : value(vdupq_n_u32(b ? 0xFFFFFFFF : 0)) {}
        RELEASE_INLINE mask_vector(bool a, bool b, bool c, bool d) {
            uint32_t values[4] = {
                a ? 0xFFFFFFFF : 0,
                b ? 0xFFFFFFFF : 0,
                c ? 0xFFFFFFFF : 0,
                d ? 0xFFFFFFFF : 0
            };
            value = vld1q_u32(values);
        }
        RELEASE_INLINE mask_vector(vbitfield_type b) {
            uint32_t values[4] = {
                (b.v & 0x01) ? 0xFFFFFFFF : 0,
                (b.v & 0x02) ? 0xFFFFFFFF : 0,
                (b.v & 0x04) ? 0xFFFFFFFF : 0,
                (b.v & 0x08) ? 0xFFFFFFFF : 0
            };
            value = vld1q_u32(values);
        }
        RELEASE_INLINE constexpr mask_vector(uint32x4_t v) : value(v) {}

        RELEASE_INLINE bool operator[](uint32_t i) const noexcept {
            return vgetq_lane_u32(value, i) != 0;
        }
        RELEASE_INLINE void set(uint32_t i, bool v) noexcept {
            uint32_t tmp[4];
            vst1q_u32(tmp, value);
            tmp[i] = v ? 0xFFFFFFFF : 0;
            value = vld1q_u32(tmp);
        }
        RELEASE_INLINE operator vbitfield_type() const noexcept {
            uint32_t tmp[4];
            vst1q_u32(tmp, value);
            return vbitfield_type{uint8_t(
                ((tmp[0] != 0) << 0) |
                ((tmp[1] != 0) << 1) |
                ((tmp[2] != 0) << 2) |
                ((tmp[3] != 0) << 3)
            )};
        }
    };

    struct alignas(16) fp_vector {
        using wrapped_value = float;
        float32x4_t value;

        RELEASE_INLINE fp_vector() : value(vdupq_n_f32(0)) {}
        RELEASE_INLINE constexpr fp_vector(float32x4_t v) : value(v) {}
        RELEASE_INLINE fp_vector(float v) : value(vdupq_n_f32(v)) {}
        RELEASE_INLINE fp_vector(float a, float b, float c, float d) {
            float values[4] = {a, b, c, d};
            value = vld1q_f32(values);
        }

        RELEASE_INLINE float reduce() const {
            float32x2_t sum = vadd_f32(vget_low_f32(value), vget_high_f32(value));
            return vget_lane_f32(vpadd_f32(sum, sum), 0);
        }

        RELEASE_INLINE float operator[](uint32_t i) const noexcept {
            return vgetq_lane_f32(value, i);
        }
        RELEASE_INLINE void set(uint32_t i, float v) noexcept {
            float tmp[4];
            vst1q_f32(tmp, value);
            tmp[i] = v;
            value = vld1q_f32(tmp);
        }
    };
}