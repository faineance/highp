#pragma once
#include <stdint.h>
#include <math.h>
#include <xmmintrin.h>

#define INLINE __always_inline


class vec4 {
public:

    INLINE vec4() {
        m = _mm_setzero_ps();
    };
    INLINE explicit vec4(float w) {
        m = _mm_set1_ps(w);
    };

    INLINE explicit vec4(float x, float y, float z, float w) {
        m = _mm_setr_ps(z, y, x, w);
    }

    INLINE vec4(const __m128 &_m) {
        m = _m;
    }

    INLINE vec4 vec4i(int x, int y, int z, int w) {
        return vec4((float)x, (float)y, (float)z, (float)w);
    }

    INLINE friend vec4 operator+ (vec4 a, vec4 b) { return _mm_add_ps(a.m, b.m);  }
    INLINE friend vec4 operator- (vec4 a, vec4 b) { return _mm_sub_ps(a.m, b.m);  }
    INLINE friend vec4 operator* (vec4 a, vec4 b) { return _mm_mul_ps(a.m, b.m);  }
    INLINE friend vec4 operator/ (vec4 a, vec4 b) { return _mm_div_ps(a.m, b.m);  }

    INLINE friend vec4 operator* (vec4 a, float b) { return _mm_mul_ps(a.m, _mm_set1_ps(b)); }
    INLINE friend vec4 operator/ (vec4 a, float b) { return _mm_div_ps(a.m, _mm_set1_ps(b)); }
    INLINE friend vec4 operator* (float a, vec4 b) { return _mm_mul_ps(_mm_set1_ps(a), b.m); }
    INLINE friend vec4 operator/ (float a, vec4 b) { return _mm_div_ps(_mm_set1_ps(a), b.m); }

    INLINE friend vec4 operator==(vec4 a, vec4 b) { return _mm_cmpeq_ps(a.m, b.m); }
    INLINE friend vec4 operator!=(vec4 a, vec4 b) { return _mm_cmpneq_ps(a.m, b.m); }
    INLINE friend vec4 operator< (vec4 a, vec4 b) { return _mm_cmplt_ps(a.m, b.m); }
    INLINE friend vec4 operator> (vec4 a, vec4 b) { return _mm_cmpgt_ps(a.m, b.m); }
    INLINE friend vec4 operator<=(vec4 a, vec4 b) { return _mm_cmple_ps(a.m, b.m); }
    INLINE friend vec4 operator>=(vec4 a, vec4 b) { return _mm_cmpge_ps(a.m, b.m); }


    INLINE vec4 max(vec4 &v, float f) { return _mm_max_ps(v.m, _mm_set1_ps(f)); }

    INLINE vec4 max(vec4 &a, vec4 &b) { return _mm_max_ps(a.m, b.m); }

    INLINE vec4 min(vec4 &v, float f) { return _mm_min_ps(v.m, _mm_set1_ps(f)); }

    INLINE vec4 min(vec4 &a, vec4 &b) { return _mm_min_ps(a.m, b.m); }


    INLINE float sum(vec4 v) { return v.x + v.y + v.z + v.w; }

    INLINE float dot(vec4 a, vec4 b) { return sum(a * b); }

    INLINE float length(vec4 v) { return sqrtf(dot(v, v)); }

    INLINE float length_sqaured(vec4 v) { return dot(v, v); }

    INLINE vec4 norm(vec4 v) { return v * (1.0f / length(v)); }
    union {

        struct {
            float x, y, z, w;
        };

        struct {
            float r, g, b, a;
        };
    };


    __m128 m;
};