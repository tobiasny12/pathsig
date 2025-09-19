// ExtendedPrecision.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>


namespace pathsig {
    // Double-double precision type for extended precision arithmetic
    typedef double2 double108_t;


    /// Device-side arithmetic operations for extended precision (double-double)
    namespace extended_prec {

        /// Addition: a + b
        __device__ __forceinline__ double108_t add_double108_t(double108_t a, double108_t b) {
            double108_t z;
            double t1, t2, t3, t4, t5, e;
            t1 = __dadd_rn(a.y, b.y);
            t2 = __dadd_rn(t1, -a.y);
            t3 = __dadd_rn(__dadd_rn(a.y, t2 - t1), __dadd_rn(b.y, -t2));
            t4 = __dadd_rn(a.x, b.x);
            t2 = __dadd_rn(t4, -a.x);
            t5 = __dadd_rn(__dadd_rn(a.x, t2 - t4), __dadd_rn(b.x, -t2));
            t3 = __dadd_rn(t3, t4);
            t4 = __dadd_rn(t1, t3);
            t3 = __dadd_rn(t1 - t4, t3);
            t3 = __dadd_rn(t3, t5);
            z.y = e = __dadd_rn(t4, t3);
            z.x = __dadd_rn(t4 - e, t3);
            return z;
        }


        /// Negation of a number: -a
        __device__ __forceinline__ double108_t neg_double108_t(double108_t a) {
            return make_double2(-a.x, -a.y);
        }


        /// Subtraction: a - b
        __device__ __forceinline__ double108_t sub_double108_t(double108_t a, double108_t b) {
            return add_double108_t(a, neg_double108_t(b));
        }


        /// Multiplication: : a * b
        __device__ __forceinline__ double108_t mul_double108(double108_t a, double108_t b) {
            double108_t t, z;
            double e;
            t.y = __dmul_rn(a.y, b.y);
            t.x = __fma_rn(a.y, b.y, -t.y);
            t.x = __fma_rn(a.x, b.x, t.x);
            t.x = __fma_rn(a.y, b.x, t.x);
            t.x = __fma_rn(a.x, b.y, t.x);
            z.y = e = __dadd_rn(t.y, t.x);
            z.x = __dadd_rn(t.y - e, t.x);
            return z;
        }


        /// Division by integer: a / divisor
        template<typename InvIntsArray>
     __device__ __forceinline__ double108_t div_double108_by_int(
         double108_t a,
         unsigned divisor,
         const InvIntsArray& c_inv_ints) {
            if (divisor == 1) return a;
            return mul_double108(a, c_inv_ints[divisor]);
        }
    } // namespace extended_prec
} // namespace pathsig