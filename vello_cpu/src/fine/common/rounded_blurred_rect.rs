// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Drawing blurred, rounded rectangles.
//!
//! Implementation is adapted from: <https://git.sr.ht/~raph/blurrr/tree/master/src/distfield.rs>.

use crate::fine::{NumericVec, PosExt, ShaderResultF32};
use crate::kurbo::{Point, Vec2};
use vello_common::encode::EncodedBlurredRoundedRectangle;
use vello_common::fearless_simd::{Simd, SimdBase, SimdFloat, f32x4, f32x8, u8x16};

/// Columns between exact alpha evaluations on the subsampled path. Each
/// 64-byte u8 chunk is four columns of four rows, so a stride of four puts
/// one exact column at the start of every chunk.
const SUBSAMPLE_STRIDE: usize = 4;

/// Largest pixel step, in standard deviations, that may interpolate. The
/// alpha has second derivative of order 1/std_dev², so linear interpolation
/// over four columns stays within one u8 step of exact evaluation from
/// std_dev = 16 pixels up; smaller blurs evaluate every pixel.
const SUBSAMPLE_MAX_STEP_PER_STD_DEV: f32 = 1.0 / 16.0;

#[cfg(feature = "poly-pnorm")]
use vello_common::fearless_simd::Select;

#[cfg(all(not(feature = "std"), not(feature = "poly-pnorm")))]
use vello_common::kurbo::common::FloatFuncs as _;

#[derive(Debug)]
pub(crate) struct BlurredRoundedRectFiller<S: Simd> {
    r: f32x8<S>,
    g: f32x8<S>,
    b: f32x8<S>,
    a: f32x8<S>,
    invert: bool,
    subsample: bool,
    alpha_calculator: AlphaCalculator<S>,
}

impl<S: Simd> BlurredRoundedRectFiller<S> {
    pub(crate) fn new(
        simd: S,
        rect: &EncodedBlurredRoundedRectangle,
        start_x: f64,
        start_y: f64,
    ) -> Self {
        simd.vectorize(
            #[inline(always)]
            || {
                let start_pos = rect.transform * Point::new(start_x, start_y);
                let color_components = rect.color.as_premul_f32().components;
                let r = f32x8::splat(simd, color_components[0]);
                let g = f32x8::splat(simd, color_components[1]);
                let b = f32x8::splat(simd, color_components[2]);
                let a = f32x8::splat(simd, color_components[3]);
                // Interpolation runs along x only, so the x step must not
                // move y and must be small next to the blur.
                let axis_aligned = rect.x_advance.y == 0.0 && rect.y_advance.x == 0.0;
                let subsample = axis_aligned
                    && (rect.x_advance.x.abs() as f32) * rect.std_dev_inv
                        <= SUBSAMPLE_MAX_STEP_PER_STD_DEV;
                let simd_rect = SimdRoundedBlurredRect::new(rect, simd);
                let alpha_calculator = AlphaCalculator::new(
                    start_pos,
                    rect.x_advance,
                    rect.y_advance,
                    simd_rect,
                    simd,
                );

                Self {
                    alpha_calculator,
                    r,
                    g,
                    b,
                    a,
                    invert: rect.invert,
                    subsample,
                }
            },
        )
    }
}

impl<S: Simd> Iterator for BlurredRoundedRectFiller<S> {
    type Item = ShaderResultF32<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let mut next = self.alpha_calculator.next().unwrap();
        if self.invert {
            next = f32x8::splat(next.simd, 1.0) - next;
        }
        let r = self.r * next;
        let g = self.g * next;
        let b = self.b * next;
        let a = self.a * next;

        Some(ShaderResultF32 { r, g, b, a })
    }
}

impl<S: Simd> BlurredRoundedRectFiller<S> {
    /// Store two four-row columns of alphas each, as premultiplied RGBA8.
    #[inline(always)]
    fn store_columns(&self, first: f32x8<S>, second: f32x8<S>, chunk: &mut [u8; 64]) {
        let simd = self.a.simd;
        let (first, second) = if self.invert {
            let one = f32x8::splat(simd, 1.0);
            (one - first, one - second)
        } else {
            (first, second)
        };
        let r = u8x16::from_f32(simd, simd.combine_f32x8(self.r * first, self.r * second));
        let g = u8x16::from_f32(simd, simd.combine_f32x8(self.g * first, self.g * second));
        let b = u8x16::from_f32(simd, simd.combine_f32x8(self.b * first, self.b * second));
        let a = u8x16::from_f32(simd, simd.combine_f32x8(self.a * first, self.a * second));
        simd.store_four_interleaved_u8x16([r, g, b, a], chunk);
    }

    /// Evaluate one column in every `SUBSAMPLE_STRIDE` and interpolate the
    /// columns between, so a wide blur costs a quarter of the evaluations.
    fn paint_u8_subsampled(self, buf: &mut [u8]) {
        let simd = self.a.simd;
        simd.vectorize(
            #[inline(always)]
            || {
                let calc = &self.alpha_calculator;
                let stride = SUBSAMPLE_STRIDE as f64;
                // Each exact evaluation yields two columns; the second is
                // the right edge of the following chunk.
                let (mut left, ahead) = simd.split_f32x8(calc.columns(0.0, stride));
                let mut pending = Some(ahead);
                let mut col = 0.0;
                let quarter = f32x4::splat(simd, 0.25);
                let half = f32x4::splat(simd, 0.5);
                let three_quarters = f32x4::splat(simd, 0.75);
                let (chunks, _) = buf.as_chunks_mut::<64>();
                for chunk in chunks {
                    let right = match pending.take() {
                        Some(right) => right,
                        None => {
                            let (right, next) =
                                simd.split_f32x8(calc.columns(col + stride, col + 2.0 * stride));
                            pending = Some(next);
                            right
                        }
                    };
                    let delta = right - left;
                    self.store_columns(
                        simd.combine_f32x4(left, delta.mul_add(quarter, left)),
                        simd.combine_f32x4(
                            delta.mul_add(half, left),
                            delta.mul_add(three_quarters, left),
                        ),
                        chunk,
                    );
                    left = right;
                    col += stride;
                }
            },
        );
    }
}

impl<S: Simd> crate::fine::Painter for BlurredRoundedRectFiller<S> {
    fn paint_u8(mut self, buf: &mut [u8]) {
        if self.subsample {
            return self.paint_u8_subsampled(buf);
        }
        self.a.simd.vectorize(
            #[inline(always)]
            || {
                for chunk in buf.chunks_exact_mut(64) {
                    let first = self.next().unwrap();
                    let simd = first.r.simd;
                    let second = self.next().unwrap();

                    let r = u8x16::from_f32(simd, simd.combine_f32x8(first.r, second.r));
                    let g = u8x16::from_f32(simd, simd.combine_f32x8(first.g, second.g));
                    let b = u8x16::from_f32(simd, simd.combine_f32x8(first.b, second.b));
                    let a = u8x16::from_f32(simd, simd.combine_f32x8(first.a, second.a));

                    simd.store_four_interleaved_u8x16(
                        [r, g, b, a],
                        (&mut chunk[..]).try_into().unwrap(),
                    );
                }
            },
        );
    }

    fn paint_f32(mut self, buf: &mut [f32]) {
        self.a.simd.vectorize(
            #[inline(always)]
            || {
                for chunk in buf.chunks_exact_mut(32) {
                    let [c1, c2] = self.next().unwrap().get();
                    c1[0]
                        .simd
                        .store_four_interleaved_f32x4(c1, (&mut chunk[..16]).try_into().unwrap());
                    c2[0]
                        .simd
                        .store_four_interleaved_f32x4(c2, (&mut chunk[16..]).try_into().unwrap());
                }
            },
        );
    }
}

#[derive(Debug)]
struct AlphaCalculator<S: Simd> {
    cur_pos: Point,
    x_advance: Vec2,
    y_advance: Vec2,
    r: SimdRoundedBlurredRect<S>,
    simd: S,
}

impl<S: Simd> AlphaCalculator<S> {
    fn new(
        start_pos: Point,
        x_advance: Vec2,
        y_advance: Vec2,
        r: SimdRoundedBlurredRect<S>,
        simd: S,
    ) -> Self {
        Self {
            cur_pos: start_pos,
            x_advance,
            y_advance,
            r,
            simd,
        }
    }
}

impl<S: Simd> AlphaCalculator<S> {
    /// Alphas of the four-row columns `a` and `b` columns past the start.
    #[inline(always)]
    fn columns(&self, a: f64, b: f64) -> f32x8<S> {
        let simd = self.simd;
        let pa = self.cur_pos + a * self.x_advance;
        let pb = self.cur_pos + b * self.x_advance;
        let i = simd.combine_f32x4(
            f32x4::splat_pos(simd, pa.x as f32, 0.0, self.y_advance.x as f32),
            f32x4::splat_pos(simd, pb.x as f32, 0.0, self.y_advance.x as f32),
        );
        let j = simd.combine_f32x4(
            f32x4::splat_pos(simd, pa.y as f32, 0.0, self.y_advance.y as f32),
            f32x4::splat_pos(simd, pb.y as f32, 0.0, self.y_advance.y as f32),
        );
        self.alpha(i, j)
    }

    #[inline(always)]
    fn alpha(&self, i: f32x8<S>, j: f32x8<S>) -> f32x8<S> {
        let r = &self.r;

        let y = j - r.v1 * r.height;
        // Equivalent to r.r1 + y.abs() - (r.h * r.v1)
        let y0 = r.r1 - r.h.mul_sub(r.v1, y.abs());
        let y1 = y0.max(r.v0);

        let x = i - r.v1 * r.width;
        // Equivalent to r.r1 + x.abs() - (r.w * r.v1)
        let x0 = r.r1 - r.w.mul_sub(r.v1, x.abs());
        let x1 = x0.max(r.v0);
        #[cfg(feature = "poly-pnorm")]
        let d_pos = {
            // x1,y1 >= 0, and outside the corner squares one of them is exactly 0,
            // where (a^e + 0)^(1/e) == a, so only corners need the p-norm.
            let corner = self.simd.simd_gt_f32x8(x1.min(y1), r.v0);
            if self.simd.any_true_mask32x8(corner) {
                // (x^e + y^e)^(1/e) factors as max * phi(min/max); phi is bounded to
                // [1, 2^(1/e)] on [0,1], so a degree-7 fit covers it.

                // A lane with max == 0 divides to NaN; the select drops it.
                let mx = x1.max(y1);
                let t = x1.min(y1) / mx;
                let pnorm = mx
                    * r.phi[7]
                        .mul_add(t, r.phi[6])
                        .mul_add(t, r.phi[5])
                        .mul_add(t, r.phi[4])
                        .mul_add(t, r.phi[3])
                        .mul_add(t, r.phi[2])
                        .mul_add(t, r.phi[1])
                        .mul_add(t, r.phi[0]);
                corner.select(pnorm, x1 + y1)
            } else {
                x1 + y1
            }
        };
        #[cfg(not(feature = "poly-pnorm"))]
        let d_pos = (x1.powf(r.exponent) + y1.powf(r.exponent)).powf(r.recip_exponent);
        let d_neg = x0.max(y0).min(r.v0);
        let d = d_pos + d_neg - r.r1;
        r.scale
            * (f32x8::compute_erf7(self.simd, r.std_dev_inv * (r.min_edge + d))
                - f32x8::compute_erf7(self.simd, r.std_dev_inv * d))
    }
}

impl<S: Simd> Iterator for AlphaCalculator<S> {
    type Item = f32x8<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let i = f32x8::splat_pos(
            self.simd,
            self.cur_pos.x as f32,
            self.x_advance.x as f32,
            self.y_advance.x as f32,
        );
        let j = f32x8::splat_pos(
            self.simd,
            self.cur_pos.y as f32,
            self.x_advance.y as f32,
            self.y_advance.y as f32,
        );
        let z = self.alpha(i, j);
        self.cur_pos += 2.0 * self.x_advance;
        Some(z)
    }
}

#[cfg(feature = "poly-pnorm")]
// phi(t) = (1+t^e)^(1/e) as a degree-7 polynomial in t.
// Coefficients are degree-7 polynomials in u = (e-2)/1.478, e in [2, 3.478].

// PHI[k][j] is the u^j term of the t^k coefficient, fitted offline
// against exact pow. Worst case stays inside half an LSB.
/// Collapse `exponent` into the eight t-coefficients of phi. Public only so a
/// host test can sweep the fit against exact pow; PHI is otherwise unreachable.
pub fn phi_coefficients(exponent: f32) -> [f32; 8] {
    let u = (exponent - 2.0) / 1.478;
    core::array::from_fn(|k| {
        let row = &PHI[k];
        let mut c = row[7];
        for j in (0..7).rev() {
            c = c * u + row[j];
        }
        c
    })
}

#[cfg(feature = "poly-pnorm")]
const PHI: [[f32; 8]; 8] = [
    [
        0.999999523,
        0.000287747884,
        -0.00145391282,
        0.00325703970,
        -0.00416825246,
        0.00321776350,
        -0.00141784921,
        0.000275894592,
    ],
    [
        0.0000549813849,
        -0.0471561439,
        0.256915241,
        -0.630659223,
        0.894888878,
        -0.767846942,
        0.372579396,
        -0.0785049349,
    ],
    [
        0.498794109,
        -2.01443839,
        2.92649627,
        -0.978838682,
        -2.44131088,
        3.66302776,
        -2.15749884,
        0.496441036,
    ],
    [
        0.00960323121,
        4.57015324,
        -12.1590319,
        9.96781921,
        2.75401354,
        -10.4653912,
        7.21364307,
        -1.76673782,
    ],
    [
        -0.161359087,
        -6.13405132,
        23.4197960,
        -25.9309597,
        1.87568510,
        17.2587509,
        -13.6561842,
        3.50852728,
    ],
    [
        0.0696837083,
        5.55993032,
        -24.9317474,
        32.7130051,
        -7.60265636,
        -16.4634781,
        14.7194462,
        -3.92560983,
    ],
    [
        0.00442494359,
        -3.00162649,
        14.1223030,
        -20.5818748,
        6.76344347,
        8.46083069,
        -8.42855167,
        2.31515384,
    ],
    [
        -0.00698873494,
        0.704757929,
        -3.32011175,
        5.17456198,
        -2.03643107,
        -1.81569624,
        1.99027336,
        -0.559714437,
    ],
];

#[derive(Debug)]
struct SimdRoundedBlurredRect<S: Simd> {
    #[cfg(feature = "poly-pnorm")]
    pub phi: [f32x8<S>; 8],
    #[cfg(not(feature = "poly-pnorm"))]
    pub exponent: f32,
    #[cfg(not(feature = "poly-pnorm"))]
    pub recip_exponent: f32,
    pub scale: f32x8<S>,
    pub std_dev_inv: f32x8<S>,
    pub min_edge: f32x8<S>,
    pub w: f32x8<S>,
    pub h: f32x8<S>,
    pub width: f32x8<S>,
    pub height: f32x8<S>,
    pub r1: f32x8<S>,
    pub v0: f32x8<S>,
    pub v1: f32x8<S>,
}

impl<S: Simd> SimdRoundedBlurredRect<S> {
    fn new(encoded: &EncodedBlurredRoundedRectangle, s: S) -> Self {
        s.vectorize(
            #[inline(always)]
            || {
                let h = f32x8::splat(s, encoded.h);
                let w = f32x8::splat(s, encoded.w);
                let width = f32x8::splat(s, encoded.width);
                let height = f32x8::splat(s, encoded.height);
                let r1 = f32x8::splat(s, encoded.r1);
                #[cfg(feature = "poly-pnorm")]
                let coeffs = phi_coefficients(encoded.exponent);
                #[cfg(feature = "poly-pnorm")]
                let phi = core::array::from_fn(|k| f32x8::splat(s, coeffs[k]));
                #[cfg(not(feature = "poly-pnorm"))]
                let exponent = encoded.exponent;
                #[cfg(not(feature = "poly-pnorm"))]
                let recip_exponent = encoded.recip_exponent;
                let scale = f32x8::splat(s, encoded.scale);
                let min_edge = f32x8::splat(s, encoded.min_edge);
                let std_dev_inv = f32x8::splat(s, encoded.std_dev_inv);
                let v0 = f32x8::splat(s, 0.0);
                let v1 = f32x8::splat(s, 0.5);

                Self {
                    #[cfg(feature = "poly-pnorm")]
                    phi,
                    #[cfg(not(feature = "poly-pnorm"))]
                    exponent,
                    #[cfg(not(feature = "poly-pnorm"))]
                    recip_exponent,
                    scale,
                    std_dev_inv,
                    min_edge,
                    w,
                    v0,
                    v1,
                    h,
                    width,
                    height,
                    r1,
                }
            },
        )
    }
}

trait FloatExt<S: Simd> {
    // See https://raphlinus.github.io/audio/2018/09/05/sigmoid.html for a little
    // explanation of this approximation to the erf function.
    /// Approximate the erf function.
    fn compute_erf7(simd: S, x: Self) -> Self;
    #[cfg(not(feature = "poly-pnorm"))]
    fn powf(self, x: f32) -> Self;
}

impl<S: Simd> FloatExt<S> for f32x8<S> {
    #[inline(always)]
    fn compute_erf7(simd: S, x: Self) -> Self {
        // Clamp `x`, because for large `x` the terms here become `inf`, causing the result to be 0 or
        // `NaN`. This clamping doesn't lose any information, because `erf(±10) ≈ 1` well within `f64`
        // machine precision, let alone `f32`.
        let x = x.max(Self::splat(simd, -10.0)).min(Self::splat(simd, 10.0));
        let x = x * Self::splat(simd, core::f32::consts::FRAC_2_SQRT_PI);
        let xx = x * x;
        let p1 = Self::splat(simd, 0.0104).mul_add(xx, Self::splat(simd, 0.03395));
        let p2 = p1.mul_add(xx, Self::splat(simd, 0.24295));
        let p3 = x * xx;
        let x = p2.mul_add(p3, x);
        let denom = x.mul_add(x, Self::splat(simd, 1.0)).sqrt();
        x / denom
    }

    #[cfg(not(feature = "poly-pnorm"))]
    #[inline]
    fn powf(mut self, x: f32) -> Self {
        // TODO: SIMD
        self[0] = self[0].powf(x);
        self[1] = self[1].powf(x);
        self[2] = self[2].powf(x);
        self[3] = self[3].powf(x);
        self[4] = self[4].powf(x);
        self[5] = self[5].powf(x);
        self[6] = self[6].powf(x);
        self[7] = self[7].powf(x);

        self
    }
}
