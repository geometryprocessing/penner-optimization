/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2023 Paper     *
*  `Metric Optimization in Penner Coordinates`           *
*  v1.0                                                                          *
*                                                                                *
*  The MIT License                                                               *
*                                                                                *
*  Permission is hereby granted, free of charge, to any person obtaining a       *
*  copy of this software and associated documentation files (the "Software"),    *
*  to deal in the Software without restriction, including without limitation     *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,      *
*  and/or sell copies of the Software, and to permit persons to whom the         *
*  Software is furnished to do so, subject to the following conditions:          *
*                                                                                *
*  The above copyright notice and this permission notice shall be included in    *
*  all copies or substantial portions of the Software.                           *
*                                                                                *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE  *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING       *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  *
*  IN THE SOFTWARE.                                                              *
*                                                                                *
*  Author(s):                                                                    *
*  Ryan Capouellez, Denis Zorin,                                                 *
*  Courant Institute of Mathematical Sciences, New York University, USA          *
*                                          *                                     *
*********************************************************************************/
#include "area.hh"

#include "embedding.hh"

namespace CurvatureMetric {

Scalar squared_area(Scalar li, Scalar lj, Scalar lk)
{
    // Sort the lengths for numerical stability
    Scalar a = li;
    Scalar b = lj;
    Scalar c = lk;
    if (a < b) swap(a, b);
    if (a < c) swap(a, c);
    if (b < c) swap(b, c);

    // Compute the area
    Scalar A = a + (b + c);
    Scalar B = c - (a - b);
    Scalar C = c + (a - b);
    Scalar D = a + (b - c);

    return A * B * C * D / 16.0;
}

Scalar squared_area_length_derivative(Scalar variable_length, Scalar lj, Scalar lk)
{
    // Sort the lengths and keep track of the derivative edge.
    Scalar a = variable_length;
    Scalar b = lj;
    Scalar c = lk;
    bool deriv_a = true;
    bool deriv_b = false;
    if (a < b) {
        swap(a, b);
        deriv_a = false;
        deriv_b = true;
    }
    if (a < c) {
        swap(a, c);
        deriv_a = false;
    }
    if (b < c) {
        swap(b, c);
        // The derivative is b iff the derivate was c before the swap
        deriv_b = !(deriv_a || deriv_b);
    }

    // Compute stable factors for Heron's formula
    Scalar S = a + (b + c);
    Scalar A = c - (a - b);
    Scalar B = c + (a - b);
    Scalar C = a + (b - c);

    // Compute terms minus a term
    Scalar TmS = (A * B * C) / 16.0;
    Scalar TmA = (S * B * C) / 16.0;
    Scalar TmB = (S * A * C) / 16.0;
    Scalar TmC = (S * A * B) / 16.0;

    // Compute the derivative for li
    if (deriv_a) {
        return TmS - TmA + TmB + TmC;
    } else if (deriv_b) {
        return TmS + TmA - TmB + TmC;
    } else {
        return TmS + TmA + TmB - TmC;
    }
}

VectorX squared_areas(const DifferentiableConeMetric& cone_metric)
{
    int num_halfedges = cone_metric.n_halfedges();
    VectorX he2areasq(num_halfedges);

    int num_faces = cone_metric.h.size();
    // #pragma omp parallel for
    for (int f = 0; f < num_faces; f++) {
        // Get halfedges of face f
        int hi = cone_metric.h[f];
        int hj = cone_metric.n[hi];
        int hk = cone_metric.n[hj];

        // Get lengths of the halfedges
        Scalar li = cone_metric.l[hi];
        Scalar lj = cone_metric.l[hj];
        Scalar lk = cone_metric.l[hk];

        // Compute the area of the face adjacent to the halfedges
        Scalar areasq = squared_area(li, lj, lk);
        he2areasq[hi] = areasq;
        he2areasq[hj] = areasq;
        he2areasq[hk] = areasq;
    }

    return he2areasq;
}

VectorX areas(const DifferentiableConeMetric& cone_metric)
{
    int num_halfedges = cone_metric.n_halfedges();

    // Compute squared areas
    VectorX he2areasq = squared_areas(cone_metric);
    assert(he2areasq.size() == num_halfedges);

    // Take square roots
    VectorX he2area(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        he2area[h] = sqrt(std::max<Scalar>(he2areasq[h], 0.0));
    }

    return he2area;
}

VectorX squared_area_length_derivatives(const DifferentiableConeMetric& cone_metric)
{
    int num_halfedges = cone_metric.n_halfedges();
    VectorX he2areasqderiv(num_halfedges);

    // Compute maps from halfedges to derivatives of area with respect to the edge
    // length
    int num_faces = cone_metric.h.size();
    // #pragma omp parallel for
    for (int f = 0; f < num_faces; f++) {
        // Get halfedges of face f
        int hi = cone_metric.h[f];
        int hj = cone_metric.n[hi];
        int hk = cone_metric.n[hj];

        // Get lengths of the halfedges
        Scalar li = cone_metric.l[hi];
        Scalar lj = cone_metric.l[hj];
        Scalar lk = cone_metric.l[hk];

        // Compute the derivative of the area of f with respect to each halfedge
        he2areasqderiv[hi] = squared_area_length_derivative(li, lj, lk);
        he2areasqderiv[hj] = squared_area_length_derivative(lj, lk, li);
        he2areasqderiv[hk] = squared_area_length_derivative(lk, li, lj);
    }

    return he2areasqderiv;
}

VectorX squared_area_log_length_derivatives(const DifferentiableConeMetric& cone_metric)
{
    int num_halfedges = cone_metric.n_halfedges();

    // Compute squared areas length derivatives
    VectorX he2areasq_deriv = squared_area_length_derivatives(cone_metric);
    assert(he2areasq_deriv.size() == num_halfedges);

    // Apply chain rule to A(l) = A(e^(lambda/2))
    VectorX he2areasq_log_deriv(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        he2areasq_log_deriv[h] = he2areasq_deriv[h] * cone_metric.l[h] / 2.0;
    }

    return he2areasq_log_deriv;
}

} // namespace CurvatureMetric
