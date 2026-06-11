// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "metric/common.h"
#include "metric/cone_metric.h"
#include "util/vector.h"
#include "util/map.h"

// TODO: This needs a lot of cleanup. To make it clean:
//   - The rounder should be moved to another file, and the CoMISo changes made in a fork
//   - The facet principal curvature should be in another file
//   - The double code should be much cleaner, e.g., by making a derived class

namespace Penner {
namespace Field {

class Rounder {
public:
    Rounder(
        const Mesh<Scalar>& m,
        const std::vector<int>& _var2he,
        const std::vector<int>& _halfedge_var_id,
        const std::vector<int>& _base_cones,
        const std::vector<int>& _min_cones)
    : var2he(_var2he)
    , halfedge_var_id(_halfedge_var_id)
    , base_cones(_base_cones)
    , min_cones(_min_cones)
    , is_rounded(m.n_halfedges(), false)
    , values(m.n_halfedges(), 0)
    , to(vector_compose(m.v_rep, m.to))
    , opp(m.opp)
    , is_double(m.type[0] > 0)
    {
        int num_halfedges = m.n_halfedges();
        int num_vertices = m.n_ind_vertices();
        cone_period_jumps = std::vector<std::vector<int>>(num_vertices, std::vector<int>());
        for (int hij = 0; hij < num_halfedges; ++hij)
        {
            // only add variable period jumps
            if (halfedge_var_id[hij] == -1) continue;

            // add halfedge to tip
            int vj = m.v_rep[m.to[hij]];
            cone_period_jumps[vj].push_back(hij);

            // add opposite halfedge to base
            int hji = m.opp[hij];
            int vi = m.v_rep[m.to[hji]];
            cone_period_jumps[vi].push_back(hji);
        }
    }

    int compute_cone_correction(int hij)
    {
        int vj = to[hij];
        int cone = base_cones[vj];
        for (int h : cone_period_jumps[vj])
        {
            cone += (is_double) ? (2 * values[h]) : values[h];
        }

        return (min_cones[vj] - cone);
    }

    bool is_zero_cone(int hij)
    {
        int vj = to[hij];
        int cone = base_cones[vj];
        for (int h : cone_period_jumps[vj])
        {
            if ((h != hij) && (!is_rounded[h])) return false;
            cone += (is_double) ? (2 * values[h]) : values[h];
        }

        return (cone < min_cones[vj]);
    }

    int test_round(int id, Scalar x)
    {
        int hij = var2he[id];
        //int rounded_value = ((x)<0?int((x)-0.5):int((x)+0.5));
        int rounded_value = lround(x);
        int hji = opp[hij];
        values[hij] = rounded_value;
        values[hji] = -rounded_value;
        bool is_tip_cone = is_zero_cone(hij);
        bool is_base_cone = is_zero_cone(hji);
        
        if ((is_tip_cone) && (is_base_cone))
        {
            spdlog::warn("Cone at both tip and base of period jump halfedge");
            return rounded_value;
        }
        if (is_tip_cone)
        {
            spdlog::trace("Cone at tip of period jump halfedge");
            int n = compute_cone_correction(hij);
            if (n > 1) spdlog::trace("correction at tip is {}", n);
            if (n < 0) spdlog::error("correction at tip is {}", n);
            return rounded_value + n;
        }
        if (is_base_cone)
        {
            spdlog::trace("Cone at base of period jump halfedge");
            int n = compute_cone_correction(hji);
            if (n > 1) spdlog::trace("correction at base is {}", n);
            if (n < 0) spdlog::error("correction at tip is {}", n);
            return rounded_value - n;
        }

        return rounded_value;

    }

    int commit_round(int id, Scalar x)
    {
        int rounded_value = test_round(id, x);
        int hij = var2he[id];
        int hji = opp[hij];
        values[hij] = rounded_value;
        values[hji] = -rounded_value;
        is_rounded[hij] = true;
        is_rounded[hji] = true;
        return rounded_value;
    }

private:
    std::vector<int> var2he;
    std::vector<int> halfedge_var_id;
    std::vector<int> base_cones;
    std::vector<int> min_cones;
    std::vector<bool> is_rounded;
    std::vector<int> values;
    std::vector<std::vector<int>> cone_period_jumps;
    std::vector<int> to;
    std::vector<int> opp;
    bool is_double;

};

}
}