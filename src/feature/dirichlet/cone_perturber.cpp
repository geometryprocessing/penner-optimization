#include "feature/dirichlet/cone_perturber.h"

#include "optimization/core/constraint.h"
#include "util/vf_corners.h"

namespace Penner {
namespace Feature {

void ConePerturber::perturb_cone(DirichletPennerConeMetric& m, VectorX& rotation_form, int hij)
{
    // modify the rotation form and resulting cone angles
    int hji = m.opp[hij];
    int vi = m.v_rep[m.to[hji]];
    int vj = m.v_rep[m.to[hij]];
    rotation_form[hij] -= (M_PI / 2.);
    rotation_form[hji] += M_PI / 2.;

    // doubled mesh case
    if (m.type[hij] != 0) {
        // modify reflected edge 
        rotation_form[m.R[hij]] += M_PI / 2.;
        rotation_form[m.opp[m.R[hij]]] -= M_PI / 2.;

        // propagate (doubled) change in sector angles to derived angle sums
        m.Th_hat[vi] += M_PI;
        m.Th_hat[vj] -= M_PI;
    }
    // closed mesh case
    else {
        // propagate change in sector angles to derived angle sums
        m.Th_hat[vi] += M_PI / 2.;
        m.Th_hat[vj] -= M_PI / 2.;
    }
}


void ConePerturber::perturb_boundary_cones(
    DirichletPennerConeMetric& m,
    VectorX& rotation_form,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    Scalar min_angle)
{
    // get initial metric actual angles
    VectorX Theta = Optimization::compute_cone_angles(m);

    // get map from halfedge endpoints to halfedges
    Eigen::SparseMatrix<int> vv2he = generate_VV_to_halfedge_map(m, vtx_reindex, V_map);

    // get set of degenerate cones to correct
    std::queue<int> degenerate_cones = find_degenerate_cones(m, min_angle);
    spdlog::info("Fixing {} degenerate cones", degenerate_cones.size());

    // iterate until all degenerate cones gone
    while (!degenerate_cones.empty()) {
        // get cone to fix
        int vi = degenerate_cones.front();
        degenerate_cones.pop();

        // check if cone already nondegenerate
        if (m.Th_hat[m.v_rep[vi]] > min_angle - 1e-6) continue;
        spdlog::info("Fixing {} cone at {} in rotation form", m.Th_hat[m.v_rep[vi]], vi);

        // find boundary edges (or determine if interior edge)
        // TODO: should check both adjacent edges for max cone
        std::array<int, 2> h_bd = find_adjacent_boundary_edges(m, vi);
        if ((h_bd[0] < 0) || (h_bd[1] < 0)) continue; // skip interior edges

        // case: both adjacent boundary angles are small; push defect to interior
        Scalar theta_0 = m.Th_hat[m.v_rep[m.to[h_bd[0]]]];
        Scalar theta_1 = m.Th_hat[m.v_rep[m.to[h_bd[1]]]];
        spdlog::info("Candidate cones: {} and {}", theta_0, theta_1);
        if ((theta_0 < min_angle + 1e-6) && (theta_1 < min_angle + 1e-6)) {
            // find the adjacent cone with largest curvature
            int h_opt = find_max_adjacent_cone(m, vi);

            // perturb cone angles along the optimal halfedge
            int vj = m.to[h_opt];
            spdlog::debug("Decreasing cone {} at {}", m.Th_hat[m.v_rep[vj]], vj);
            perturb_cone(m, rotation_form, h_opt);

            // check if any modified vertices are now a cone
            for (int v : {vi, vj}) {
                if (m.Th_hat[m.v_rep[v]] < min_angle - 1e-6) {
                    degenerate_cones.push(v);
                }
            }

            continue;
        }

        // check difference between target and actual cone angle
        Scalar dtheta_0 = Theta[m.v_rep[m.to[h_bd[0]]]] - theta_0;
        Scalar dtheta_1 = Theta[m.v_rep[m.to[h_bd[1]]]] - theta_1;
        spdlog::trace("Candidate cone decreases: {} and {}", dtheta_0, dtheta_1);

        // use angle with smaller difference from target, unless near degenerate
        bool use_next_primal = (dtheta_0 < dtheta_1);
        if (theta_0 < min_angle + 1e-6) use_next_primal = false;
        if (theta_1 < min_angle + 1e-6) use_next_primal = true;

        // find cones to modify
        int h0, h1;
        int vj, wi, wj;
        if (use_next_primal) {
            // use next primal halfege
            h0 = h_bd[0];

            // find paired primal halfedge in cut mesh
            vj = m.to[h0];
            int Vi = V_map[vtx_reindex[m.v_rep[vi]]];
            int Vj = V_map[vtx_reindex[m.v_rep[vj]]];
            h1 = vv2he.coeffRef(Vj, Vi) - 1;
        } else {
            // use next double halfege (previous in primal)
            h0 = h_bd[1];

            // find paired double halfedge in cut mesh 
            vj = m.to[h0];
            int Vi = V_map[vtx_reindex[m.v_rep[vi]]];
            int Vj = V_map[vtx_reindex[m.v_rep[vj]]];
            h1 = m.opp[vv2he.coeffRef(Vi, Vj) - 1];
        }
        wi = m.to[h1];
        wj = m.to[m.opp[h1]];

        // modify cones on target edge
        spdlog::debug("Decreasing cone {} at {}", m.Th_hat[m.v_rep[vj]], vj);
        perturb_cone(m, rotation_form, h0);

        // modify cones on opposite edge if sufficient cone angle
        if (m.Th_hat[m.v_rep[wi]] > (2. * M_PI) - 1e-6) {
            spdlog::debug("Decreasing cone {} at {}", m.Th_hat[m.v_rep[wi]], wi);
            spdlog::debug("Increasing cone {} at {}", m.Th_hat[m.v_rep[wj]], wj);
            perturb_cone(m, rotation_form, h1);
        }

        // check if any modified vertices are now a cone
        for (int v : {vi, vj, wi, wj}) {
            if (m.Th_hat[m.v_rep[v]] < min_angle - 1e-6) {
                degenerate_cones.push(v);
            }
        }
    }
}

bool ConePerturber::remove_cone_pair(
    DirichletPennerConeMetric& m,
    VectorX& rotation_form,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map,
    Scalar min_angle)
{
    // get sum of sector angle defects
    auto glued_angle_defects = compute_glued_angle_defects(m, vtx_reindex, V_map);

    int num_halfedges = m.n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // get vertices at ends of edge
        int hji = m.opp[hij];
        int vi = m.to[hji];
        int vj = m.to[hij];
        int Vi = V_map[vtx_reindex[m.v_rep[vi]]];
        int Vj = V_map[vtx_reindex[m.v_rep[vj]]];
        spdlog::trace("Candidate period jump change halfedge: {}", hij);

        // only modify halfedge with positive defect at base and negative cone at tip
        if (glued_angle_defects[Vi] <= 1e-6) continue; // skip nonpositive base
        if (glued_angle_defects[Vj] >= -1e-6) continue; // skip nonnegative tip
        spdlog::debug("Positive candidate cone has defect {}", glued_angle_defects[Vi]);
        spdlog::debug("Negative candidate cone has defect {}", glued_angle_defects[Vj]);

        // don't reduce sector angles below the minimum cone angle
        spdlog::debug("Negative candidate cone: {}", m.Th_hat[m.v_rep[vj]]);
        if (m.Th_hat[m.v_rep[vj]] < min_angle + 1e-6) continue;

        // perform perturbation
        spdlog::info("Fixing pos {} and neg {} cone pair", Vi, Vj);
        spdlog::debug("Reducing {} cone at {} in rotation form", m.Th_hat[m.v_rep[vj]], vi);
        spdlog::debug("Increasing {} cone at {} in rotation form", m.Th_hat[m.v_rep[vi]], vi);
        perturb_cone(m, rotation_form, hij);

        return true;
    }

    // if no cone modified, report failure
    return false;
}

// create queue of all cones below the minimal angle adjacent to vi
std::queue<int> ConePerturber::find_degenerate_cones(Mesh<Scalar>& m, Scalar min_angle)
{
    // find all cones below the minimm angle
    int num_vertices = m.n_vertices();
    std::queue<int> degenerate_cones = {};
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (m.Th_hat[m.v_rep[vi]] < min_angle - 1e-6) {
            degenerate_cones.push(vi);
        }
    }

    return degenerate_cones;
}

// find two outgoing boundary edges of boundary vertex vi
std::array<int, 2> ConePerturber::find_adjacent_boundary_edges(Mesh<Scalar>& m, int vi)
{
    // search outgoing halfedges for boundary edges
    int h_start = m.out[vi];
    int hij = h_start;
    std::array<int, 2> h_bd = {-1, -1};
    do {
        // check for primal boundary halfedge
        if ((m.type[hij] == 1) && (m.type[m.opp[hij]] == 2)) {
            spdlog::debug("boundary edge found");
            h_bd[0] = hij;
        }

        // check for boundary halfedge in double 
        if ((m.type[hij] == 2) && (m.type[m.opp[hij]] == 1)) {
            spdlog::debug("opp boundary edge found");
            h_bd[1] = hij;
        }

        // circulate halfedge
        hij = m.opp[m.n[m.n[hij]]];
    } while (hij != h_start);

    return h_bd;
}

// find halfedge outgoing from vi with maximum curvature at tip
int ConePerturber::find_max_adjacent_cone(Mesh<Scalar>& m, int vi)
{
    // search outgoing halfedges for maximum angle
    int h_start = m.out[vi];
    int hij = h_start;
    Scalar max_angle = -1.;
    int vj, h_opt;
    do {
        vj = m.to[hij];

        // FIXME if ((m.R[m.opp[hij]] != hij) && (Th_hat[vj] > max_angle))
        if (m.Th_hat[m.v_rep[vj]] > max_angle) {
            max_angle = m.Th_hat[m.v_rep[vj]];
            h_opt = hij;
            spdlog::info("Max angle {} at {}", max_angle, vj);
        }

        hij = m.opp[m.n[m.n[hij]]];
    } while (hij != h_start);

    return h_opt;
}


} // namespace Feature
} // namespace Penner
