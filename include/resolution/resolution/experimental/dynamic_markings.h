
#pragma once

#include "resolution/core/common.h"
#include "holonomy/core/common.h"
#include "holonomy/holonomy/newton.h"
#include "conformal_ideal_delaunay/ConformalInterface.hh"
#include "holonomy/holonomy/holonomy.h"
#include "resolution/core/holonomy_coordinates.h"

namespace Penner {
namespace Resolution {

class DynamicPennerConeMetric : public MarkedPennerConeMetric
{
public:
    VectorX m_alpha;
    VectorX m_rotation_form;

    DynamicPennerConeMetric(
        const Mesh<Scalar>& m,
        const VectorX& metric_coords,
        const VectorX& alpha,
        const VectorX& rotation_form)
        : MarkedPennerConeMetric(m, metric_coords, {}, {})
        , m_alpha(alpha)
        , m_rotation_form(rotation_form)
    {}

    VectorX& get_rotation_form() { return m_rotation_form; }
    const VectorX& get_rotation_form() const { return m_rotation_form; }

    VectorX& get_holonomy_coordinates() { return m_alpha; }
    const VectorX& get_holonomy_coordinates() const { return m_alpha; }

    virtual bool flip_ccw(int _h, bool Ptolemy = true) override
    {
        auto& rotation_form = get_rotation_form();

        // make rotation trivial across edge
        align_rotation_form_faces(*this, rotation_form, _h);

        // update holonomy 
        VectorX& alpha = get_holonomy_coordinates();
        update_holonomy_coordinates_under_ccw_flip(*this, alpha, _h);

        bool success = MarkedPennerConeMetric::flip_ccw(_h, Ptolemy);

        return success;
    }

    virtual void make_discrete_metric() override
    {
        // Make the copied mesh Delaunay with Ptolemy flips
        VectorX u;
        u.setZero(n_ind_vertices());
        DelaunayStats del_stats;
        SolveStats<Scalar> solve_stats;
        bool use_ptolemy = true;
        ConformalIdealDelaunay<Scalar>::MakeDelaunay(*this, u, del_stats, solve_stats, use_ptolemy);
        m_is_discrete_metric = true;

        // update markings
        MarkedMetricParameters marked_metric_params;
        marked_metric_params.use_connectivity = false;
        generate_basis_loops(*this, m_homology_basis_loops, marked_metric_params, {});

        // compute rotation
        const auto& rotation_form = get_rotation_form();
        const VectorX& alpha = get_holonomy_coordinates();
        int num_basis_loops = m_homology_basis_loops.size();
        kappa_hat.resize(num_basis_loops);
        for (int i = 0; i < num_basis_loops; ++i) {
            // Compute field rotation and metric holonomy
            Scalar rotation = Holonomy::compute_dual_loop_rotation(*this, rotation_form, *m_homology_basis_loops[i]);
            Scalar holonomy = Holonomy::compute_dual_loop_holonomy(*this, alpha, *m_homology_basis_loops[i]);

            // Constraint is the difference of the holonomy and rotation
            kappa_hat[i] = holonomy - rotation;
            spdlog::info("new kappa {}: {}", i, kappa_hat[i]);
        } 

        return;
    }

    virtual std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const override
    {
        return std::make_unique<DynamicPennerConeMetric>(DynamicPennerConeMetric(*this));
    }

    virtual std::unique_ptr<MarkedPennerConeMetric> clone_marked_metric() const override
    {
        return std::make_unique<DynamicPennerConeMetric>(DynamicPennerConeMetric(*this));
    }


};

}
}