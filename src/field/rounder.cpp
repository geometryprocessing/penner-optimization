#include "field/rounder.h"

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

// WARNING: CoMISo uses a more restrictive copyleft license
#if USE_COMISO
#include <CoMISo/Solver/ConstrainedSolver.hh>
#include <CoMISo/Solver/GMM_Tools.hh>
#include <CoMISo/Solver/MISolver.hh>


// TODO Make option and make cone rounder code public
class ConeMISolver : public COMISO::MISolver
{
public:
    ConeMISolver() {};
    void solve_cone_rounding( 
        CSCMatrix& _A, 
        Vecd&      _x, 
        Vecd&      _rhs, 
        Veci&      _to_round,
        Rounder& rounder) {
            // StopWatch
            COMISO::StopWatch sw;
            double time_search_next_integer = 0;

            // some statistics
            n_local_ = 0;
            n_cg_    = 0;
            n_full_  = 0;

            // reset cholmod step flag
            cholmod_step_done_ = false;

            Veci to_round(_to_round);
            // copy to round vector and make it unique
            std::sort(to_round.begin(), to_round.end());
            Veci::iterator last_unique;
            last_unique = std::unique(to_round.begin(), to_round.end());
            int r = last_unique - to_round.begin();
            to_round.resize( r);

            // initalize old indices
            Veci old_idx(_rhs.size());
            for(unsigned int i=0; i<old_idx.size(); ++i)
                old_idx[i] = i;

            if( initial_full_solution_)
            {
                if( noisy_ > 2) std::cerr << "initial full solution" << std::endl;
                direct_solver_.calc_system_gmm(_A);
                direct_solver_.solve(_x, _rhs);

                cholmod_step_done_ = true;

                ++n_full_;
            }

            // neighbors for local optimization
            Vecui neigh_i;

            // Vector for reduced solution
            Vecd xr(_x);

            // loop until solution computed
            for(unsigned int i=0; i<to_round.size(); ++i)
            {
                if( noisy_ > 0)
                {
                std::cerr << "Integer DOF's left: " << to_round.size()-(i+1) << " ";
                if( noisy_ > 1)
                    std::cerr << "residuum_norm: " << COMISO_GMM::residuum_norm( _A, xr, _rhs) << std::endl;
                }

                // position in round vector
                std::vector<int> tr_best;

                sw.start();

                RoundingSet rset;
                rset.set_threshold(multiple_rounding_threshold_);

                // find index yielding smallest rounding error
                for(unsigned int j=0; j<to_round.size(); ++j)
                {
                if( to_round[j] != -1)
                {
                int cur_idx = to_round[j];
                double rnd_error = fabs( rounder.test_round(old_idx[cur_idx], xr[cur_idx]) - xr[cur_idx]);

                rset.add(j, rnd_error);
                }
                }

                rset.get_ids( tr_best);

                time_search_next_integer += sw.stop();
            
                // nothing more to do?
                if( tr_best.empty() )
                break;

                if( noisy_ > 5)
                std::cerr << "round " << tr_best.size() << " variables simultaneously\n";

                // clear neigh for local update
                neigh_i.clear();

                for(unsigned int j = 0; j<tr_best.size(); ++j)
                {
                int i_cur = to_round[tr_best[j]];

                // store rounded value
                double rnd_x = rounder.commit_round(old_idx[i_cur], xr[i_cur]);
                _x[ old_idx[i_cur] ] = rnd_x;

                // compute neighbors
                Col col = gmm::mat_const_col(_A, i_cur);
                ColIter it  = gmm::vect_const_begin( col);
                ColIter ite = gmm::vect_const_end  ( col);
                for(; it!=ite; ++it)
                if(it.index() != (unsigned int)i_cur)
                neigh_i.push_back(it.index());

                // eliminate var
                COMISO_GMM::fix_var_csc_symmetric( i_cur, rnd_x, _A, xr, _rhs);
                to_round[tr_best[j]] = -1;
                }

                // 3-stage update of solution w.r.t. roundings
                // local GS / CG / SparseCholesky
                update_solution( _A, xr, _rhs, neigh_i);
            }

            // final full solution?
            if( final_full_solution_)
            {
                if( noisy_ > 2) std::cerr << "final full solution" << std::endl;

                if( gmm::mat_ncols( _A) > 0)
                {
                if(cholmod_step_done_)
                direct_solver_.update_system_gmm(_A);
                else
                direct_solver_.calc_system_gmm(_A);

                direct_solver_.solve( xr, _rhs);
                ++n_full_;
                }
            }

            // store solution values to result vector
            for(unsigned int i=0; i<old_idx.size(); ++i)
            {
                _x[ old_idx[i] ] = xr[i];
            }

            // output statistics
            if( stats_)
            {
                std::cerr << "\t" << __FUNCTION__ << " *** Statistics of MiSo Solver ***\n";
                std::cerr << "\t\t Number of CG    iterations  = " << n_cg_ << std::endl;
                std::cerr << "\t\t Number of LOCAL iterations  = " << n_local_ << std::endl;
                std::cerr << "\t\t Number of FULL  iterations  = " << n_full_ << std::endl;
                std::cerr << "\t\t Number of ROUNDING          = " << _to_round.size() << std::endl;
                std::cerr << "\t\t time searching next integer = " << time_search_next_integer / 1000.0 <<"s\n";
                std::cerr << std::endl;
        }

    }

};

#endif

}
}