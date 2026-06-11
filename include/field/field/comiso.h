// Copyright (C) 2026 Ryan Capouellez
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#pragma once

// WARNING: CoMISo uses a more restrictive copyleft license
#if USE_COMISO
#include <CoMISo/Solver/ConstrainedSolver.hh>
#include <CoMISo/Solver/GMM_Tools.hh>
#include <CoMISo/Solver/MISolver.hh>

#include "field/rounder.h"

namespace Penner {
namespace Field {



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


}
}

#endif