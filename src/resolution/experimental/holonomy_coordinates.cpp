#include "resolution/experimental/holonomy_coordinates.h"

namespace Penner
{
	namespace Resolution
	{

		/**
		 * @brief Build minimal norm change of angle coordinate matrix
		 * 
		 * WARNING: Rank 5 matrix
		 * 
		 * @return Change of coordinate matrix.
		 */
		Eigen::Matrix<Scalar, 6, 6> build_pseduoinverse_ccw_flip_matrix()
		{
			Eigen::Matrix<Scalar, 6, 6> C;
			C.row(0) << 0., 0.25, -0.5, 0., 0.25, 0.5;
			C.row(1) << 0., 0., 1., 1., 0., 0.;
			C.row(2) << 0., -0.25,-0.5, 0., 0.75, 0.5;
			C.row(3) << 0., 0.25, 0.5, 0., 0.25, -0.5;
			C.row(4) << 1., 0., 0., 0., 0., 1.;
			C.row(5) << 0., 0.75, 0.5, 0., -0.25, -0.5;

			return C;
		}

		/**
		 * @brief Build change of angle coordinate matrix satisfying C^4 = I
		 * 
		 * WARNING: Observed to create unstable large angles.
		 * 
		 * @return Change of coordinate matrix.
		 */
		Eigen::Matrix<Scalar, 6, 6> build_algebraic_ccw_flip_matrix()
		{
			Eigen::Matrix<Scalar, 6, 6> C;
			C.row(0) << 0., 1., -1., 0., 0., 0.;
			C.row(1) << 0., 0., 1., 1., 0., 0.;
			C.row(2) << 0., -1., 0., 0., 1., 1.;
			C.row(3) << 0., 1., 0., 0., 0., -1.;
			C.row(4) << 1., 0., 0., 0., 0., 1.;
			C.row(5) << 0., 0., 1., 0., 0., 0.;

			return C;
		}

		void update_holonomy_coordinates_under_ccw_flip(
				const Mesh<Scalar> &m,
				VectorX &alpha,
				int halfedge_index)
		{
			// get facet halfedges
			int hij = halfedge_index;
			int hji = m.opp[hij];
			int hjk = m.n[hij];
			int hki = m.n[hjk];
			int hil = m.n[hji];
			int hlj = m.n[hil];

			// get holonomy angles before rotation
			Eigen::Vector<Scalar, 6> a;
			a[0] = alpha[hki];
			a[1] = alpha[hij];
			a[2] = alpha[hjk];
			a[3] = alpha[hlj];
			a[4] = alpha[hji];
			a[5] = alpha[hil];

			// get new angle coordinates
			Eigen::Matrix<Scalar, 6, 6> C = build_pseduoinverse_ccw_flip_matrix();
			//Eigen::Matrix<Scalar, 6, 6> C = build_algebraic_ccw_flip_matrix();
			Eigen::Vector<Scalar, 6> b = C * a;

			// rename halfedges for clarity
			int hkl = hji;
			int hlk = hij;

			// set new angles
			alpha[hil] = b[0];
			alpha[hlk] = b[1];
			alpha[hki] = b[2];
			alpha[hjk] = b[3];
			alpha[hkl] = b[4];
			alpha[hlj] = b[5];
		}

		void update_holonomy_coordinates_under_clockwise_flip(
				const Mesh<Scalar> &m,
				VectorX &alpha,
				int halfedge_index)
		{
			// get facet halfedges
			int hlk = halfedge_index;
			int hkl = m.opp[hlk];
			int hki = m.n[hlk];
			int hil = m.n[hki];
			int hlj = m.n[hkl];
			int hjk = m.n[hlj];

			// get holonomy angles before rotation
			Scalar bk_il = alpha[hil];
			Scalar bi_lk = alpha[hlk];
			Scalar bl_ki = alpha[hki];
			Scalar bl_jk = alpha[hjk];
			Scalar bj_kl = alpha[hkl];
			Scalar bk_lj = alpha[hlj];

			// rename halfedges for clarity
			int hij = hlk;
			int hji = hkl;

			// set new angles, arbitrarily setting hij to 0
			alpha[hki] = -bk_il + bl_jk + bj_kl - bk_lj;
			alpha[hij] = bk_il + bk_lj;
			alpha[hjk] = bk_lj;
			alpha[hlj] = bi_lk - bk_lj;
			alpha[hji] = bl_ki + bl_jk;
			alpha[hil] = bk_il - bl_jk + bk_lj;
		}

	} // namespace Resolution
} // namespace Penner