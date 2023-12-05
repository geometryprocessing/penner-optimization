#pragma once

#include <memory>

#include "common.hh"
#include "embedding.hh"

namespace CurvatureMetric
{

	class DifferentiableConeMetric : public Mesh<Scalar>
	{
	public:
		DifferentiableConeMetric(const Mesh<Scalar> &m)
				: Mesh<Scalar>(m), m_is_discrete_metric(false), m_num_flips(0)
		{
			// Build edge maps
			build_edge_maps(m, he2e, e2he);
			m_identification = build_edge_matrix(he2e, e2he);
		}

		std::vector<int> he2e; // map from halfedge to edge
		std::vector<int> e2he; // map from edge to halfedge

		bool is_discrete_metric() const { return m_is_discrete_metric; };
		int num_flips() { return m_num_flips; };

		virtual VectorX get_metric_coordinates() const
		{
			int num_halfedges = n_halfedges();
			VectorX metric_coords(num_halfedges);
			for (int h = 0; h < num_halfedges; ++h)
			{
				metric_coords[h] = 2.0 * log(l[h]);
			}

			return metric_coords;
		}

		// TODO Make virtual
		VectorX get_reduced_metric_coordinates() const
		{
			return reduce_metric_coordinates(get_metric_coordinates());
		}

		MatrixX change_metric_to_reduced_coordinates(const std::vector<int> &I, const std::vector<int> &J, const std::vector<Scalar> &V, int num_rows) const
		{
			assert(I.size() == V.size());
			assert(J.size() == V.size());
			int num_entries = V.size();

			typedef Eigen::Triplet<Scalar> T;
			std::vector<T> tripletList;
			tripletList.reserve(num_entries);
			for (int k = 0; k < num_entries; ++k)
			{
				tripletList.push_back(T(I[k], J[k], V[k]));
			}

			// Build Jacobian from triplets
			int num_halfedges = n_halfedges();
			MatrixX halfedge_jacobian(num_rows, num_halfedges);
			halfedge_jacobian.reserve(num_entries);
			halfedge_jacobian.setFromTriplets(tripletList.begin(), tripletList.end());

			// Get transition Jacobian
			MatrixX J_transition = get_transition_jacobian();

			return halfedge_jacobian * J_transition;
		}

		MatrixX change_metric_to_reduced_coordinates(const MatrixX &halfedge_jacobian) const
		{
			MatrixX J_transition = get_transition_jacobian();
			return halfedge_jacobian * J_transition;
		}

		virtual ~DifferentiableConeMetric() = default;

		virtual bool flip_ccw(int _h, bool Ptolemy = true)
		{
			m_num_flips++;
			return Mesh<Scalar>::flip_ccw(_h, Ptolemy);
		}

		virtual std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const = 0;
		virtual std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(const VectorX &metric_coords) const = 0;
		virtual std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const = 0;

		// TODO Make private
		virtual VectorX reduce_metric_coordinates(const VectorX& metric_coords) const = 0;

		virtual void make_discrete_metric() = 0;
		virtual MatrixX get_transition_jacobian() const = 0;

	protected:
		bool m_is_discrete_metric;
		int m_num_flips;
		MatrixX m_identification;
	};

	class PennerConeMetric : public DifferentiableConeMetric
	{
	public:
		PennerConeMetric(const Mesh<Scalar> &m, const VectorX &metric_coords)
				: DifferentiableConeMetric(m)
		{
			build_refl_proj(m, he2e, e2he, m_proj, m_embed);
			build_refl_matrix(m_proj, m_embed, m_projection);
			expand_metric_coordinates(metric_coords);

			// Initialize jacobian to the identity
			int num_edges = e2he.size();
			m_transition_jacobian_lol = std::vector<std::map<int, Scalar>>(num_edges,
																																		 std::map<int, Scalar>());
			for (int e = 0; e < num_edges; ++e)
			{
				m_transition_jacobian_lol[e][e] = 1.0;
			}
		}

		VectorX reduce_metric_coordinates(const VectorX& metric_coords) const override
		{
			int num_reduced_coordinates = m_embed.size();
			VectorX reduced_metric_coords(num_reduced_coordinates);
			for (int E = 0; E < num_reduced_coordinates; ++E) {
				int h = e2he[m_embed[E]];
				reduced_metric_coords[E] = metric_coords[h];
			}

			return reduced_metric_coords;
		}

		std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const override
		{
			return std::make_unique<PennerConeMetric>(PennerConeMetric(*this));
		}

		std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(const VectorX &metric_coords) const override
		{
			return std::make_unique<PennerConeMetric>(PennerConeMetric(*this, metric_coords));
		}

		std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const override
		{
			int num_reduced_coordinates = m_embed.size();
			VectorX reduced_metric_coords(num_reduced_coordinates);
			for (int E = 0; E < num_reduced_coordinates; ++E) {
				int h = e2he[m_embed[E]];
				reduced_metric_coords[E] =
					2.0 * log(l[h]) + (u[v_rep[to[h]]] + u[v_rep[to[opp[h]]]]);
			}

			return set_metric_coordinates(reduced_metric_coords);
		}

		MatrixX get_transition_jacobian() const override
		{

			std::vector<T> tripletList;
			int num_edges = e2he.size();
			tripletList.reserve(5 * num_edges);
			for (int i = 0; i < num_edges; ++i)
			{
				for (auto it : m_transition_jacobian_lol[i])
				{
					tripletList.push_back(T(i, it.first, it.second));
				}
			}

			// Create the matrix from the triplets
			MatrixX transition_jacobian;
			transition_jacobian.resize(num_edges, num_edges);
			transition_jacobian.reserve(tripletList.size());
			transition_jacobian.setFromTriplets(tripletList.begin(), tripletList.end());

			return m_identification * (transition_jacobian * m_projection);
		}

		void make_discrete_metric() override
		{
			// Make the copied mesh Delaunay with Ptolemy flips
			VectorX u;
			u.setZero(n_ind_vertices());
			DelaunayStats del_stats;
			SolveStats<Scalar> solve_stats;
			bool use_ptolemy = true;
			ConformalIdealDelaunay<Scalar>::MakeDelaunay(*this, u, del_stats, solve_stats, use_ptolemy);
			m_flip_seq = del_stats.flip_seq; //TODO Append
			int num_flips = m_flip_seq.size();
			assert(num_flips == m_num_flips);
			m_is_discrete_metric = true;
			return;
		}

		bool flip_ccw(int _h, bool Ptolemy = true) override
		{
			Scalar zero_threshold = 1e-15;

			// Perform the flip in the base class
			bool success = DifferentiableConeMetric::flip_ccw(_h, Ptolemy);

			// Get local mesh information near hd
			int hd = _h;
			int hb = n[hd];
			int ha = n[hb];
			int hdo = opp[hd];
			int hao = n[hdo];
			int hbo = n[hao];

			// Get edges corresponding to halfedges
			int ed = he2e[hd];
			int eb = he2e[hb];
			int ea = he2e[ha];
			int eao = he2e[hao];
			int ebo = he2e[hbo];

			// Compute the shear for the edge ed
			// TODO Check if edge or halfedge coordinate
			Scalar la = l[ha];
			Scalar lb = l[hb];
			Scalar lao = l[hao];
			Scalar lbo = l[hbo];
			Scalar x = (la * lbo) / (lb * lao);

			// The matrix Pd corresponding to flipping edge ed is the identity except for
			// the row corresponding to edge ed, which has entries defined by Pd_scalars
			// in the column corresponding to the edge with the same index in Pd_edges
			std::vector<int> Pd_edges = {ed, ea, ebo, eao, eb};
			std::vector<Scalar> Pd_scalars = {
					-1.0, x / (1.0 + x), x / (1.0 + x), 1.0 / (1.0 + x), 1.0 / (1.0 + x)};

			// Compute the new row of J_del corresponding to edge ed, which is the only
			// edge that changes
			std::map<int, Scalar> J_del_d_new;
			for (int i = 0; i < 5; ++i)
			{
				int ei = Pd_edges[i];
				Scalar Di = Pd_scalars[i];
				for (auto it : m_transition_jacobian_lol[ei])
				{
					J_del_d_new[it.first] += Di * it.second;

					// Delete the updated entry if it is near 0
					if (abs(J_del_d_new[it.first]) < zero_threshold)
						J_del_d_new.erase(it.first);
				}
			}

			m_transition_jacobian_lol[ed] = J_del_d_new;

			return success;
		}

	private:
		std::vector<int> m_embed;
		std::vector<int> m_proj;
		MatrixX m_projection;
		std::vector<std::map<int, Scalar>> m_transition_jacobian_lol;
		std::vector<int> m_flip_seq;

		void expand_metric_coordinates(const VectorX &metric_coords)
		{
			int num_embedded_edges = m_embed.size();
			int num_edges = e2he.size();
			int num_halfedges = he2e.size();

			// Embedded edge lengths provided
			if (metric_coords.size() == num_embedded_edges)
			{
				for (int h = 0; h < num_halfedges; ++h)
				{
					l[h] = exp(metric_coords[m_proj[he2e[h]]] / 2.0);
				}
			}
			else if (metric_coords.size() == num_edges)
			{
				for (int h = 0; h < num_halfedges; ++h)
				{
					l[h] = exp(metric_coords[he2e[h]] / 2.0);
				}
			}
			else if (metric_coords.size() == num_halfedges)
			{
				for (int h = 0; h < num_halfedges; ++h)
				{
					l[h] = exp(metric_coords[h] / 2.0);
				}
			}
			// TODO error
		}
	};

	class DiscreteMetric : public DifferentiableConeMetric
	{
	public:
		DiscreteMetric(const Mesh<Scalar> &m, const VectorX &log_length_coords)
				: DifferentiableConeMetric(m)
		{
			m_is_discrete_metric = true;
			build_refl_proj(m, he2e, e2he, m_proj, m_embed);
			build_refl_matrix(m_proj, m_embed, m_projection);
			expand_metric_coordinates(log_length_coords);
		}

		VectorX reduce_metric_coordinates(const VectorX& metric_coords) const override
		{
			int num_reduced_coordinates = m_embed.size();
			VectorX reduced_metric_coords(num_reduced_coordinates);
			for (int E = 0; E < num_reduced_coordinates; ++E) {
				int h = e2he[m_embed[E]];
				reduced_metric_coords[E] = metric_coords[h];
			}

			return reduced_metric_coords;
		}

		std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const override
		{
			return std::make_unique<DiscreteMetric>(DiscreteMetric(*this));
		}

		std::unique_ptr<DifferentiableConeMetric> set_metric_coordinates(const VectorX &metric_coords) const override
		{
			return std::make_unique<DiscreteMetric>(DiscreteMetric(*this, metric_coords));
		}

		std::unique_ptr<DifferentiableConeMetric> scale_conformally(const VectorX& u) const override
		{
			int num_reduced_coordinates = m_embed.size();
			VectorX reduced_metric_coords(num_reduced_coordinates);
			for (int E = 0; E < num_reduced_coordinates; ++E) {
				int h = e2he[m_embed[E]];
				reduced_metric_coords[E] =
					2.0 * log(l[h]) + (u[v_rep[to[h]]] + u[v_rep[to[opp[h]]]]);
			}

			return set_metric_coordinates(reduced_metric_coords);
		}


		MatrixX get_transition_jacobian() const override
		{
			return m_identification * m_projection;
		}

		void make_discrete_metric() override
		{
			// Always true
			return;
		}

		bool flip_ccw(int _h, bool Ptolemy = true) override
		{
			// Perform the flip in the base class
			// TODO Add warning and validity check
			return DifferentiableConeMetric::flip_ccw(_h, Ptolemy);
		}

	private:
		std::vector<int> m_embed;
		std::vector<int> m_proj;
		MatrixX m_projection;

		void expand_metric_coordinates(const VectorX &metric_coords)
		{
			int num_embedded_edges = m_embed.size();
			int num_edges = e2he.size();
			int num_halfedges = he2e.size();

			// Embedded edge lengths provided
			if (metric_coords.size() == num_embedded_edges)
			{
				for (int h = 0; h < num_halfedges; ++h)
				{
					l[h] = exp(metric_coords[m_proj[he2e[h]]] / 2.0);
				}
			}
			else if (metric_coords.size() == num_edges)
			{
				for (int h = 0; h < num_halfedges; ++h)
				{
					l[h] = exp(metric_coords[he2e[h]] / 2.0);
				}
			}
			else if (metric_coords.size() == num_halfedges)
			{
				for (int h = 0; h < num_halfedges; ++h)
				{
					l[h] = exp(metric_coords[h] / 2.0);
				}
			}
			// TODO error
		}
	};

	/* Reminders of code to add

			/// Constrain descent direction

			// If using regular length values, apply the change of coordinates from the
			// log values
			if (!opt_params.use_log)
			{
				MatrixX J_length;
				length_jacobian(reduced_metric_coords, J_length);
				J_constraint = J_constraint * J_length;
			}

			// Remove the fixed degrees of freedom
			VectorX variable_descent_direction, variable_constraint;
			MatrixX restricted_J_constraint = J_constraint * reduction_maps.projection;
			MatrixX variable_J_constraint;
			compute_subset(
					descent_direction, reduction_maps.free_e, variable_descent_direction);
			compute_subset(constraint, reduction_maps.free_v, variable_constraint);
			compute_submatrix(restricted_J_constraint,
												reduction_maps.free_v,
												reduction_maps.free_e,
												variable_J_constraint);

			// Project the descent direction to the constraint tangent plane
			VectorX projected_variable_descent_direction = project_descent_direction(
					variable_descent_direction, variable_constraint, variable_J_constraint);

			// Expand projected descent direction on the free edges to the full edge list
			projected_descent_direction.setZero(num_reduced_edges);
			write_subset(projected_variable_descent_direction,
									 reduction_maps.free_e,
									 projected_descent_direction);

			/// Optimal descent direction

			// Restrict the constraint Jacobian to the variables
			MatrixX reduced_J_constraint = J_constraint * R;
			MatrixX variable_J_constraint;
			compute_submatrix(reduced_J_constraint,
												reduction_maps.free_v,
												reduction_maps.free_e,
												variable_J_constraint);

			// Get projection matrix
			const MatrixX &R = reduction_maps.projection;

			// Expand the metric coordinates to the doubled surface
			VectorX metric_coords;
			expand_reduced_function(
					reduction_maps.proj, reduced_metric_coords, metric_coords);

			// Remove the fixed degrees of freedom from the gradient and descent direction
			VectorX variable_gradient;
			compute_subset(gradient, reduction_maps.free_e, variable_gradient);
			VectorX variable_descent_direction;
			compute_subset(descent_direction, reduction_maps.free_e, variable_descent_direction);

			// Compute hessian with fixed degrees of freedom removed
			MatrixX hessian = opt_energy.hessian();
			MatrixX reduced_hessian = R.transpose() * (hessian * R);
			MatrixX variable_hessian;
			compute_submatrix(reduced_hessian,
												reduction_maps.free_e,
												reduction_maps.free_e,
												variable_hessian);

				/// Line Step

				// Make a line step in length space
				else
				{
					size_t num_reduced_edges = reduction_maps.num_reduced_edges;
					reduced_line_step_metric_coords.resize(num_reduced_edges);

					// For each edge, go to length space, take a step, and then return to log space
					for (size_t Ei = 0; Ei < num_reduced_edges; ++Ei)
					{
						Scalar initial_length_coord =
								exp(initial_reduced_metric_coords[Ei] / 2.0);
						Scalar length_coord =
								initial_length_coord + beta * descent_direction[Ei];
						reduced_line_step_metric_coords[Ei] = 2 * log(length_coord);
					}
				}
	*/

} // namespace CurvatureMetric
