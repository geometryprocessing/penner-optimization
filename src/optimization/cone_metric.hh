#pragma once

#include <memory>

#include "common.hh"
#include "embedding.hh"

namespace CurvatureMetric
{

	class DifferentiableConeMetric : public Mesh<Scalar>
	{
	public:
		DifferentiableConeMetric(const Mesh<Scalar> &m, VectorX &metric_coords)
				: Mesh<Scalar>(m)
		{
			// Build edge maps
			build_edge_maps(m, m_he2e, m_e2he);
			build_refl_proj(m, m_he2e, m_e2he, m_proj, m_embed);

			// Copy log metric coords to mesh (for stable rational flips)
			set_metric_coordinates(metric_coords);
		}

		void set_metric_coordinates(const VectorX &metric_coords)
		{
			int num_embedded_edges = m_embed.size();
			int num_edges = m_e2he.size();
			int num_halfedges = m_he2e.size();

			// Embedded edge lengths provided
			if (metric_coords.size() == num_embedded_edges)
			{
				for (int h = 0; h < num_halfedges; ++h)
				{
					l[h] = exp(metric_coords[m_proj[m_he2e[h]]] / 2.0);
				}
			}
			if (metric_coords.size() == num_edges)
			{
				for (int h = 0; h < num_halfedges; ++h)
				{
					l[h] = exp(metric_coords[m_he2e[h]] / 2.0);
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

		VectorX get_edge_metric_coordinates() const
		{
			int num_edges = n_edges();
			VectorX metric_coords(num_edges);
			for (int e = 0; e < num_edges; ++e)
			{
				metric_coords[e] = 2.0 * log(l[m_e2he[e]]);
			}

			return metric_coords;
		}

		VectorX get_halfedge_metric_coordinates() const
		{
			int num_halfedges = n_halfedges();
			VectorX halfedge_metric_coords(num_halfedges);
			for (int h = 0; h < num_halfedges; ++h)
			{
				halfedge_metric_coords[h] = 2.0 * log(l[h]);
			}

			return halfedge_metric_coords;
		}

		virtual ~DifferentiableConeMetric() = default;

		virtual std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const = 0;
		virtual void make_discrete_metric() = 0;
		virtual MatrixX get_transition_jacobian() const = 0;
		virtual bool flip_ccw(int _h, bool Ptolemy = true)
		{
			return Mesh<Scalar>::flip_ccw(_h, Ptolemy);
		}

	protected:
		std::vector<int> m_he2e; // map from halfedge to edge
		std::vector<int> m_e2he; // map from edge to halfedge
		std::vector<int> m_proj;
		std::vector<int> m_embed;
	};

	class PennerConeMetric : public DifferentiableConeMetric
	{
	public:
		PennerConeMetric(const Mesh<Scalar> &m, VectorX &metric_coords)
				: DifferentiableConeMetric(m, metric_coords)
		{
			// Initialize jacobian to the identity
			int num_edges = m_e2he.size();
			m_transition_jacobian_lol = std::vector<std::map<int, Scalar>>(num_edges,
																																		 std::map<int, Scalar>());
			for (int e = 0; e < num_edges; ++e)
			{
				m_transition_jacobian_lol[e][e] = 1.0;
			}
		}

		std::unique_ptr<DifferentiableConeMetric> clone_cone_metric() const
		{
			return std::make_unique<PennerConeMetric>(*this);
		}

		MatrixX get_transition_jacobian() const
		{

			std::vector<T> tripletList;
			int num_edges = m_e2he.size();
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

			return transition_jacobian;
		}

		void make_discrete_metric()
		{
			// Make the copied mesh Delaunay with Ptolemy flips
			VectorX u;
			u.setZero(n_ind_vertices());
			DelaunayStats del_stats;
			SolveStats<Scalar> solve_stats;
			bool use_ptolemy = true;
			ConformalIdealDelaunay<Scalar>::MakeDelaunay(*this, u, del_stats, solve_stats, use_ptolemy);
			m_flip_seq = del_stats.flip_seq;
			return;
		}

		bool flip_ccw(int _h, bool Ptolemy = true)
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
			int ed = m_he2e[hd];
			int eb = m_he2e[hb];
			int ea = m_he2e[ha];
			int eao = m_he2e[hao];
			int ebo = m_he2e[hbo];

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
		std::vector<std::map<int, Scalar>> m_transition_jacobian_lol;
		std::vector<int> m_flip_seq;
	};

} // namespace CurvatureMetric
