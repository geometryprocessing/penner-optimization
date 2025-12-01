#pragma once

#include "feature/core/common.h"

namespace Penner {
namespace Feature {

template <typename OverlayScalar>
class LayoutOptimizer {
private:
    int num_halfedges;
    int num_vars;
    Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1> x;
    Eigen::SparseMatrix<OverlayScalar> D;
    Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1> d;
    Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1> l0;

    Eigen::SparseMatrix<OverlayScalar> H;
    Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1> g;

    Eigen::SparseMatrix<OverlayScalar> Hr;
    Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1> gr;

    typedef Eigen::Triplet<OverlayScalar> ScalarTrip;

    int index(int hij, int coord) const { return (2 * hij) + coord; }
    int f2h(int fijk, int i) const { return (3 * fijk) + i; }

    OverlayScalar pow(OverlayScalar x, int n) const
    {
        if (n < 1) return 1.;
        return x * pow(x, n - 1);
    }

    void build_difference_matrix(const Mesh<OverlayScalar>& m)
    {
        typedef Eigen::Triplet<OverlayScalar> ScalarTrip;
        std::vector<ScalarTrip> system_trips;
        system_trips.reserve(4 * num_halfedges);

        // set rows of matrix as halfedge difference vectors
        for (int hij = 0; hij < num_halfedges; ++hij) {
            int hki = m.n[m.n[hij]];
            for (int coord = 0; coord < 2; ++coord)
            {
                system_trips.push_back(ScalarTrip(index(hij, coord), index(hij, coord), 1.));
                system_trips.push_back(ScalarTrip(index(hij, coord), index(hki, coord), -1.));
            }
        }

        // build matrix
        D.setFromTriplets(system_trips.begin(), system_trips.end());
    }

    void build_difference_matrix(const Eigen::MatrixXi& F)
    {
        spdlog::info("building basis matrix");
        typedef Eigen::Triplet<OverlayScalar> ScalarTrip;
        std::vector<ScalarTrip> system_trips;
        system_trips.reserve(4 * num_halfedges);

        // set rows of matrix as halfedge difference vectors
        int num_faces = F.rows();
        for (int fijk = 0; fijk < num_faces; ++fijk)
        {
            for (int i = 0; i < 3; ++i)
            {
                int j = (i + 1) % 3;
                int k = (j + 1) % 3;
                for (int coord = 0; coord < 2; ++coord)
                {
                    int hij = f2h(fijk, k);
                    system_trips.push_back(ScalarTrip(index(hij, coord), index(F(fijk, j), coord), 1.));
                    system_trips.push_back(ScalarTrip(index(hij, coord), index(F(fijk, i), coord), -1.));
                }
            }
        }

        // build matrix
        D.setFromTriplets(system_trips.begin(), system_trips.end());
    }

    void initialize(
        const Mesh<OverlayScalar>& m,
        const std::vector<OverlayScalar>& u,
        const std::vector<OverlayScalar>& v)
    {
        num_halfedges = m.n_halfedges();
        num_vars = 2 * num_halfedges;
        g.resize(2 * num_halfedges);
        x.resize(2 * num_halfedges);
        l0.resize(num_halfedges);
        D.resize(2 * num_halfedges, 2 * num_halfedges);
        H.resize(2 * num_halfedges, 2 * num_halfedges);

        for (int hij = 0; hij < num_halfedges; ++hij) {
            x[index(hij, 0)] = u[hij];
            x[index(hij, 1)] = v[hij];
            l0[hij] = pow(m.l[hij], 2);
        }

        build_difference_matrix(m);
    }

    void initialize(
        const Eigen::MatrixXd& uv,
        const Eigen::MatrixXi& FT,
        const Eigen::MatrixXi& F,
        std::vector<OverlayScalar>& u,
        std::vector<OverlayScalar>& v)
    {
        int num_faces = F.rows();
        num_halfedges = 3 * num_faces;
        num_vars = u.size();
        g.resize(2 * num_halfedges);
        D.resize(2 * num_halfedges, 2 * num_vars);
        H.resize(2 * num_halfedges, 2 * num_halfedges);

        spdlog::info("initializing variables");
        x.resize(2 * num_vars);
        for (int i = 0; i < num_vars; ++i)
        {
            x[index(i, 0)] = u[i];
            x[index(i, 1)] = v[i];
        }

        spdlog::info("computing target lengths");
        l0.resize(num_halfedges);
        for (int fijk = 0; fijk < num_faces; ++fijk)
        {
            for (int i = 0; i < 3; ++i)
            {
                int j = (i + 1) % 3;
                int k = (j + 1) % 3;
                Eigen::VectorXd dij = uv.row(FT(fijk, j)) - uv.row(FT(fijk, i));
                l0[f2h(fijk, k)] = dij.dot(dij);
            }
        }

        build_difference_matrix(F);
    }

    OverlayScalar compute_length_difference(int hij) const
    {
        int uij = index(hij, 0);
        int vij = index(hij, 1);
        OverlayScalar lij = pow(x[uij], 2) + pow(x[vij], 2);
        return lij - l0(hij);
    }

    void build_quadratic_approximation()
    {
        // get halfedge direction vectors
        d = D * x;

        std::vector<ScalarTrip> system_trips;
        system_trips.reserve(4 * num_halfedges);

        // set rows of matrix as halfedge difference vectors
        for (int hij = 0; hij < num_halfedges; ++hij) {
            int uij = index(hij, 0);
            int vij = index(hij, 1);
            OverlayScalar lij = pow(d[uij], 2) + pow(d[vij], 2);
            OverlayScalar dij = lij - l0[hij];
            OverlayScalar dpij = pow(dij, p - 2);
            OverlayScalar a;

            // compute gradient
            g[uij] = d[uij] * dij * dpij;
            g[vij] = d[vij] * dij * dpij;

            // compute u diagonal
            a = (p * pow(d[uij], 2) + pow(d[vij], 2)) * dpij;
            system_trips.push_back(ScalarTrip(uij, uij, a));

            // compute v diagonal
            a = (pow(d[uij], 2) + p * pow(d[vij], 2)) * dpij;
            system_trips.push_back(ScalarTrip(vij, vij, a));

            // compute cross terms
            a = (p - 1) * d[uij] * d[vij] * dpij;
            system_trips.push_back(ScalarTrip(uij, vij, a));
            system_trips.push_back(ScalarTrip(vij, uij, a));
        }

        // build matrix
        H.setFromTriplets(system_trips.begin(), system_trips.end());
    }

    OverlayScalar compute_energy()
    {
        spdlog::info("computing energy");

        // get halfedge direction vectors
        d = D * x;

        OverlayScalar energy(0.);
        for (int hij = 0; hij < num_halfedges; ++hij) {
            int uij = index(hij, 0);
            int vij = index(hij, 1);
            OverlayScalar lij = pow(d[uij], 2) + pow(d[vij], 2);
            OverlayScalar dij = lij - l0[hij];
            energy += pow(dij, p);
        }

        return (1. / (2. * p)) * energy;
    }

    Eigen::Matrix<OverlayScalar, Eigen::Dynamic, 1> compute_descent_direction()
    {
        // build difference coordinate quadratic system
        build_quadratic_approximation();

        // build reduced variable quadratic system
        Hr = D.transpose() * H * D;
        gr = D.transpose() * g;
        bool use_gradient = false;
        if (use_gradient) return -gr;

        // initialize solver
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<OverlayScalar>> solver;
        solver.compute(Hr);

        return -solver.solve(gr);
    }
    
    void do_line_step()
    {
        auto y = compute_descent_direction();
        OverlayScalar beta = 1.;
        OverlayScalar E0 = compute_energy();
        spdlog::info("initial error {}", E0);

        auto x0 = x;
        x = x0 + beta * y;
        OverlayScalar E = compute_energy();
        spdlog::info("error with line step {} is {}",beta, E);
        while (E > E0)
        {
            beta = beta / 2.;
            x = x0 + beta * y;
            E = compute_energy();
            spdlog::info("error with line step {} is {}",beta, E);
        }
    }

    std::vector<OverlayScalar> get_coordinates(int coord) const
    {
        std::vector<OverlayScalar> coords(num_vars);
        for (int hij = 0; hij < num_vars; ++hij) {
            coords[hij] = x[index(hij, coord)];
        }

        return coords;
    }

public:
    int p;
    LayoutOptimizer() 
    : p(2)
    {}

    void run(
        const Mesh<OverlayScalar>& m,
        std::vector<OverlayScalar>& u,
        std::vector<OverlayScalar>& v)
    {
        initialize(m, u, v);
        OverlayScalar energy = compute_energy();
        spdlog::info("initial error is {}", energy);

        for (int i = 0; i < 1000; ++i)
        {
            do_line_step();
        }

        energy = compute_energy();
        spdlog::info("final error is {}", energy);

        u = get_coordinates(0);
        v = get_coordinates(1);
    }

    void run(
        const Eigen::MatrixXd& uv,
        const Eigen::MatrixXi& FT,
        const Eigen::MatrixXi& F,
        std::vector<OverlayScalar>& u,
        std::vector<OverlayScalar>& v)
        {
            initialize(uv, FT, F, u, v);

            OverlayScalar energy = compute_energy();
            spdlog::info("initial error is {}", energy);

            for (int i = 0; i < 5; ++i)
            {
                do_line_step();
            }

            energy = compute_energy();
            spdlog::info("final error is {}", energy);

            u = get_coordinates(0);
            v = get_coordinates(1);
        }
};

} // namespace Feature
} // namespace Penner