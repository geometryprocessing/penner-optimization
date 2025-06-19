
struct ParametrizedMesh
{
    ParametrizedMesh(
        const Eigen::MatrixXd& V_,
        const Eigen::MatrixXi& F_,
        const Eigen::MatrixXd& uv_,
        const Eigen::MatrixXi& F_uv_)
        : V(V_)
        , F(F_)
        , uv(uv_)
        , F_uv(F_uv_)
    {}

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi> unpack()
    {
        return std::make_tuple(V, F, uv, F_uv);
    }


    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd uv;
    Eigen::MatrixXi F_uv;
};

struct ParametrizedOverlayMesh
{
    ParametrizedOverlayMesh(
        const Eigen::MatrixXd& V_,
        const Eigen::MatrixXi& F_,
        const Eigen::MatrixXd& uv_,
        const Eigen::MatrixXi& F_uv_,
        const std::vector<int>& Fn_to_F_,
        const std::vector<std::pair<int, int>>& endpoints_)
        : V(V_)
        , F(F_)
        , uv(uv_)
        , F_uv(F_uv_)
        , Fn_to_F(Fn_to_F_)
        , endpoints(endpoints_)
    {}

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd uv;
    Eigen::MatrixXi F_uv;
    std::vector<int> Fn_to_F;
    std::vector<std::pair<int, int>> endpoints;
};
