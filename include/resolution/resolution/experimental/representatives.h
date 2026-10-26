#pragma once
#include "resolution/core/common.h"

namespace Penner {
namespace Resolution {

std::vector<std::vector<int>> generate_radial_surface(int genus);
std::vector<std::vector<int>> generate_representative_surface(int genus);

}
}