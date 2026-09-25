import taichi as ti

import air_sph


def test_taichi_initializes_on_cpu():
    ti.init(arch=ti.cpu)

    @ti.kernel
    def one() -> ti.i32:
        return 1

    assert one() == 1
