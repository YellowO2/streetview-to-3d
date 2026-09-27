import numpy as np

from streetview_to_3d.services.segment import agree, long_poles


def _canvas():
    return np.zeros((200, 300), bool)


def test_long_straight_pole_dropped_even_with_a_lamp_arm():
    m = _canvas()
    m[20:190, 50:56] = True          # the post
    m[20:26, 50:90] = True           # its arm
    assert long_poles(m)[100, 52] and long_poles(m)[22, 80]


def test_leaning_straight_pole_dropped():
    m = _canvas()
    for y in range(20, 190):
        x = 100 + (y - 20) // 4
        m[y, x:x + 5] = True
    assert long_poles(m)[100, 100 + 80 // 4 + 2]


def test_bollard_pillar_and_curved_pole_kept():
    m = _canvas()
    m[150:190, 20:35] = True         # bollard: short
    m[20:190, 60:120] = True         # pillar: thick
    for y in range(20, 190):         # curved: an arc
        x = int(200 + 60 * np.sin((y - 20) / 170 * np.pi))
        m[y, x:x + 5] = True
    assert not long_poles(m).any()


def test_agree_keeps_a_drop_only_where_every_view_seeing_it_drops_it():
    # views at yaw 0 and 330, 90 deg wide: 0's centre is 30 deg right in
    # 330's view; 0's right edge (40 deg) is outside 330's view
    h, w = 90, 160
    f = w / 2
    a = np.zeros((h, w), bool)
    a[40:50, 75:85] = True                        # 0's centre
    a[40:50, 145:155] = True                      # 0's right edge
    names = ["pano_0_da3_0_0.jpg", "pano_0_da3_330_0.jpg"]
    out = agree([a, np.zeros((h, w), bool)], names, 90.0)
    assert not out[0][45, 80]                     # 330 sees it and says keep
    assert out[0][45, 150]                        # nobody else sees it
    b = np.zeros((h, w), bool)
    x = int(round((w - 1) / 2 + f * np.tan(np.radians(30))))
    b[38:52, x - 7:x + 7] = True                  # 330 drops the same spot
    assert agree([a, b], names, 90.0)[0][45, 80]
