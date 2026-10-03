import numpy as np

from streetview_to_3d.models.segment import in_view, long_poles


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


def test_a_view_takes_its_own_part_of_the_pano_mask():
    # a spot on the horizon 30 deg right of the pano's middle
    pano = np.zeros((512, 1024), bool)
    pano[250:262, 594:614] = True
    assert in_view(pano, 30, 90.0, 160, 90)[45, 80]           # the view facing it: its centre
    assert not in_view(pano, 0, 90.0, 160, 90)[45, 80]
    x = int(round(79.5 + 80 * np.tan(np.radians(30))))       # the view at 0: 30 deg right
    assert in_view(pano, 0, 90.0, 160, 90)[45, x]
