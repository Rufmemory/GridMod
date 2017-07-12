import GridMod as GM

def test_grid_mod():
    viewer = GM.GridDisplay(1200, 860)
    viewer.add_grid('plane',  GM.shape_plane(0,0,0,10,10,10,10))
    viewer.add_grid('box', GM.shape_box(0,0,100,50,50,50))
    viewer.add_grid('sphere', GM.shape_ball(0,0,-100,100,100,100))
    viewer.run()
test_grid_mod()
