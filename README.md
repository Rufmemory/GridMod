# GridMod
Hello dear folks! Looking for a fully modular, open source, Pygame 3d-Engine? 
Well this may be a good start. 
GridMod is initialized by classes and methods, 
and is capable of displaying static custom three dimensional shapes.

By groupping nodes, vectors and matrices, and by applying common matrix operations 
to those vertices we get to display predefined 3d objects 
along with a scaling, rotation, and colour factor, 
and groupping them all together into a clustered scene.

Feel free to let me know if you want to add or ask anything, or participate.
And since it is be the main 3d-engine concept im working on at the moment, 
I will soon try to implement kinetics and motion for further use...

In order to test this current build you can try this basic example 
where a ball box and plane are alligned together in space.

def test_grid_mod():
    viewer = GridDisplay(1200, 860)
    viewer.add_grid('plane',  shape_plane(0,0,0,10,10,10,10))
    viewer.add_grid('box', shape_box(0,0,100,50,50,50))
    viewer.add_grid('sphere', shape_ball(0,0,-100,100,100,100))
    viewer.run()
test_grid_mod()

You are definetly also welcome to tell me if you simply liked the project. 
Ruf.
