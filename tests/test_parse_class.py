import os

def test_parse():
    file = open("testfile.txt","w") 
    file.write('hello =  world\n \n #derp = derp')
    if p.ploop("testfile.txt") == {'hello': 'world'}:
        pass
    os.remove("testfile.txt")