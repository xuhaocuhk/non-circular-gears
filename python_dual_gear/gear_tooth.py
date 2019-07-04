import math

def teeth_straight(x: float, height: float, width: float):
    assert 0 <= x <= 1
    if x < width/7*2:
        y = height * (x / (width/7*2))
    elif x < width/7*5:
        y = height
    elif x < width:
        y = height * (width - x) / (width/7*2)
    else:
        y = 0.0
    return y-height/2

def teeth_sine(x: float, height: float, width: float):
    assert 0 <= x <= 1
    if x < width:
        return height * math.sin(x * 2 * math.pi)
    else:
        return 0

def teeth_involute(x: float, height: float, width: float):
    assert 0 <= x <= 1
    assert 0 < width < 1
    fake_height = height / (-(2/3-1)**2 + 1)
    if x < width/3:
        y = fake_height*(-(2/width*x-1)**2 + 1)
    elif x < width/3*2:
        y = height
    elif x < width:
        y = fake_height*(-(2/width*x-1)**2 + 1)
    else:
        y = 0
    return y-height/2

def teeth_involute_sin(x: float, height: float, width: float):
    assert 0 <= x <= 1
    assert 0 < width < 1
    fake_height = height / math.sin(math.pi/3)
    if x < width/3:
        y = fake_height* math.sin(x/width * math.pi)
    elif x < width/3*2:
        y = height
    elif x < width:
        y = fake_height* math.sin(x/width * math.pi)
    else:
        y = 0
    return y-height/2