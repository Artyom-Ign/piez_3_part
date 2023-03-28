xmin = _xmin_;
ymin = _ymin_;
xmax = _xmax_;
ymax = _ymax_;
 
r1 = _r1_;
x1 = _x1_;
y1 = _y1_;

r2 = _r2_;
x2 = _x2_;
y2 = _y2_;

r3 = _r3_;
x3 = _x3_;
y3 = _y3_;

out = _lc_ext_;
in = _lc_circ_;

Point(1) = {xmin, ymin, 0, out};
Point(2) = {xmin, ymax, 0, out};
Point(3) = {xmax, ymax, 0, out};
Point(4) = {xmax, ymin, 0, out};

Point(15) = {x1, y1, 0, in};
Point(16) = {r1 + x1, y1, 0, in};
Point(17) = {x1, y1 + r1, 0, in};
Point(18) = {x1 -r1, y1, 0, in};
Point(19) = {x1, y1-r1, 0, in};

Point(25) = {x2, y2, 0, in};
Point(26) = {r2 + x2, y2, 0, in};
Point(27) = {x2, y2 + r2, 0, in};
Point(28) = {x2 - r2, y2, 0, in};
Point(29) = {x2, y2-r2, 0, in};

Point(35) = {x3, y3, 0, in};
Point(36) = {r3 + x3, y3, 0, in};
Point(37) = {x3, y3 + r3, 0, in};
Point(38) = {x3 - r3, y3, 0, in};
Point(39) = {x3, y3-r3, 0, in};

Line(1) = {1,4};
Line(2) = {4,3};
Line(3) = {3,2};
Line(4) = {2,1};
Circle(15) = {16,15,17};
Circle(16) = {17,15,18};
Circle(17) = {18,15,19};
Circle(18) = {19,15,16};

Circle(25) = {26,25,27};
Circle(26) = {27,25,28};
Circle(27) = {28,25,29};
Circle(28) = {29,25,26};

Circle(35) = {36,35,37};
Circle(36) = {37,35,38};
Circle(37) = {38,35,39};
Circle(38) = {39,35,36};

Line Loop(5) = {2,3,4,1};
Line Loop(10) = {15,16,17,18};
Line Loop(20) = {25,26,27,28};
Line Loop(30) = {35,36,37,38};

Plane Surface(0) = {5, 10, 20, 30};
Plane Surface(10) = {10};
Plane Surface(11) = {20};
Plane Surface(12) = {30};

Physical Surface(0) = {0};
Physical Surface(10) = {10};
Physical Surface(11) = {11};
Physical Surface(12) = {12};

