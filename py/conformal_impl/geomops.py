
from conformal_impl.overload_math import *

# angle opposite l3 from 3 edge lengths, using atan2
def ang_from_len_atan(l1,l2,l3):  
    s = 0.5*(l1 +l2 + l3)
    return  matan2(  4.0*sqrt(s*(s-l1)*(s-l2)*(s-l3)), (l1**2 + l2**2 - l3**2))

# angle opposite l3 from 3 edge lengths, using atan2, alternative
def ang_from_len_atan_half(l1,l2,l3):  
    t31 = l1+l2-l3; t23 = l1-l2+l3; t12 = -l1+l2+l3; 
    l123 = l1+l2+l3; 
    denom = msqrt(t12*t23*t31*l123);
    return 2*matan2(t12*t23,denom);

ang_from_len = ang_from_len_atan_half

def area_from_len(l1,l2,l3):  # area from 3 edge lengths
    s = 0.5*(l1 +l2 + l3)
    return msqrt(s*(s-l1)*(s-l2)*(s-l3))

def cot_from_len(l1,l2,l3):  # cot of the angle opposite l3 from 3 edge length
    return (l1**2 + l2**2 - l3**2)/area_from_len(l1,l2,l3)/4.0

