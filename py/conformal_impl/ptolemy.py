from conformal_impl.overload_math import *

def ptolemy_length_regular(ld,la,lb,lao,lbo):
    return (la/ld)*lbo + (lb/ld)*lao
# for the flip  11T-> TPQ returns new edge length and quad diagonal (diagonals are equal)

#[hd1,ha1,hb1,(hd2),ha2,hb2,hp2]
def ptolemy_length_11T(ld, la,lb, lp):
    return ( (la/ld)* ((la/ld)*lp + 2*lb), (la/ld)*lp + lb)

# for the flip  TPQ-> 11T returns new edge length (2 new edges are equal)
def ptolemy_length_TPQ(ld, la, lb, lp, lq): 
    return (la/ld)*(lb+lq); 

# for the flip  11Q -> QPQ   returns new edge length ld and diagonals of the new 
# quads lq1, lq1
def ptolemy_length_11Q(ld, la, lb, lp1, lp2, lq):
    return ( ( (la/ld)*la *(lp2/ld) + (lp1/ld)*(lb/ld)*lb + 2*(la/ld)*(lq/ld)*lb ), \
            (la/ld)*lq + lp1*(lb/ld), \
            lq*(lb/ld) + lp2*(la/ld) )

# for the flip  QPQ->11Q  returns new edge len and diagonal of the quad

def ptolemy_length_QPQ(ld,la,lb,lp1,lp2, lq1, lq2):
    return ( (la/ld)*lq2+(lb/ld)*lq1, (la/ld)*lb + lq1*(lq2/ld))

def check_tri_ineq(l1,l2,l3):
    return ( l1+l2 - l3 >=0  and  l2+l3 >= l1 and  l3+l1 >= l2) 

def euclidean_length(ld, la, lb, lao, lbo):
  s1 = (ld+la+lb)/2; s2 = (ld+lao+lbo)/2
  f1 = max((0.0), s1*(s1-ld)*(s1-la)*(s1-lb))
  f2 = max((0.0), s2*(s2-ld)*(s2-lao)*(s2-lbo))
  A1 = msqrt(f1); A2 = msqrt(f2)
  h1 = 2*A1/ld; h2 = 2*A2/ld
  if not(la >= h1 and lao >= h2):
    raise Exception("invalid euclidean flip")
  x1 = np.sign(ld**2+la**2-lb**2)  *msqrt(la**2 -h1**2); y1 = h1
  x2 = np.sign(ld**2+lao**2-lbo**2)*msqrt(lao**2-h2**2); y2 = -h2
  return msqrt((x1-x2)**2 + (y1-y2)**2)

LengthFun = {'111':ptolemy_length_regular, '1A2':ptolemy_length_regular, 'TPT':ptolemy_length_regular, '11T':ptolemy_length_11T, '11Tr':ptolemy_length_11T,'TPQ': ptolemy_length_TPQ, 'QPT':ptolemy_length_TPQ, '11Q':ptolemy_length_11Q, 'QPQ':ptolemy_length_QPQ}
