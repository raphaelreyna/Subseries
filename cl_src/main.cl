/*
  These kernel methods are for computing subseries.
  Each work thread decides whether or not to drop a term (not add it) based on the switch.
*/
float2 cadd(float2 a, float2 b){
  /*
    Used to compute complex addition of float2.
  */
    float res_real = a.x + b.x;
    float res_imag = a.y + b.y;
    return (float2) (res_real, res_imag);
}

float2 csub(float2 a, float2 b){
  float res_real = a.x - b.x;
  float res_imag = a.y - b.y;
  return (float2) (res_real, res_imag);
}

float2 cmultr(float2 a, float b){
  return (float2) (b*a.x, b*a.y);
}

float2 make_term(uint s, float re, float im){
  /*
    Returns either 0 or the term which was given by the host.
  */
    if (s == ((uint) 1)){
        return (float2) (re, im);
    }
    else {
        return (float2) (0.0, 0.0);
    }
}

__kernel void update_points(__global float2 *points, float term_real, float term_imag, __global uint *switches)
{
  /*
    Adds either 0 or the term sent by the host depending on the switch.
  */
    int gid = get_global_id(0);
    uint s = switches[gid];
    float2 old = points[gid];
    float2 term = make_term(s, term_real, term_imag);
    points[gid] = cadd(old, term);
}

__kernel void subtract_complex_number(__global float2 *points, float z_real, float z_imag)
{
  int gid = get_global_id(0);
  float2 z = (float2) (z_real, z_imag);
  points[gid] = csub(points[gid], z);
}
