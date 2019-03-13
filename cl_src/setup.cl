__kernel void setup_points(__global float2* points){
  int id = get_global_id(0);
  float2 point = points[id];
  points[id] = (float2) (point.x + 1.0, point.y);
}
