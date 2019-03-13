/*
  These kernel functions are what converts integers into binary in a step-by-step fashion.
  We run this algorithm for a different integer on each worker thread.
  We also include the function that resets the algorithm.
*/
__kernel void update_codes(__global uint *code)
{
  /*
    This advances the integer to binary algorithm by dividing the current integer by 2.
  */
  int gid = get_global_id(0);
  code[gid] = code[gid] >> 1;
}

__kernel void decode(__global uint *switches, __global uint *code)
{
  /*
    This is is the last step of each iteration in the integer to binary algorithm.
    This is the step that determines if we get a 1 or 0.
  */
  int gid = get_global_id(0);
  switches[gid] = code[gid] % 2;
}


__kernel void reset_codes(__global uint* codes, __global uint* master_codes){
  /*
    Resets the code array by loading in the master copy which is read only.
  */
  int id = get_global_id(0);
  codes[id] = master_codes[id];
}
