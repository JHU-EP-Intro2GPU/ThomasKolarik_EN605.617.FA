// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

#define SUB_BUFFER_SIZE     2

# Takes the average of the values within the subuffer and puts the result into the outputbuffer.
__kernel void average(__global * subBuffer, __global * outputBuffer)
{
	size_t id = get_global_id(0);
    
    float average = 0;
    
    for (int i = 0; i < SUB_BUFFER_SIZE; ++i)
    {
        average += subBuffer[i];
    }
    
    average = average / SUB_BUFFER_SIZE;
    
	buffer[id] = average;
}