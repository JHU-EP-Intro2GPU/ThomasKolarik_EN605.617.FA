// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

// Takes the average of the values within the subuffer and puts the result into the outputbuffer.
__kernel void average(constant __global float * subBuffer, __global float * outputBuffer, const int size)
{
	size_t id = get_global_id(0);
    
    float average = 0;
    
    for (int i = 0; i < size; ++i)
    {
        average += subBuffer[i];
    }
    
    average = average / size;
    
	buffer[id] = average;
}