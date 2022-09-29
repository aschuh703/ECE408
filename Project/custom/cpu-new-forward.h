#ifndef SRC_LAYER_CPU_NEW_FORWARD_H
#define SRC_LAYER_CPU_NEW_FORWARD_H

void conv_forward_cpu(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);

#endif // SRC_LAYER_CPU_NEW_FORWARD_H
