#include<iostream>
#include"unet.hpp"


using namespace std;
//线性整流函数
float relu(float x) {
    return (x < 0) ? 0 : x;
}

float ***make_3darray(int channels,int dim)
{
	int dim1=channels, dim2=dim, dim3=dim;
	int i,j;
	float *** array = (float ***)malloc(dim1*sizeof(float**));

	for (i = 0; i< dim1; i++)
	{
		array[i] = (float **) malloc(dim2*sizeof(float *));
		for (j = 0; j < dim2; j++)
			array[i][j] = (float *)malloc(dim3*sizeof(float));
	}
	return array;
}


//////////////////////// CONVOLUTIONS //////////////////////////

void conv(struct conv_data_ *ptr_conv_data)
{
	//f_dim = 3, stride = 1.
	float ***conv_in,***conv_out;
	float ****filter,*bias;
	int dim,f_num,ch_num,o_dim, mode;
	//Unpacking data
	conv_in = ptr_conv_data->conv_in;  //输入的图片信息 RGB三通道
	filter = ptr_conv_data->filter;   //weight
	bias = ptr_conv_data->bias;   //神经元的bia
	dim = ptr_conv_data->dim;
	mode = ptr_conv_data->mode;  //padding mode
	ch_num = ptr_conv_data->ch_num;//input channel nunmber
	f_num = ptr_conv_data->f_num; //filter number
	int s=1;                      //stride
	int f = ptr_conv_data->f_dim; // transp convulution :f=2 , convolution : f=3

	
	int pad = mode;
	o_dim = dim;
	int dim_t = dim + 2*pad;
	float ***conv_in_t = make_3darray(ch_num, dim_t);
	conv_out = make_3darray(f_num, o_dim); //number of filters will determine the number of out image channels, dim will be the same in this case.
	
	//zero padding
	for(int i=0; i< ch_num; i++)
	{
		for(int x = 0; x<pad ; x++)
		{
			for(int y = 0; y< dim_t; y++)
			{
				//Zero-padding
				conv_in_t[i][x][y] = 0;
				conv_in_t[i][y][x] = 0;
				conv_in_t[i][(dim_t-1)-x][y] = 0;
				conv_in_t[i][y][(dim_t-1)-x] = 0;
			}
		}

		//fill the empty center space with conv_in--> then the result will be the conv_in padded(conv_in_t)
		for(int x=pad; x<(dim_t-pad); x++)
			for(int y=pad; y<(dim_t-pad); y++)
				conv_in_t[i][x][y] = conv_in[i][x-pad][y-pad];
	}

	// Now we can start the convolution
	float sum;
	for (int i=0; i<f_num; i++)//number of filters
	{
		for(int x=0; x<o_dim; x++)//output height
		{
			for(int y=0; y<o_dim; y++)//output width
			{
				sum=0;
				//seeking on the temp image sub array that we want to mult item wise and then add them for the (x,y) result
				for(int j=0; j < ch_num ; j++)
				{
					for(int k=x; k<(x + f); k++)
					{
						for(int l =y; l<(y+f); l++)
						{
							sum += conv_in_t[j][k][l]*filter[i][j][k-x][l-y];
						}
					}
				}
				conv_out[i][x][y] = sum + bias[i];
			}
		}
	}
    
	//push results into structure
	ptr_conv_data->conv_out = conv_out;
	ptr_conv_data->o_dim = o_dim;

	//number of channels is known before func call,(o_ch)num == f_num)
}

//////////////////////// MAXPOOL //////////////////////////
void maxpool(struct maxpool_data_ *ptr_maxpool_data) {

	int fmap_channel;//特征图通道
	int shift_col_num;//卷积核列 横移次数
	int shift_row_num;//卷积核行 竖移次数
	int poolkernel_row;
	int poolkernel_col; //最大池化核扫描计算
	float max_value;

	int fmap_size_in,fmap_size_out,Kernel_size,Kernel_stride,fmap_channel_in,fmap_channel_out;
	float ***input_fmap;
	float ***output_fmap;
	fmap_size_in = ptr_maxpool_data->fmap_size_in;
	fmap_size_out = ptr_maxpool_data->fmap_size_out;
	Kernel_size = ptr_maxpool_data->Kernel_size;
	Kernel_stride = ptr_maxpool_data->Kernel_stride;
	fmap_channel_in = ptr_maxpool_data->fmap_channel_in;
	fmap_channel_out = ptr_maxpool_data->fmap_channel_out;
	
	for (fmap_channel = 0; fmap_channel < fmap_channel_out; fmap_channel++)
	{
		for (shift_col_num = 0; shift_col_num < (((fmap_size_in - Kernel_size) / Kernel_stride) + 1); shift_col_num++)
		{
			for (shift_row_num = 0; shift_row_num < (((fmap_size_in - Kernel_size) / Kernel_stride) + 1); shift_row_num++)
			{
				for (poolkernel_row = 0; poolkernel_row < Kernel_size; poolkernel_row++)
				{
					for (poolkernel_col = 0; poolkernel_col < Kernel_size; poolkernel_col++)
					{
						max_value = (poolkernel_row == 0 && poolkernel_col == 0) ? input_fmap[fmap_channel][poolkernel_row + shift_col_num * Kernel_stride][poolkernel_col + shift_row_num * Kernel_stride] : (input_fmap[fmap_channel][poolkernel_row + shift_col_num * Kernel_stride][poolkernel_col + shift_row_num * Kernel_stride] > max_value) ? input_fmap[fmap_channel][poolkernel_row + shift_col_num * Kernel_stride][poolkernel_col + shift_row_num * Kernel_stride] : max_value;
					}
				}
				output_fmap[fmap_channel][shift_col_num][shift_row_num] = max_value;
			}
		}
	}
}

//////////////////////// TRANS //////////////////////////
void open_PNG() {
	;

}
