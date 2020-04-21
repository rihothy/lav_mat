/* ************************************************************************
 * Copyright 2020 Rihothy.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/

/* ************************************************************************
 * Author        : ¿Ó∫ÈˆŒ(Rihothy)
 * File name     : convolution.cpp
 * Version       : 1.0
 * Last modified : 2020-4-21
 * Describe      : size[0]: f's width
 *                 size[1]: f's height
 *                 size[2]: f's channel
 *                 size[3]: g's width
 *                 size[4]: batch size
 *
 * See https://github.com/rihothy/lav_mat to get source code.
 * ************************************************************************/

#include <lav_mat/lav_mat.h>

using namespace lav;
namespace boc = boost::compute;

Mat lav::conv4d(Mat& f, Mat& g, std::vector<size_t> size, const size_t& stride, const std::string padding)
{
	f.upload();
	g.upload();
	const auto& t_f = f;
	const auto& t_g = g;
	return conv4d(t_f, t_g, size, stride, padding);
}

Mat lav::conv4d(Mat& f, const Mat& g, std::vector<size_t> size, const size_t& stride, const std::string padding)
{
	f.upload();
	const auto& t_f = f;
	return conv4d(t_f, g, size, stride, padding);
}

Mat lav::conv4d(const Mat& f, Mat& g, std::vector<size_t> size, const size_t& stride, const std::string padding)
{
	g.upload();
	const auto& t_g = g;
	return conv4d(f, t_g, size, stride, padding);
}

Mat lav::conv4d(const Mat& f, const Mat& g, std::vector<size_t> size, const size_t& stride, const std::string padding)
{
	static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE
	(
		__kernel void fun(__global float* input, __global float* output, size_t w, size_t h, size_t channel, size_t nw, size_t f, size_t s, int valid_padding, size_t group_r, size_t group_c)
		{
			const int r = get_global_id(0);
			const int c = get_global_id(1);
			const int i = r % group_r;
			const int j = c % group_c;
			const int b_id = r / group_r;
			const int c_id = c / group_c;

			const int nr = i / nw;
			const int nc = i % nw;

			const int fr = j / f;
			const int fc = j % f;

			if (valid_padding)
			{
				const int or = nr * s + fr;
				const int oc = nc * s + fc;

				output[r * get_global_size(1) + c] = input[(w * h * b_id + (or *w + oc)) * channel + c_id];
			}
			else
			{
				const int or = nr * s + fr - f / 2;
				const int oc = nc * s + fc - f / 2;

				if (or < 0 || or >= h || oc < 0 || oc >= w)
				{
					output[r * get_global_size(1) + c] = 0;
				}
				else
				{
					output[r * get_global_size(1) + c] = input[(w * h * b_id + (or *w + oc)) * channel + c_id];
				}
			}
		}
	);

	static boc::program fun_program = boc::program::build_with_source(source, Mat::context);
	static boc::kernel fun_kernel(fun_program, "fun");

	bool valid_padding = padding == "valid";

	if (valid_padding || padding == "same")
	{
		if (size.size() <= 5)
		{
			if (size.size() < 5)
			{
				size.insert(size.end(), 5 - size.size(), 1);
			}

			if (size[3] % 2 == 0)
			{
				throw std::runtime_error("Conv4d: Side length of filter must be odd!");
			}

			if (size[0] * size[1] * size[4] == f.rows && size[2] == f.cols && size[3] * size[3] * size[2] == g.rows)
			{
				size_t nw = valid_padding ? (size[0] - size[3]) / stride + 1 : (size[0] + stride - 1) / stride;
				size_t nh = valid_padding ? (size[1] - size[3]) / stride + 1 : (size[1] + stride - 1) / stride;

				Mat temp(nw * nh * size[4], g.rows, true);

				auto fun = [&](auto& f_g_buffer, auto& temp_g_buffer)
				{
					fun_kernel.set_arg(0, f_g_buffer);
					fun_kernel.set_arg(1, temp_g_buffer);
					fun_kernel.set_arg(2, size[0]);
					fun_kernel.set_arg(3, size[1]);
					fun_kernel.set_arg(4, size[2]);
					fun_kernel.set_arg(5, nw);
					fun_kernel.set_arg(6, size[3]);
					fun_kernel.set_arg(7, stride);
					fun_kernel.set_arg(8, int(valid_padding));
					fun_kernel.set_arg(9, nw * nh);
					fun_kernel.set_arg(10, size[3] * size[3]);

					auto event = Mat::queue.enqueue_nd_range_kernel(fun_kernel, boc::extents<2>({ 0, 0 }), boc::extents<2>({ nw * nh * size[4], g.rows }), boc::extents<2>({ 1, 1 }));
					event.wait();
				};

				if (f.uploaded)
				{
					fun(f.g_buffer, temp.g_buffer);
				}
				else
				{
					decltype(f.g_buffer) t_f_g_buffer(f.c_buffer.begin(), f.c_buffer.end(), Mat::queue);
					fun(t_f_g_buffer, temp.g_buffer);
				}

				return lav::mul(temp, g);
			}
			else
			{
				throw std::runtime_error("Conv4d: The size of matrices f and g must match the value of vector size!");
			}
		}
		else
		{
			throw std::runtime_error("Conv4d: Size vector must have 5 dimensions, which is width, height, channel, filter size and batch size!");
		}
	}
	else
	{
		throw std::runtime_error("Conv4d: The value of padding can only be \"valid\" or \"same\"!");
	}
}