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
 * File name     : algorithm.cpp
 * Version       : 1.0
 * Last modified : 2020-4-10
 *
 * See https://github.com/rihothy/lav_mat to get source code.
 * ************************************************************************/

#include <lav_mat/lav_mat.h>

using namespace lav;
namespace boc = boost::compute;

Mat Mat::t(void)
{
	upload();
	const auto& temp = *this;
	return temp.t();
}

Mat Mat::t(void) const
{
	static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE
	(
		__kernel void fun(__global float* input, __global float* output, size_t rows, size_t cols)
		{
			const uint i = get_global_id(0);

			output[i % cols * rows + i / cols] = input[i];
		}
	);

	static boc::program fun_program = boc::program::build_with_source(source, Mat::context);
	static boc::kernel fun_kernel(fun_program, "fun");

	auto&& fun = [&](auto& input, auto& output)
	{
		fun_kernel.set_arg(0, input);
		fun_kernel.set_arg(1, output);
		fun_kernel.set_arg(2, rows);
		fun_kernel.set_arg(3, cols);

		auto event = Mat::queue.enqueue_1d_range_kernel(fun_kernel, 0, rows * cols, 0);
		event.wait();
	};

	Mat ans(cols, rows, true);

	if (uploaded)
	{
		fun(g_buffer, ans.g_buffer);
	}
	else
	{
		decltype(g_buffer) t_g_buffer(c_buffer.begin(), c_buffer.end(), Mat::queue);
		fun(t_g_buffer, ans.g_buffer);
	}

	return std::move(ans);
}

float Mat::max(void)
{
	upload();
	const auto& temp = *this;
	return temp.max();
}

float Mat::max(void) const
{
	if (uploaded)
	{
		return *boc::max_element(g_buffer.begin(), g_buffer.end(), Mat::queue);
	}
	else
	{
		if (rows * cols < 1e6)
		{
			return *std::max_element(c_buffer.begin(), c_buffer.end());
		}
		else
		{
			decltype(g_buffer) t_g_buffer(c_buffer.begin(), c_buffer.end(), Mat::queue);
			return *boc::max_element(t_g_buffer.begin(), t_g_buffer.end(), Mat::queue);
		}
	}
}

Mat Mat::max(bool axis)
{
	upload();
	const auto& temp = *this;
	return temp.max(axis);
}

Mat Mat::max(bool axis) const
{
	static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE
	(
		__kernel void fun(__global float* input, __global float* output, size_t rows, size_t cols, size_t axis)
		{
			const uint i = get_global_id(0);
		
			if (axis)
			{
				output[i] = input[i * cols];

				for (size_t j = 0; j < cols; ++j)
				{
					output[i] = output[i] > input[i * cols + j] ? output[i] : input[i * cols + j];
				}
			}
			else
			{
				output[i] = input[i];

				for (size_t j = 0; j < rows; ++j)
				{
					output[i] = output[i] > input[j * cols + i] ? output[i] : input[j * cols + i];
				}
			}
		}
	);

	static boc::program fun_program = boc::program::build_with_source(source, Mat::context);
	static boc::kernel fun_kernel(fun_program, "fun");

	auto&& fun = [&](auto& input, auto& output)
	{
		fun_kernel.set_arg(0, input);
		fun_kernel.set_arg(1, output);
		fun_kernel.set_arg(2, rows);
		fun_kernel.set_arg(3, cols);
		fun_kernel.set_arg(4, size_t(axis));

		auto event = Mat::queue.enqueue_1d_range_kernel(fun_kernel, 0, axis ? rows : cols, 0);
		event.wait();
	};

	Mat ans(axis ? rows : 1, axis ? 1 : cols, true);

	if (uploaded)
	{
		fun(g_buffer, ans.g_buffer);
	}
	else
	{
		decltype(g_buffer) t_g_buffer(c_buffer.begin(), c_buffer.end(), Mat::queue);
		fun(t_g_buffer, ans.g_buffer);
	}

	return std::move(ans);
}

float Mat::min(void)
{
	upload();
	const auto& temp = *this;
	return temp.min();
}

float Mat::min(void) const
{
	if (uploaded)
	{
		return *boc::min_element(g_buffer.begin(), g_buffer.end(), Mat::queue);
	}
	else
	{
		if (rows * cols < 1e6)
		{
			return *std::min_element(c_buffer.begin(), c_buffer.end());
		}
		else
		{
			decltype(g_buffer) t_g_buffer(c_buffer.begin(), c_buffer.end(), Mat::queue);
			return *boc::min_element(t_g_buffer.begin(), t_g_buffer.end(), Mat::queue);
		}
	}
}

Mat Mat::min(bool axis)
{
	upload();
	const auto& temp = *this;
	return temp.min(axis);
}

Mat Mat::min(bool axis) const
{
	static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE
	(
		__kernel void fun(__global float* input, __global float* output, size_t rows, size_t cols, size_t axis)
		{
			const uint i = get_global_id(0);

			if (axis)
			{
				output[i] = input[i * cols];

				for (size_t j = 0; j < cols; ++j)
				{
					output[i] = output[i] < input[i * cols + j] ? output[i] : input[i * cols + j];
				}
			}
			else
			{
				output[i] = input[i];

				for (size_t j = 0; j < rows; ++j)
				{
					output[i] = output[i] < input[j * cols + i] ? output[i] : input[j * cols + i];
				}
			}
		}
	);

	static boc::program fun_program = boc::program::build_with_source(source, Mat::context);
	static boc::kernel fun_kernel(fun_program, "fun");

	auto&& fun = [&](auto& input, auto& output)
	{
		fun_kernel.set_arg(0, input);
		fun_kernel.set_arg(1, output);
		fun_kernel.set_arg(2, rows);
		fun_kernel.set_arg(3, cols);
		fun_kernel.set_arg(4, size_t(axis));

		auto event = Mat::queue.enqueue_1d_range_kernel(fun_kernel, 0, axis ? rows : cols, 0);
		event.wait();
	};

	Mat ans(axis ? rows : 1, axis ? 1 : cols, true);

	if (uploaded)
	{
		fun(g_buffer, ans.g_buffer);
	}
	else
	{
		decltype(g_buffer) t_g_buffer(c_buffer.begin(), c_buffer.end(), Mat::queue);
		fun(t_g_buffer, ans.g_buffer);
	}

	return std::move(ans);
}

float Mat::sum(void)
{
	upload();
	const auto& temp = *this;
	return temp.sum();
}

float Mat::sum(void) const
{
	float ans;

	if (uploaded)
	{
		boc::reduce(g_buffer.begin(), g_buffer.end(), &ans, Mat::queue);
	}
	else
	{
		if (rows * cols < 1e6)
		{
			ans = std::accumulate(c_buffer.begin(), c_buffer.end(), 0.0f);
		}
		else
		{
			decltype(g_buffer) t_g_buffer(c_buffer.begin(), c_buffer.end(), Mat::queue);
			boc::reduce(t_g_buffer.begin(), t_g_buffer.end(), &ans, Mat::queue);
		}
	}

	return ans;
}

Mat Mat::sum(bool axis)
{
	upload();
	const auto& temp = *this;
	return temp.sum(axis);
}

Mat Mat::sum(bool axis) const
{
	if (axis)
	{
		return mul(*this, Ones(cols, 1, true));
	}
	else
	{
		return mul(Ones(1, rows, true), *this);
	}
}

float Mat::mean(void)
{
	upload();
	const auto& temp = *this;
	return temp.mean();
}

float Mat::mean(void) const
{
	return sum() / (rows * cols);
}

Mat Mat::mean(bool axis)
{
	upload();
	const auto& temp = *this;
	return temp.mean(axis);
}

Mat Mat::mean(bool axis) const
{
	return sum(axis) / (axis ? cols : rows);
}

Mat Mat::max_loc(bool axis)
{
	upload();
	const auto& temp = *this;
	return temp.max_loc(axis);
}

Mat Mat::max_loc(bool axis) const
{
	static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE
	(
		__kernel void fun(__global float* input, __global float* output, size_t rows, size_t cols, size_t axis)
		{
			const uint i = get_global_id(0);

			if (axis)
			{
				output[i] = 0;
				float num = input[i * cols];
				
				for (size_t j = 0; j < cols; ++j)
				{
					if (num < input[i * cols + j])
					{
						num = input[i * cols + j];
						output[i] = j;
					}
				}
			}
			else
			{
				output[i] = 0;
				float num = input[i];
				
				for (size_t j = 0; j < rows; ++j)
				{
					if (num < input[j * cols + i])
					{
						num = input[j * cols + i];
						output[i] = j;
					}
				}
			}
		}
	);

	static boc::program fun_program = boc::program::build_with_source(source, Mat::context);
	static boc::kernel fun_kernel(fun_program, "fun");

	auto&& fun = [&](auto& input, auto& output)
	{
		fun_kernel.set_arg(0, input);
		fun_kernel.set_arg(1, output);
		fun_kernel.set_arg(2, rows);
		fun_kernel.set_arg(3, cols);
		fun_kernel.set_arg(4, size_t(axis));

		auto event = Mat::queue.enqueue_1d_range_kernel(fun_kernel, 0, axis ? rows : cols, 0);
		event.wait();
	};

	Mat ans(axis ? rows : 1, axis ? 1 : cols, true);

	if (uploaded)
	{
		fun(g_buffer, ans.g_buffer);
	}
	else
	{
		decltype(g_buffer) t_g_buffer(c_buffer.begin(), c_buffer.end(), Mat::queue);
		fun(t_g_buffer, ans.g_buffer);
	}

	return std::move(ans);
}

Mat Mat::min_loc(bool axis)
{
	upload();
	const auto& temp = *this;
	return temp.min_loc(axis);
}

Mat Mat::min_loc(bool axis) const
{
	static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE
	(
		__kernel void fun(__global float* input, __global float* output, size_t rows, size_t cols, size_t axis)
		{
			const uint i = get_global_id(0);

			if (axis)
			{
				output[i] = 0;
				float num = input[i * cols];

				for (size_t j = 0; j < cols; ++j)
				{
					if (num > input[i * cols + j])
					{
						num = input[i * cols + j];
						output[i] = j;
					}
				}
			}
			else
			{
				output[i] = 0;
				float num = input[i];

				for (size_t j = 0; j < rows; ++j)
				{
					if (num > input[j * cols + i])
					{
						num = input[j * cols + i];
						output[i] = j;
					}
				}
			}
		}
	);

	static boc::program fun_program = boc::program::build_with_source(source, Mat::context);
	static boc::kernel fun_kernel(fun_program, "fun");

	auto&& fun = [&](auto& input, auto& output)
	{
		fun_kernel.set_arg(0, input);
		fun_kernel.set_arg(1, output);
		fun_kernel.set_arg(2, rows);
		fun_kernel.set_arg(3, cols);
		fun_kernel.set_arg(4, size_t(axis));

		auto event = Mat::queue.enqueue_1d_range_kernel(fun_kernel, 0, axis ? rows : cols, 0);
		event.wait();
	};

	Mat ans(axis ? rows : 1, axis ? 1 : cols, true);

	if (uploaded)
	{
		fun(g_buffer, ans.g_buffer);
	}
	else
	{
		decltype(g_buffer) t_g_buffer(c_buffer.begin(), c_buffer.end(), Mat::queue);
		fun(t_g_buffer, ans.g_buffer);
	}

	return std::move(ans);
}

Mat lav::max(const float& th, Mat& mat)
{
	return max(mat, th);
}

Mat lav::max(const float& th, const Mat& mat)
{
	return max(mat, th);
}

Mat lav::max(Mat& mat, const float& th)
{
	mat.upload();
	return Mat::unary_op(mat, boc::lambda::max(boc::lambda::_1, th));
}

Mat lav::max(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::max(boc::lambda::_1, th));
}

Mat lav::max(Mat& a, Mat& b)
{
	a.upload();
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::max(boc::lambda::_1, boc::lambda::_2));
}

Mat lav::max(Mat& a, const Mat& b)
{
	a.upload();
	return Mat::binary_op(a, b, boc::lambda::max(boc::lambda::_1, boc::lambda::_2));
}

Mat lav::max(const Mat& a, Mat& b)
{
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::max(boc::lambda::_1, boc::lambda::_2));
}

Mat lav::max(const Mat& a, const Mat& b)
{
	return Mat::binary_op(a, b, boc::lambda::max(boc::lambda::_1, boc::lambda::_2));
}

Mat lav::min(const float& th, Mat& mat)
{
	return min(mat, th);
}

Mat lav::min(const float& th, const Mat& mat)
{
	return min(mat, th);
}

Mat lav::min(Mat& mat, const float& th)
{
	mat.upload();
	return Mat::unary_op(mat, boc::lambda::min(boc::lambda::_1, th));
}

Mat lav::min(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::min(boc::lambda::_1, th));
}

Mat lav::min(Mat& a, Mat& b)
{
	a.upload();
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::min(boc::lambda::_1, boc::lambda::_2));
}

Mat lav::min(Mat& a, const Mat& b)
{
	a.upload();
	return Mat::binary_op(a, b, boc::lambda::min(boc::lambda::_1, boc::lambda::_2));
}

Mat lav::min(const Mat& a, Mat& b)
{
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::min(boc::lambda::_1, boc::lambda::_2));
}

Mat lav::min(const Mat& a, const Mat& b)
{
	return Mat::binary_op(a, b, boc::lambda::min(boc::lambda::_1, boc::lambda::_2));
}

Mat lav::shuffle(Mat& mat)
{
	mat.upload();
	const auto& temp = mat;
	return shuffle(temp);
}

Mat lav::shuffle(const Mat& mat)
{
	static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE
	(
		__kernel void fun(__global float* input, __global float* output, size_t rows, size_t cols, size_t axis)
		{
			const uint i = get_global_id(0);

			if (axis)
			{
				output[i] = 0;
				float num = input[i * cols];

				for (size_t j = 0; j < cols; ++j)
				{
					if (num > input[i * cols + j])
					{
						num = input[i * cols + j];
						output[i] = j;
					}
				}
			}
			else
			{
				output[i] = 0;
				float num = input[i];

				for (size_t j = 0; j < rows; ++j)
				{
					if (num > input[j * cols + i])
					{
						num = input[j * cols + i];
						output[i] = j;
					}
				}
			}
		}
	);

	static boc::program fun_program = boc::program::build_with_source(source, Mat::context);
	static boc::kernel fun_kernel(fun_program, "fun");

	auto&& fun = [&](auto& input, auto& output)
	{

		fun_kernel.set_arg(0, input);
		fun_kernel.set_arg(1, output);

		auto event = Mat::queue.enqueue_1d_range_kernel(fun_kernel, 0, 0, 0);
		event.wait();
	};

	Mat ans(mat.rows, mat.cols, true);

	if (mat.uploaded)
	{
		fun(mat.g_buffer, ans.g_buffer);
	}
	else
	{
		decltype(mat.g_buffer) t_g_buffer(mat.c_buffer.begin(), mat.c_buffer.end(), Mat::queue);
		fun(t_g_buffer, ans.g_buffer);
	}

	return std::move(ans);
}

Mat lav::shuffle(Mat& mat, bool axis, bool same_as_last_time)
{
	mat.upload();
	const auto& temp = mat;
	return shuffle(temp, axis, same_as_last_time);
}

Mat lav::shuffle(const Mat& mat, bool axis, bool same_as_last_time)
{
	static bool flag = false;

	if (not flag)
	{
		flag = true;
		srand(time(nullptr));
	}

	static size_t last_rows = 0, last_cols = 0;
	static std::vector<float> c_indexes;
	static boc::vector<float> g_indexes(Mat::context);

	if (!(same_as_last_time && axis ? mat.rows == last_rows : mat.cols == last_cols))
	{
		last_rows = mat.rows, last_cols = mat.cols;

		size_t i = 0;
		c_indexes.resize(axis ? mat.rows : mat.cols);
		std::generate(c_indexes.begin(), c_indexes.end(), [&] {return i++; });

		for (size_t i = c_indexes.size() - 1; i < c_indexes.size(); --i)
		{
			size_t randint = 0;

			for (size_t j = 0; j < 8; ++j)
			{
				randint = randint * 10 + rand() % 10;
			}

			std::swap(c_indexes[i], c_indexes[randint % (i + 1)]);
		}

		g_indexes.assign(c_indexes.begin(), c_indexes.end(), Mat::queue);
	}

	static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE
	(
		__kernel void fun(__global float* input, __global float* output, __global float* indexes, size_t rows, size_t cols, size_t axis)
		{
			const uint i = get_global_id(0);
			const uint row = i / cols;
			const uint col = i % cols;

			if (axis)
			{
				const size_t j = indexes[row];

				output[i] = input[j * cols + col];
			}
			else
			{
				const size_t j = indexes[col];

				output[i] = input[row * cols + j];
			}
		}
	);

	static boc::program fun_program = boc::program::build_with_source(source, Mat::context);
	static boc::kernel fun_kernel(fun_program, "fun");

	auto&& fun = [&](auto& input, auto& output)
	{
		fun_kernel.set_arg(0, input);
		fun_kernel.set_arg(1, output);
		fun_kernel.set_arg(2, g_indexes);
		fun_kernel.set_arg(3, mat.rows);
		fun_kernel.set_arg(4, mat.cols);
		fun_kernel.set_arg(5, size_t(axis));

		auto event = Mat::queue.enqueue_1d_range_kernel(fun_kernel, 0, mat.rows * mat.cols, 0);
		event.wait();
	};

	Mat ans(mat.rows, mat.cols, true);

	if (mat.uploaded)
	{
		fun(mat.g_buffer, ans.g_buffer);
	}
	else
	{
		decltype(mat.g_buffer) t_g_buffer(mat.c_buffer.begin(), mat.c_buffer.end(), Mat::queue);
		fun(t_g_buffer, ans.g_buffer);
	}

	return std::move(ans);
}