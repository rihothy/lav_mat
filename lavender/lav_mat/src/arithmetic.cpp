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
 * File name     : arithmetic.cpp
 * Version       : 1.0
 * Last modified : 2020-4-10
 *
 * See https://github.com/rihothy/lav_mat to get source code.
 * ************************************************************************/

#include <lav_mat/lav_mat.h>

#include <algorithm>
#include <clBLAS.h>

using namespace lav;
namespace boc = boost::compute;

Mat lav::mul(Mat& a, Mat& b, bool trans_a, bool trans_b)
{
	a.upload();
	b.upload();
	const auto& ta = a;
	const auto& tb = b;
	return mul(ta, tb, trans_a, trans_b);
}

Mat lav::mul(Mat& a, const Mat& b, bool trans_a, bool trans_b)
{
	a.upload();
	const auto& ta = a;
	return mul(ta, b, trans_a, trans_b);
}

Mat lav::mul(const Mat& a, Mat& b, bool trans_a, bool trans_b)
{
	b.upload();
	const auto& tb = b;
	return mul(a, tb, trans_a, trans_b);
}

Mat lav::mul(const Mat& a, const Mat& b, bool trans_a, bool trans_b)
{
	auto r_a = a.rows, c_a = a.cols, r_b = b.rows, c_b = b.cols;

	if (trans_a)
	{
		std::swap(r_a, c_a);
	}

	if (trans_b)
	{
		std::swap(r_b, c_b);
	}
	
	if (c_a == r_b)
	{
		Mat ans(r_a, c_b, true);

		auto&& fun = [&](auto& a_g_buffer, auto& b_g_buffer)
		{
			clblasSgemm
			(
				clblasRowMajor, trans_a ? clblasTrans : clblasNoTrans, trans_b ? clblasTrans : clblasNoTrans,
				r_a, c_b, c_a, 1,
				a_g_buffer, 0, a.cols,
				b_g_buffer, 0, b.cols,
				0, ans.g_buffer.get_buffer().get(), 0, ans.cols, 1,
				&Mat::queue.get(), 0, nullptr, &Mat::event.get()
			);
		};

		if (!a.uploaded && !b.uploaded)
		{
			decltype(a.g_buffer) a_g_buffer(a.c_buffer.begin(), a.c_buffer.end(), Mat::queue);
			decltype(b.g_buffer) b_g_buffer(b.c_buffer.begin(), b.c_buffer.end(), Mat::queue);
			fun(a_g_buffer.get_buffer().get(), b_g_buffer.get_buffer().get());
		}
		else if (!a.uploaded)
		{
			decltype(a.g_buffer) a_g_buffer(a.c_buffer.begin(), a.c_buffer.end(), Mat::queue);
			fun(a_g_buffer.get_buffer().get(), b.g_buffer.get_buffer().get());
		}
		else if (!b.uploaded)
		{
			decltype(b.g_buffer) b_g_buffer(b.c_buffer.begin(), b.c_buffer.end(), Mat::queue);
			fun(a.g_buffer.get_buffer().get(), b_g_buffer.get_buffer().get());
		}
		else
		{
			fun(a.g_buffer.get_buffer().get(), b.g_buffer.get_buffer().get());
		}

		clWaitForEvents(1, &Mat::event.get());

		return std::move(ans);
	}
	else
	{
		throw std::runtime_error("Mul: Size mismatch between two matrices!");
	}
}

Mat lav::operator-(Mat& mat)
{
	mat.upload();
	return Mat::unary_op(mat, 0 - boc::lambda::_1);
}

Mat lav::operator-(const Mat& mat)
{
	return Mat::unary_op(mat, 0 - boc::lambda::_1);
}

Mat lav::operator+(const float& th, Mat& mat)
{
	return mat + th;
}

Mat lav::operator+(const float& th, const Mat& mat)
{
	return mat + th;
}

Mat lav::operator+(Mat& mat, const float& th)
{
	mat.upload();
	return Mat::unary_op(mat, boc::lambda::_1 + th);
}

Mat lav::operator+(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::_1 + th);
}

Mat lav::operator+(Mat& a, Mat& b)
{
	a.upload();
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 + boc::lambda::_2);
}

Mat lav::operator+(Mat& a, const Mat& b)
{
	a.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 + boc::lambda::_2);
}

Mat lav::operator+(const Mat& a, Mat& b)
{
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 + boc::lambda::_2);
}

Mat lav::operator+(const Mat& a, const Mat& b)
{
	return Mat::binary_op(a, b, boc::lambda::_1 + boc::lambda::_2);
}

Mat lav::operator-(const float& th, Mat& mat)
{
	mat.upload();
	return Mat::unary_op(mat, th - boc::lambda::_1);
}

Mat lav::operator-(const float& th, const Mat& mat)
{
	return Mat::unary_op(mat, th - boc::lambda::_1);
}

Mat lav::operator-(Mat& mat, const float& th)
{
	mat.upload();
	return Mat::unary_op(mat, boc::lambda::_1 - th);
}

Mat lav::operator-(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::_1 - th);
}

Mat lav::operator-(Mat& a, Mat& b)
{
	a.upload();
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 - boc::lambda::_2);
}

Mat lav::operator-(Mat& a, const Mat& b)
{
	a.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 - boc::lambda::_2);
}

Mat lav::operator-(const Mat& a, Mat& b)
{
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 - boc::lambda::_2);
}

Mat lav::operator-(const Mat& a, const Mat& b)
{
	return Mat::binary_op(a, b, boc::lambda::_1 - boc::lambda::_2);
}

Mat lav::operator*(const float& th, Mat& mat)
{
	mat.upload();
	return mat * th;
}

Mat lav::operator*(const float& th, const Mat& mat)
{
	return mat * th;
}

Mat lav::operator*(Mat& mat, const float& th)
{
	mat.upload();
	return Mat::unary_op(mat, boc::lambda::_1 * th);
}

Mat lav::operator*(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::_1 * th);
}

Mat lav::operator*(Mat& a, Mat& b)
{
	a.upload();
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 * boc::lambda::_2);
}

Mat lav::operator*(Mat& a, const Mat& b)
{
	a.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 * boc::lambda::_2);
}

Mat lav::operator*(const Mat& a, Mat& b)
{
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 * boc::lambda::_2);
}

Mat lav::operator*(const Mat& a, const Mat& b)
{
	return Mat::binary_op(a, b, boc::lambda::_1 * boc::lambda::_2);
}

Mat lav::operator/(const float& th, Mat& mat)
{
	mat.upload();
	return Mat::unary_op(mat, th / boc::lambda::_1);
}

Mat lav::operator/(const float& th, const Mat& mat)
{
	return Mat::unary_op(mat, th / boc::lambda::_1);
}

Mat lav::operator/(Mat& mat, const float& th)
{
	mat.upload();
	return Mat::unary_op(mat, boc::lambda::_1 / th);
}

Mat lav::operator/(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::_1 / th);
}

Mat lav::operator/(Mat& a, Mat& b)
{
	a.upload();
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 / boc::lambda::_2);
}

Mat lav::operator/(Mat& a, const Mat& b)
{
	a.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 / boc::lambda::_2);
}

Mat lav::operator/(const Mat& a, Mat& b)
{
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 / boc::lambda::_2);
}

Mat lav::operator/(const Mat& a, const Mat& b)
{
	return Mat::binary_op(a, b, boc::lambda::_1 / boc::lambda::_2);
}

Mat lav::operator<(const float& th, Mat& mat)
{
	return mat > th;
}

Mat lav::operator<(const float& th, const Mat& mat)
{
	return mat > th;
}

Mat lav::operator<(Mat& mat, const float& th)
{
	return th > mat;
}

Mat lav::operator<(const Mat& mat, const float& th)
{
	return th > mat;
}

Mat lav::operator<(Mat& a, Mat& b)
{
	return b > a;
}

Mat lav::operator<(Mat& a, const Mat& b)
{
	return b > a;
}

Mat lav::operator<(const Mat& a, Mat& b)
{
	return b > a;
}

Mat lav::operator<(const Mat& a, const Mat& b)
{
	return b > a;
}

Mat lav::operator<=(const float& th, Mat& mat)
{
	return mat >= th;
}

Mat lav::operator<=(const float& th, const Mat& mat)
{
	return mat >= th;
}

Mat lav::operator<=(Mat& mat, const float& th)
{
	return th >= mat;
}

Mat lav::operator<=(const Mat& mat, const float& th)
{
	return th >= mat;
}

Mat lav::operator<=(Mat& a, Mat& b)
{
	return b >= a;
}

Mat lav::operator<=(Mat& a, const Mat& b)
{
	return b >= a;
}

Mat lav::operator<=(const Mat& a, Mat& b)
{
	return b >= a;
}

Mat lav::operator<=(const Mat& a, const Mat& b)
{
	return b >= a;
}

Mat lav::operator>(const float& th, Mat& mat)
{
	mat.upload();
	return Mat::unary_op(mat, th > boc::lambda::_1);
}

Mat lav::operator>(const float& th, const Mat& mat)
{
	return Mat::unary_op(mat, th > boc::lambda::_1);
}

Mat lav::operator>(Mat& mat, const float& th)
{
	mat.upload();
	return Mat::unary_op(mat, boc::lambda::_1 > th);
}

Mat lav::operator>(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::_1 > th);
}

Mat lav::operator>(Mat& a, Mat& b)
{
	a.upload();
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 > boc::lambda::_2);
}

Mat lav::operator>(Mat& a, const Mat& b)
{
	a.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 > boc::lambda::_2);
}

Mat lav::operator>(const Mat& a, Mat& b)
{
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 > boc::lambda::_2);
}

Mat lav::operator>(const Mat& a, const Mat& b)
{
	return Mat::binary_op(a, b, boc::lambda::_1 > boc::lambda::_2);
}

Mat lav::operator>=(const float& th, Mat& mat)
{
	mat.upload();
	return Mat::unary_op(mat, th >= boc::lambda::_1);
}

Mat lav::operator>=(const float& th, const Mat& mat)
{
	return Mat::unary_op(mat, th >= boc::lambda::_1);
}

Mat lav::operator>=(Mat& mat, const float& th)
{
	mat.upload();
	return Mat::unary_op(mat, boc::lambda::_1 >= th);
}

Mat lav::operator>=(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::_1 >= th);
}

Mat lav::operator>=(Mat& a, Mat& b)
{
	a.upload();
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 >= boc::lambda::_2);
}

Mat lav::operator>=(Mat& a, const Mat& b)
{
	a.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 >= boc::lambda::_2);
}

Mat lav::operator>=(const Mat& a, Mat& b)
{
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 >= boc::lambda::_2);
}

Mat lav::operator>=(const Mat& a, const Mat& b)
{
	return Mat::binary_op(a, b, boc::lambda::_1 >= boc::lambda::_2);
}

Mat lav::operator==(const float& th, Mat& mat)
{
	return mat == th;
}

Mat lav::operator==(const float& th, const Mat& mat)
{
	return mat == th;
}

Mat lav::operator==(Mat& mat, const float& th)
{
	mat.upload();
	return Mat::unary_op(mat, boc::lambda::_1 == th);
}

Mat lav::operator==(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::_1 == th);
}

Mat lav::operator==(Mat& a, Mat& b)
{
	a.upload();
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 == boc::lambda::_2);
}

Mat lav::operator==(Mat& a, const Mat& b)
{
	a.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 == boc::lambda::_2);
}

Mat lav::operator==(const Mat& a, Mat& b)
{
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 == boc::lambda::_2);
}

Mat lav::operator==(const Mat& a, const Mat& b)
{
	return Mat::binary_op(a, b, boc::lambda::_1 == boc::lambda::_2);
}

Mat lav::operator!=(const float& th, Mat& mat)
{
	return mat != th;
}

Mat lav::operator!=(const float& th, const Mat& mat)
{
	return mat != th;
}

Mat lav::operator!=(Mat& mat, const float& th)
{
	mat.upload();

	return Mat::unary_op(mat, boc::lambda::_1 != th);
}

Mat lav::operator!=(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::_1 != th);
}

Mat lav::operator!=(Mat& a, Mat& b)
{
	a.upload();
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 != boc::lambda::_2);
}

Mat lav::operator!=(Mat& a, const Mat& b)
{
	a.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 != boc::lambda::_2);
}

Mat lav::operator!=(const Mat& a, Mat& b)
{
	b.upload();
	return Mat::binary_op(a, b, boc::lambda::_1 != boc::lambda::_2);
}

Mat lav::operator!=(const Mat& a, const Mat& b)
{
	return Mat::binary_op(a, b, boc::lambda::_1 != boc::lambda::_2);
}