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
 * File name     : math.cpp
 * Version       : 1.0
 * Last modified : 2020-4-10
 *
 * See https://github.com/rihothy/lav_mat to get source code.
 * ************************************************************************/

#include <lav_mat/lav_mat.h>

using namespace lav;
namespace boc = boost::compute;

Mat lav::exp(const Mat& mat)
{
	return Mat::unary_op(mat, boc::exp<float>());
}

Mat lav::abs(const Mat& mat)
{
	return Mat::unary_op(mat, boc::fabs<float>());
}

Mat lav::log(const Mat& mat)
{
	return Mat::unary_op(mat, boc::log<float>());
}

Mat lav::log2(const Mat& mat)
{
	return Mat::unary_op(mat, boc::log2<float>());
}

Mat lav::log10(const Mat& mat)
{
	return Mat::unary_op(mat, boc::log10<float>());
}

Mat lav::sqrt(const Mat& mat)
{
	return Mat::unary_op(mat, boc::sqrt<float>());
}

Mat lav::pow(const Mat& mat, const float& th)
{
	return Mat::unary_op(mat, boc::lambda::pow(boc::lambda::_1, th));
}