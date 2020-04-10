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
 * File name     : operation.hpp
 * Version       : 1.0
 * Last modified : 2020-4-10
 * Describe      : The unary_op function applies the op operation to every
 *                 element of the matrix.
 *                 The unary_op function will perform op operation on two
 *                 matrices and return the new matrix formed by this operation.
 *                 It supports broadcasting. Of course, for coding efficiency,
 *                 I used inefficient broadcasting mechanism.
 *                 These functions will upload data to video RAM. So, when the
 *                 number of data is very small, the performance is not good
 *                 on the CPU. When the amount of data is large, there is a
 *                 lot of time wasted on data transmission.
 *
 * See https://github.com/rihothy/lav_mat to get source code.
 * ************************************************************************/

#ifndef _OPERATION_HPP_
#define _OPERATION_HPP_

#include <lav_mat/lav_mat.h>

template<typename T>
static lav::Mat lav::Mat::unary_op(const lav::Mat& mat, T&& op)
{
    Mat ans(mat.rows, mat.cols, true);

    if (ans.rows * ans.cols)
    {
        if (mat.uploaded)
        {
            boost::compute::transform(mat.g_buffer.begin(), mat.g_buffer.end(), ans.g_buffer.begin(), op, Mat::queue);
        }
        else
        {
            ans.g_buffer.assign(mat.c_buffer.begin(), mat.c_buffer.end(), Mat::queue);
            boost::compute::transform(ans.g_buffer.begin(), ans.g_buffer.end(), ans.g_buffer.begin(), op, Mat::queue);
        }
    }

    return std::move(ans);
}

template<typename T>
static lav::Mat lav::Mat::binary_op(const lav::Mat& a, const lav::Mat& b, T&& op)
{
    static auto&& fun = [&](const auto& a, const auto& b)
    {
        Mat ans(a.rows, a.cols, true);

        if (a.uploaded && b.uploaded)
        {
            boost::compute::transform(a.g_buffer.begin(), a.g_buffer.end(), b.g_buffer.begin(), ans.g_buffer.begin(), op, Mat::queue);
        }
        else if (a.uploaded)
        {
            decltype(b.g_buffer) b_g_buffer(b.c_buffer.begin(), b.c_buffer.end(), Mat::queue);
            boost::compute::transform(a.g_buffer.begin(), a.g_buffer.end(), b_g_buffer.begin(), ans.g_buffer.begin(), op, Mat::queue);
        }
        else if (b.uploaded)
        {
            decltype(a.g_buffer) a_g_buffer(a.c_buffer.begin(), a.c_buffer.end(), Mat::queue);
            boost::compute::transform(a_g_buffer.begin(), a_g_buffer.end(), b.g_buffer.begin(), ans.g_buffer.begin(), op, Mat::queue);
        }
        else
        {
            decltype(a.g_buffer) a_g_buffer(a.c_buffer.begin(), a.c_buffer.end(), Mat::queue);
            ans.g_buffer.assign(b.c_buffer.begin(), b.c_buffer.end(), Mat::queue);

            boost::compute::transform(a_g_buffer.begin(), a_g_buffer.end(), ans.g_buffer.begin(), ans.g_buffer.begin(), op, Mat::queue);
        }

        return std::move(ans);
    };

    if (a.rows == b.rows && a.cols == b.cols)
    {
        return fun(a, b);
    }
    else if (a.rows == b.rows && (a.cols == 1 || b.cols == 1))
    {
        if (a.cols == 1)
        {
            return fun(mul(a, lav::Ones(1, b.cols, true)), b);
        }
        else
        {
            return fun(a, mul(b, lav::Ones(1, a.cols, true)));
        }
    }
    else if (a.cols == b.cols && (a.rows == 1 || b.rows == 1))
    {
        if (a.rows == 1)
        {
            return fun(mul(Ones(b.rows, 1), a, true), b);
        }
        else
        {
            return fun(a, mul(Ones(a.rows, 1, true), b));
        }
    }
    else
    {
        throw std::runtime_error("Binary_op: Size mismatch between two matrices!");
    }
}

#endif