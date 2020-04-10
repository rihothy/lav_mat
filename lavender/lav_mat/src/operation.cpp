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
 * File name     : operation.cpp
 * Version       : 1.0
 * Last modified : 2020-4-10
 *
 * See https://github.com/rihothy/lav_mat to get source code.
 * ************************************************************************/

#include <lav_mat/lav_mat.h>

using namespace lav;
namespace boc = boost::compute;

void Mat::upload(void)
{
    if (!uploaded && !c_buffer.empty())
    {
        try
        {
            g_buffer.assign(c_buffer.begin(), c_buffer.end(), Mat::queue);
        }
        catch (...)
        {
            throw std::runtime_error("Upload: Vedio memory overflow!");
        }
    }

    uploaded = true;
}

void Mat::download(void)
{
    if (uploaded)
    {
        c_buffer.resize(g_buffer.size());
        boc::copy(g_buffer.begin(), g_buffer.end(), c_buffer.begin(), Mat::queue);
    }

    uploaded = false;
}

Mat Mat::row(size_t row)
{
    upload();
    const auto& temp = *this;
    return temp.row(row);
}

Mat Mat::row(size_t row) const
{
    if (row < rows)
    {
        Mat ans(1, cols, true);

        if (uploaded)
        {
            ans.g_buffer.assign(g_buffer.begin() + row * cols, g_buffer.begin() + (row + 1) * cols, Mat::queue);
        }
        else
        {
            ans.g_buffer.assign(c_buffer.begin() + row * cols, c_buffer.begin() + (row + 1) * cols, Mat::queue);
        }

        return std::move(ans);
    }
    else
    {
        throw std::runtime_error("Operator(): Subscript out of range!");
    }
}

Mat Mat::col(size_t col)
{
    return this->operator()(0, 0, col, col + 1);
}

Mat Mat::col(size_t col) const
{
    return this->operator()(0, 0, col, col + 1);
}

Mat& Mat::reshape(size_t rows, size_t cols)
{
    if (rows * cols == this->rows * this->cols)
    {
        this->rows = rows;
        this->cols = cols;

        return *this;
    }
    else
    {
        throw std::runtime_error("Reshape: The total number of elements cannot be changed!");
    }
}

void Mat::push_back(const Mat& another)
{
    if (!cols || another.cols == cols)
    {
        cols = another.cols;
        rows += another.rows;

        if (another.uploaded)
        {
            upload();

            g_buffer.insert(g_buffer.end(), another.g_buffer.begin(), another.g_buffer.end(), Mat::queue);

            uploaded = true;
        }
        else
        {
            if (uploaded)
            {
                g_buffer.insert(g_buffer.end(), another.c_buffer.begin(), another.c_buffer.end(), Mat::queue);
            }
            else
            {
                c_buffer.insert(c_buffer.end(), another.c_buffer.begin(), another.c_buffer.end());
            }
        }
    }
    else
    {
        throw std::runtime_error("Push_back: The number of columns of two matrices must be the same!");
    }
}

void Mat::push_back(const std::initializer_list<float>& vec)
{
    if (!cols || vec.size() == cols)
    {
        cols = vec.size();
        ++rows;

        if (uploaded)
        {
            g_buffer.insert(g_buffer.end(), vec.begin(), vec.end(), Mat::queue);
        }
        else
        {
            c_buffer.insert(c_buffer.end(), vec.begin(), vec.end());
        }
    }
    else
    {
        throw std::runtime_error("Push_back: The length of that vector must be the same as the number of cols of the matrix!");
    }
}

Mat& Mat::operator=(Mat&& another) noexcept
{
    if (this != &another)
    {
        rows = another.rows;
        cols = another.cols;
        uploaded = another.uploaded;

        if (another.uploaded)
        {
            g_buffer = std::move(another.g_buffer);
        }
        else
        {
            c_buffer = std::move(another.c_buffer);
        }
    }

    return *this;
}

Mat& Mat::operator=(const Mat& another)
{
    if (this != &another)
    {
        rows = another.rows;
        cols = another.cols;
        uploaded = another.uploaded;

        if (another.uploaded)
        {
            g_buffer.assign(another.g_buffer.begin(), another.g_buffer.end(), Mat::queue);
        }
        else
        {
            c_buffer.assign(another.c_buffer.begin(), another.c_buffer.end());
        }
    }

    return *this;
}

float& Mat::operator()(const size_t& row, const size_t& col)
{
    download();
    return c_buffer[row * cols + col];
}

float Mat::operator()(const size_t& row, const size_t& col) const
{
    if (!uploaded)
    {
        return c_buffer[row * cols + col];
    }
    else
    {
        throw std::runtime_error("Operator(): The data is on the vedio memory and cannot be accessed!");
    }
}

Mat Mat::operator()(size_t first_row, size_t last_row, size_t first_col, size_t last_col)
{
    upload();
    const auto& temp = *this;
    return temp(first_row, last_row, first_col, last_col);
}

Mat Mat::operator()(size_t first_row, size_t last_row, size_t first_col, size_t last_col) const
{
    static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE
    (
        __kernel void fun(__global float* input, __global float* output, size_t first_row, size_t last_row, size_t first_col, size_t last_col, size_t cols)
        {
            const uint i = get_global_id(0);
            uint r = i / cols;
            uint c = i % cols;
        
            if (r >= first_row && r < last_row && c >= first_col && c < last_col)
            {
                r -= first_row;
                c -= first_col;
                output[r * (last_col - first_col) + c] = input[i];
            }
        }
    );

    static boc::program fun_program = boc::program::build_with_source(source, Mat::context);
    static boc::kernel fun_kernel(fun_program, "fun");

    if (first_row == last_row && !first_row)
    {
        last_row = rows;
    }

    if (first_col == last_col && !first_col)
    {
        last_col = cols;
    }

    if (!(last_row > rows || last_col > cols || first_row >= last_row || first_col >= last_col))
    {
        Mat ans(last_row - first_row, last_col - first_col, true);

        static auto&& fun = [&](auto& input, auto& output)
        {
            fun_kernel.set_arg(0, input);
            fun_kernel.set_arg(1, output);
            fun_kernel.set_arg(2, first_row);
            fun_kernel.set_arg(3, last_row);
            fun_kernel.set_arg(4, first_col);
            fun_kernel.set_arg(5, last_col);
            fun_kernel.set_arg(6, cols);

            auto event = Mat::queue.enqueue_1d_range_kernel(fun_kernel, 0, input.size(), 0);
            event.wait();
        };

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
    else
    {
        throw std::runtime_error("Operator(): Subscript out of range!");
    }
}

std::ostream& lav::operator<<(std::ostream& cout, Mat& mat)
{
    mat.download();
    const auto& temp = mat;
    return cout << temp;
}

std::ostream& lav::operator<<(std::ostream& cout, const Mat& mat)
{
    std::vector<float> vec;

    if (mat.uploaded)
    {
        vec.resize(mat.g_buffer.size());
        boc::copy(mat.g_buffer.begin(), mat.g_buffer.end(), vec.begin(), Mat::queue);
    }

    cout << std::setiosflags(std::ios::fixed) << std::setprecision(4) << "[";

    for (size_t i = 0; i < mat.rows; ++i)
    {
        cout << (i ? "  [ " : " [ ");

        for (size_t j = 0; j < mat.cols; ++j)
        {
            if (j != mat.cols - 1)
            {
                cout << std::setw(16) << std::setiosflags(std::ios::left);
            }

            cout << (mat.uploaded ? vec : mat.c_buffer)[i * mat.cols + j];
        }

        cout << " ]" << (i == mat.rows - 1 ? "" : "\n");
    }

    cout << " ]" << std::defaultfloat;

    return cout;
}

std::ofstream& lav::operator<<(std::ofstream& out, Mat& mat)
{
    mat.download();
    const auto& temp = mat;
    return out << temp;
}

std::ofstream& lav::operator<<(std::ofstream& out, const Mat& mat)
{
    std::vector<float> vec;

    if (mat.uploaded)
    {
        vec.resize(mat.g_buffer.size());
        boc::copy(mat.g_buffer.begin(), mat.g_buffer.end(), vec.begin(), Mat::queue);
    }

    for (size_t i = 0; i < mat.rows; ++i)
    {
        for (size_t j = 0; j < mat.cols; ++j)
        {
            auto data = (mat.uploaded ? vec : mat.c_buffer)[i * mat.cols + j];
            out << data << (j == mat.cols - 1 ? "" : ",");
        }

        out << std::endl;
    }

    return out;
}