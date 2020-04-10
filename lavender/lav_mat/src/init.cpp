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
 * File name     : init.cpp
 * Version       : 1.0
 * Last modified : 2020-4-10
 *
 * See https://github.com/rihothy/lav_mat to get source code.
 * ************************************************************************/

#include <lav_mat/lav_mat.h>

#include <algorithm>
#include <sstream>
#include <fstream>
#include <random>
#include <ctime>

using namespace lav;
namespace boc = boost::compute;

std::function<void(void)> Mat::init = []
{
    Mat::device = boc::system::default_device();
    Mat::context = boc::context(Mat::device);
    Mat::queue = boc::command_queue(Mat::context, Mat::device);
};

boc::event Mat::event;
boc::device Mat::device;
boc::context Mat::context;
boc::command_queue Mat::queue;

Mat::Mat(const size_t& rows, const size_t& cols, const std::vector<float>& vec, bool upload_flag) :
	rows(rows), cols(cols)
{
    Mat::init();
    Mat::init = [] {};

    uploaded = upload_flag;
    g_buffer = boc::vector<float>(Mat::context);

    if (!vec.empty() && vec.size() == rows * cols)
    {
        if (upload_flag)
        {
            g_buffer.assign(vec.begin(), vec.end(), Mat::queue);
        }
        else
        {
            c_buffer.assign(vec.begin(), vec.end());
        }
    }
    else if (vec.empty())
    {
        if (rows * cols)
        {
            if (upload_flag)
            {
                g_buffer.resize(rows * cols, Mat::queue);
            }
            else
            {
                c_buffer.resize(rows * cols);
            }
        }
    }
    else
    {
        throw std::runtime_error("Constructor: Initialization size is different from vector!");
    }
}

Mat::Mat(const std::string& path, char delimiter, bool upload_flag)
{
    std::ifstream istrm(path);

    if (istrm)
    {
        size_t rows = 0;
        size_t cols = 0;
        std::string line;
        std::vector<float> vec;

        while (std::getline(istrm, line) && line != "")
        {
            std::string data;
            size_t t_cols = 0;
            std::stringstream sstrm(line);

            while (std::getline(sstrm, data, delimiter))
            {
                ++t_cols;
                vec.push_back(atof(data.c_str()));
            }

            if (cols && t_cols != cols)
            {
                throw std::runtime_error("Constructor: File " + path + " is not a matrix!");
            }

            ++rows;
            cols = t_cols;
        }

        new (this)Mat(rows, cols, vec);
    }
    else
    {
        throw std::runtime_error("Constructor: Could not find file named " + path);
    }
}

Mat::Mat(const std::initializer_list<float>& vec, bool upload_flag) :
    Mat(1, vec.size(), vec, upload_flag)
{

}

Mat::Mat(const size_t& rows, const size_t& cols, bool upload_flag) :
    Mat(rows, cols, {}, upload_flag)
{

}

Mat::Mat(Mat&& another) noexcept :
    Mat(another.rows, another.cols)
{
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

Mat::Mat(const Mat& another) :
    Mat(another.rows, another.cols)
{
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

Mat::Mat(void) :
    Mat(0, 0)
{

}

Mat lav::Eyes(const size_t& n, bool upload_flag)
{
    std::vector<float> vec(n * n, 0);

    for (size_t i = 0; i < vec.size(); i += n + 1)
    {
        vec[i] = 1;
    }

    return Mat(n, n, vec, upload_flag);
}

Mat lav::Ones(const size_t& rows, const size_t& cols, bool upload_flag)
{
    return Mat(rows, cols, std::vector<float>(rows * cols, 1), upload_flag);
}

Mat lav::Zeros(const size_t& rows, const size_t& cols, bool upload_flag)
{
    return Mat(rows, cols, std::vector<float>(rows * cols, 0), upload_flag);
}

Mat lav::randn(const size_t& rows, const size_t& cols, bool upload_flag)
{
    return randn(rows, cols, 0, 1, upload_flag);
}

Mat lav::randu(const size_t& rows, const size_t& cols, bool upload_flag)
{
    return randu(rows, cols, 0, 1, upload_flag);
}

Mat lav::randn(const size_t& rows, const size_t& cols, float mean, float sigma, bool upload_flag)
{
    static std::default_random_engine e(time(nullptr));
    std::normal_distribution<float> rand_float(mean, sigma);

    Mat mat(rows, cols, upload_flag);

    std::vector<float> vec(rows * cols);
    std::generate(vec.begin(), vec.end(), [&] {return rand_float(e); });

    if (upload_flag)
    {
        mat.g_buffer.assign(vec.begin(), vec.end(), Mat::queue);
    }
    else
    {
        mat.c_buffer = std::move(vec);
    }

    return std::move(mat);
}

Mat lav::randu(const size_t& rows, const size_t& cols, float lower, float upper, bool upload_flag)
{
    static std::default_random_engine e(time(nullptr));
    std::uniform_real_distribution<float> rand_float(lower, upper);

    Mat mat(rows, cols, upload_flag);

    std::vector<float> vec(rows * cols);
    std::generate(vec.begin(), vec.end(), [&] {return rand_float(e); });

    if (upload_flag)
    {
        mat.g_buffer.assign(vec.begin(), vec.end(), Mat::queue);
    }
    else
    {
        mat.c_buffer = std::move(vec);
    }

    return std::move(mat);
}