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
 * File name     : lav_mat.h
 * Version       : 1.0
 * Last modified : 2020-4-10
 *
 * See https://github.com/rihothy/lav_mat to get source code.
 * ************************************************************************/

#ifndef _LAV_MAT_H_
#define _LAV_MAT_H_

#include <boost/compute.hpp>
#include <initializer_list>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>

const bool _DEFAULT_ON_VIDEO_RAM_ = false;//When the matrix is created, is the datas on RAM or on VRAM.

namespace lav
{
	class Mat
	{
	protected:

		bool uploaded = false;//Whether the datas is on the VRAM.

		std::vector<float> c_buffer;//Datas buffer on RAM.
		boost::compute::vector<float> g_buffer;//Datas buffer on VRAM.

	public:

		size_t rows;
		size_t cols;

		static boost::compute::event event;
		static boost::compute::device device;
		static boost::compute::context context;
		static boost::compute::command_queue queue;
		
		explicit Mat(const size_t& rows, const size_t& cols, const std::vector<float>& vec = {}, bool upload_flag = _DEFAULT_ON_VIDEO_RAM_);
		explicit Mat(const std::string& path, char delimiter = ' ', bool upload_flag = _DEFAULT_ON_VIDEO_RAM_);
		explicit Mat(const std::initializer_list<float>& vec, bool upload_flag = _DEFAULT_ON_VIDEO_RAM_);
		explicit Mat(const size_t& rows, const size_t& cols, bool upload_flag);
		Mat(Mat&& another) noexcept;
		Mat(const Mat& another);
		explicit Mat(void);

		Mat t(void);
		Mat t(void) const;
		float max(void);
		float max(void) const;
		Mat max(bool axis);
		Mat max(bool axis) const;
		float min(void);
		float min(void) const;
		Mat min(bool axis);
		Mat min(bool axis) const;
		float sum(void);
		float sum(void) const;
		Mat sum(bool axis);
		Mat sum(bool axis) const;
		float mean(void);
		float mean(void) const;
		Mat mean(bool axis);
		Mat mean(bool axis) const;
		Mat max_loc(bool axis);
		Mat max_loc(bool axis) const;
		Mat min_loc(bool axis);
		Mat min_loc(bool axis) const;

		friend Mat max(const float& th, Mat& mat);
		friend Mat max(const float& th, const Mat& mat);
		friend Mat max(Mat& mat, const float& th);
		friend Mat max(const Mat& mat, const float& th);
		friend Mat max(Mat& a, Mat& b);
		friend Mat max(Mat& a, const Mat& b);
		friend Mat max(const Mat& a, Mat& b);
		friend Mat max(const Mat& a, const Mat& b);
		friend Mat min(const float& th, Mat& mat);
		friend Mat min(const float& th, const Mat& mat);
		friend Mat min(Mat& mat, const float& th);
		friend Mat min(const Mat& mat, const float& th);
		friend Mat min(Mat& a, Mat& b);
		friend Mat min(Mat& a, const Mat& b);
		friend Mat min(const Mat& a, Mat& b);
		friend Mat min(const Mat& a, const Mat& b);

		friend Mat Eyes(const size_t& n, bool upload_flag);
		friend Mat Ones(const size_t& rows, const size_t& cols, bool upload_flag);
		friend Mat Zeros(const size_t& rows, const size_t& cols, bool upload_flag);
		friend Mat randn(const size_t& rows, const size_t& cols, bool upload_flag);
		friend Mat randu(const size_t& rows, const size_t& cols, bool upload_flag);
		friend Mat randn(const size_t& rows, const size_t& cols, float mean, float sigma, bool upload_flag);
		friend Mat randu(const size_t& rows, const size_t& cols, float lower, float upper, bool upload_flag);

		friend Mat mul(Mat& a, Mat& b, bool trans_a, bool trans_b);
		friend Mat mul(Mat& a, const Mat& b, bool trans_a, bool trans_b);
		friend Mat mul(const Mat& a, Mat& b, bool trans_a, bool trans_b);
		friend Mat mul(const Mat& a, const Mat& b, bool trans_a, bool trans_b);
		friend Mat operator-(Mat& mat);
		friend Mat operator-(const Mat& mat);
		friend Mat operator+(const float& th, Mat& mat);
		friend Mat operator+(const float& th, const Mat& mat);
		friend Mat operator+(Mat& mat, const float& th);
		friend Mat operator+(const Mat& mat, const float& th);
		friend Mat operator+(Mat& a, Mat& b);
		friend Mat operator+(Mat& a, const Mat& b);
		friend Mat operator+(const Mat& a, Mat& b);
		friend Mat operator+(const Mat& a, const Mat& b);
		friend Mat operator-(const float& th, Mat& mat);
		friend Mat operator-(const float& th, const Mat& mat);
		friend Mat operator-(Mat& mat, const float& th);
		friend Mat operator-(const Mat& mat, const float& th);
		friend Mat operator-(Mat& a, Mat& b);
		friend Mat operator-(Mat& a, const Mat& b);
		friend Mat operator-(const Mat& a, Mat& b);
		friend Mat operator-(const Mat& a, const Mat& b);
		friend Mat operator*(const float& th, Mat& mat);
		friend Mat operator*(const float& th, const Mat& mat);
		friend Mat operator*(Mat& mat, const float& th);
		friend Mat operator*(const Mat& mat, const float& th);
		friend Mat operator*(Mat& a, Mat& b);
		friend Mat operator*(Mat& a, const Mat& b);
		friend Mat operator*(const Mat& a, Mat& b);
		friend Mat operator*(const Mat& a, const Mat& b);
		friend Mat operator/(const float& th, Mat& mat);
		friend Mat operator/(const float& th, const Mat& mat);
		friend Mat operator/(Mat& mat, const float& th);
		friend Mat operator/(const Mat& mat, const float& th);
		friend Mat operator/(Mat& a, Mat& b);
		friend Mat operator/(Mat& a, const Mat& b);
		friend Mat operator/(const Mat& a, Mat& b);
		friend Mat operator/(const Mat& a, const Mat& b);
		friend Mat operator<(const float& th, Mat& mat);
		friend Mat operator<(const float& th, const Mat& mat);
		friend Mat operator<(Mat& mat, const float& th);
		friend Mat operator<(const Mat& mat, const float& th);
		friend Mat operator<(Mat& a, Mat& b);
		friend Mat operator<(Mat& a, const Mat& b);
		friend Mat operator<(const Mat& a, Mat& b);
		friend Mat operator<(const Mat& a, const Mat& b);
		friend Mat operator<=(const float& th, Mat& mat);
		friend Mat operator<=(const float& th, const Mat& mat);
		friend Mat operator<=(Mat& mat, const float& th);
		friend Mat operator<=(const Mat& mat, const float& th);
		friend Mat operator<=(Mat& a, Mat& b);
		friend Mat operator<=(Mat& a, const Mat& b);
		friend Mat operator<=(const Mat& a, Mat& b);
		friend Mat operator<=(const Mat& a, const Mat& b);
		friend Mat operator>(const float& th, Mat& mat);
		friend Mat operator>(const float& th, const Mat& mat);
		friend Mat operator>(Mat& mat, const float& th);
		friend Mat operator>(const Mat& mat, const float& th);
		friend Mat operator>(Mat& a, Mat& b);
		friend Mat operator>(Mat& a, const Mat& b);
		friend Mat operator>(const Mat& a, Mat& b);
		friend Mat operator>(const Mat& a, const Mat& b);
		friend Mat operator>=(const float& th, Mat& mat);
		friend Mat operator>=(const float& th, const Mat& mat);
		friend Mat operator>=(Mat& mat, const float& th);
		friend Mat operator>=(const Mat& mat, const float& th);
		friend Mat operator>=(Mat& a, Mat& b);
		friend Mat operator>=(Mat& a, const Mat& b);
		friend Mat operator>=(const Mat& a, Mat& b);
		friend Mat operator>=(const Mat& a, const Mat& b);
		friend Mat operator==(const float& th, Mat& mat);
		friend Mat operator==(const float& th, const Mat& mat);
		friend Mat operator==(Mat& mat, const float& th);
		friend Mat operator==(const Mat& mat, const float& th);
		friend Mat operator==(Mat& a, Mat& b);
		friend Mat operator==(Mat& a, const Mat& b);
		friend Mat operator==(const Mat& a, Mat& b);
		friend Mat operator==(const Mat& a, const Mat& b);
		friend Mat operator!=(const float& th, Mat& mat);
		friend Mat operator!=(const float& th, const Mat& mat);
		friend Mat operator!=(Mat& mat, const float& th);
		friend Mat operator!=(const Mat& mat, const float& th);
		friend Mat operator!=(Mat& a, Mat& b);
		friend Mat operator!=(Mat& a, const Mat& b);
		friend Mat operator!=(const Mat& a, Mat& b);
		friend Mat operator!=(const Mat& a, const Mat& b);

		Mat row(size_t row);
		Mat row(size_t row) const;
		Mat col(size_t col);
		Mat col(size_t col) const;
		Mat& reshape(size_t rows, size_t cols);
		void push_back(const Mat& another);
		void push_back(const std::initializer_list<float>& vec);
		//void push_right(const Mat& another);//Too fucking slow, don't use it.
		//void push_right(const std::initializer_list<float>& vec);//Too fucking slow, don't use it.

		Mat& operator=(Mat&& another) noexcept;
		Mat& operator=(const Mat& another);
		float& operator()(const size_t& row, const size_t& col);
		float operator()(const size_t& row, const size_t& col) const;
		Mat operator()(size_t first_row, size_t last_row, size_t first_col, size_t last_col);
		Mat operator()(size_t first_row, size_t last_row, size_t first_col, size_t last_col) const;

		friend std::ostream& operator<<(std::ostream& cout, Mat& mat);
		friend std::ostream& operator<<(std::ostream& cout, const Mat& mat);
		friend std::ofstream& operator<<(std::ofstream& out, Mat& mat);
		friend std::ofstream& operator<<(std::ofstream& out, const Mat& mat);

		friend Mat exp(const Mat& mat);
		friend Mat abs(const Mat& mat);
		friend Mat log(const Mat& mat);
		friend Mat log2(const Mat& mat);
		friend Mat log10(const Mat& mat);
		friend Mat sqrt(const Mat& mat);
		friend Mat pow(const Mat& mat, const float& th);

	protected:

		void upload(void);
		void download(void);

		static std::function<void(void)> init;

		template<typename T>
		static Mat unary_op(const Mat& mat, T&& op);

		template<typename T>
		static Mat binary_op(const Mat& a, const Mat& b, T&& op);
	};

	Mat Eyes(const size_t& n, bool upload_flag = _DEFAULT_ON_VIDEO_RAM_);
	Mat Ones(const size_t& rows, const size_t& cols, bool upload_flag = _DEFAULT_ON_VIDEO_RAM_);
	Mat Zeros(const size_t& rows, const size_t& cols, bool upload_flag = _DEFAULT_ON_VIDEO_RAM_);
	Mat randn(const size_t& rows, const size_t& cols, bool upload_flag = _DEFAULT_ON_VIDEO_RAM_);
	Mat randu(const size_t& rows, const size_t& cols, bool upload_flag = _DEFAULT_ON_VIDEO_RAM_);
	Mat randn(const size_t& rows, const size_t& cols, float mean, float sigma, bool upload_flag = _DEFAULT_ON_VIDEO_RAM_);
	Mat randu(const size_t& rows, const size_t& cols, float lower, float upper, bool upload_flag = _DEFAULT_ON_VIDEO_RAM_);

	Mat mul(Mat& a, Mat& b, bool trans_a = false, bool trans_b = false);
	Mat mul(Mat& a, const Mat& b, bool trans_a = false, bool trans_b = false);
	Mat mul(const Mat& a, Mat& b, bool trans_a = false, bool trans_b = false);
	Mat mul(const Mat& a, const Mat& b, bool trans_a = false, bool trans_b = false);
}

#include <lav_mat/src/operation.hpp>

#endif