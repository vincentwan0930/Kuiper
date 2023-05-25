//
// Created by fss on 22-12-16.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <memory>

namespace kuiper_infer {

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, channels);
}

// 浅拷贝构造
Tensor<float>::Tensor(const Tensor &tensor) {
  this->data_ = tensor.data_;
  this->raw_shapes_ = tensor.raw_shapes_;
}

// 浅拷贝赋值
Tensor<float> &Tensor<float>::operator=(const Tensor &tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

// 注意这类“get成员”的函数，都应声明为const成员函数（即在函数头后加const）。表示该成员函数的函数体内不会修改调用它的对象的的类成员变量
uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

// 这行代码使用了 CHECK 宏，如果断言失败，会抛出一个 glog 的 FATAL 级别的错误，并终止程序的运行。另外，<< 运算符用于将错误信息和变量值拼接成一个字符串
void Tensor<float>::set_data(const arma::fcube &data) {
  CHECK(data.n_rows == this->data_.n_rows) << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols) << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices) << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

bool Tensor<float>::empty() const {
  return this->data_.empty();
}

float Tensor<float>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size());
  // 调用cube类的at(offset)，把cube当作一维vector，返回mem[offset]
  return this->data_.at(offset);
}

std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

arma::fcube &Tensor<float>::data() {
  return this->data_;
}

// 当tensor类对象被声明为const对象时，会调用const版本的data成员函数
const arma::fcube &Tensor<float>::data() const {
  return this->data_;
}

arma::fmat &Tensor<float>::at(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::fmat &Tensor<float>::at(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

void Tensor<float>::Padding(const std::vector<uint32_t> &pads, float padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  //todo 请把代码补充在这里1
  // 调用arma::cube类的成员函数：insert_rows(插入位置（第几行）, 插入行数)
  this->data_.insert_rows(this->data_.n_rows, pad_rows2);
  this->data_.insert_rows(0, pad_rows1);
  this->data_.insert_cols(this->data_.n_cols, pad_cols2);
  this->data_.insert_cols(0, pad_cols1);
  this->raw_shapes_ = this->shapes();
}

void Tensor<float>::Fill(float value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

// 用一个size==cude内体素个数的vector填充cube。注意：arma::cube是列主序的
void Tensor<float>::Fill(const std::vector<float> &values) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);

  const uint32_t rows = this->rows();
  const uint32_t cols = this->cols();
  const uint32_t planes = rows * cols;
  const uint32_t channels = this->data_.n_slices;

  //todo 请把代码补充在这里2
  // 法一：最慢方法。。。
  // for(uint ch=0, offset=0; ch<channels; ++ch){
  //   for(uint r=0; r<rows; ++r){
  //     for(uint c=0; c<cols; ++c){
  //       this->data_.at(r, c, ch) = values[offset++];
  //     }
  //   }
  // }
  // 法二：较快方法: 以slice为单位fill，用arma::fmat(values.data()+ch*planes, rows, cols)可直接用参1开始的一片连续内存空间初始化一个mat对象，再把它赋给cude的一个slice
  // std::vector::data() 是STL的vector的成员函数，它返回一个指向内存数组的直接指针，该内存数组由向量内部用于存储其拥有的元素。
  // 这里调用的是arma::mat的这个构造函数：inline Mat(const eT* aux_mem, const uword aux_n_rows, const uword aux_n_cols);
  // for(uint ch=0; ch<channels; ++ch){
  //   arma::fmat &channel_data = this->data_.slice(ch);
  //   const arma::fmat &channel_data_t = arma::fmat(values.data()+ch*planes, rows, cols);
  //   channel_data = channel_data_t.t();
  // }

  // 自创法三：利用arma::cube的构造函数，可以把参1开始的一片连续内存空间用来初始化一个cube对象，但其中的mat默认是列主序的，因此还需要把每个slice转置
  this->data_ = arma::fcube(values.data(), rows, cols, channels);
  for(uint ch=0; ch<channels; ++ch){
    this->data_.slice(ch) = this->data_.slice(ch).t();
  }
}

void Tensor<float>::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.slice(i);
  }
}

void Tensor<float>::Flatten() {
  CHECK(!this->data_.empty());
  const uint32_t size = this->data_.size();
  // 用一个shape为(体素数，1,1)的arma::cube作为flatten后的结果
  arma::fcube linear_cube(size, 1, 1);

  uint32_t channel = this->channels();
  uint32_t rows = this->rows();
  uint32_t cols = this->cols();
  uint32_t index = 0;

  for (uint32_t c = 0; c < channel; ++c) {
    // arma::cube对象.slice(indx)可直接用来初始化arma::matrix对象
    const arma::fmat &matrix = this->data_.slice(c);

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c_ = 0; c_ < cols; ++c_) {
        linear_cube.at(index, 0, 0) = matrix.at(r, c_);
        index += 1;
      }
    }
  }
  CHECK_EQ(index, size);
  this->data_ = linear_cube;
  this->raw_shapes_ = std::vector<uint32_t>{size};
}

void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

void Tensor<float>::Ones() {
  CHECK(!this->data_.empty());
  this->data_.fill(1.);
}
}
