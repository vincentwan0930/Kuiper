//
// Created by fss on 22-12-16.
//

#ifndef KUIPER_COURSE_INCLUDE_TENSOR_HPP_
#define KUIPER_COURSE_INCLUDE_TENSOR_HPP_
#include <memory>
#include <vector>
#include <armadillo>

// 将Tensor类定义在名为kuiper_infer的命名空间中。
namespace kuiper_infer {

// 定义一个类模板Tensor，可以接受不同的数据类型。
template<typename T>
class Tensor {

};

// 为uint8_t类型的数据进行类模板特化
template<>
class Tensor<uint8_t> {
  // 待实现
};

// 为float类型的数据进行类模板特化
template<>
class Tensor<float> {
 public:
  // 为 Tensor 类提供一个默认的构造函数。当你创建一个新的 Tensor 对象时，如果没有提供任何参数，
  // 这个默认构造函数会被调用。使用 = default 表示编译器会自动生成一个默认的构造函数实现
  // 此处Tensor 类的默认构造函数不会执行任何特殊的操作，因为类的成员变量 raw_shapes_ 和 data_ 
  // 都有自己的默认构造函数。所以，当你使用默认构造函数创建一个新的 Tensor 对象时，raw_shapes_ 和 data_ 会被自动初始化为默认值 
  explicit Tensor() = default;

  // 虽然不接收单实参，但加explicit还是能避免某些情况下的间接隐式调用。如：
  /*
  class A {
  public:
    A(uint32_t x, uint32_t y, uint32_t z) {}
  };

  class B {
  public:
    B(const A &a) {}
  };

  void foo(B b) {
    // ...
  }
  uint32_t a = 3, b = 4, c = 5;
  foo({a, b, c}); // 隐式地将 a、b、c 转换为 A 对象，然后将 A 对象转换为 B 对象 
  */
  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  // 拷贝构造函数：用已存在的对象初始化同类新对象。声明格式：类名 (const 类名&)；
  // 这里实现的是浅拷贝
  Tensor(const Tensor &tensor);

  // 拷贝赋值函数：用已存在的对象初始化同类已存在的对象。声明格式：类名& operator=(const 类名&)；  
  // 也是浅拷贝
  Tensor<float> &operator=(const Tensor &tensor);

  // rows()、cols()、channels()：分别返回Tensor对象的行数、列数和通道数。
  // 意这类“get成员”的函数，都应声明为const成员函数（即在函数头后加const）。表示该成员函数不会修改调用它的对象的的类成员变量
  uint32_t rows() const;

  uint32_t cols() const;

  uint32_t channels() const;

  // size()：返回Tensor对象的总元素（体素）个数。
  uint32_t size() const;

  // set_data()：用arma::fcube数据类型设置Tensor对象的数据。
  void set_data(const arma::fcube &data);

  bool empty() const;

  // 用偏移量（索引）访问tensor中的数据
  float index(uint32_t offset) const;

  // 返回tensor的形状
  std::vector<uint32_t> shapes() const;

  // 返回tensor的数据
  arma::fcube &data();

  const arma::fcube &data() const;

  // at()：根据通道、行和列索引获取Tensor对象中的元素值
  arma::fmat &at(uint32_t channel);

  const arma::fmat &at(uint32_t channel) const;

  float at(uint32_t channel, uint32_t row, uint32_t col) const;

  float &at(uint32_t channel, uint32_t row, uint32_t col);

  // 
  void Padding(const std::vector<uint32_t> &pads, float padding_value);

  //
  void Fill(float value);

  //
  void Fill(const std::vector<float> &values);

  void Ones();

  void Rand();

  void Show();

  // 将Tensor对象展平为一维数组。
  void Flatten();

 private:
  std::vector<uint32_t> raw_shapes_;
  arma::fcube data_;
};
}
#endif //KUIPER_COURSE_INCLUDE_TENSOR_HPP_
