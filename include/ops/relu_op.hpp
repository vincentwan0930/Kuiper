//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_OPS_RELU_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_RELU_OP_HPP_
#include "op.hpp"
namespace kuiper_infer {
class ReluOperator : public Operator {
 public:
 // 声明为override会告诉编译器这是一个虚函数，编译器也会来检查到底是不是
  ~ReluOperator() override = default;

  // 派生类构造函数的初始化列表中须初始化基类对象：基类(基类::基类构造参数)
  explicit ReluOperator(float thresh);

  void set_thresh(float thresh);

  float get_thresh() const;

 private:
  // 需要传递到reluLayer中，怎么传递？？？ =>
  // reluLayer类有一个unique_ptr<ReluOperator>类的成员指针，指向reluOp对象，用该指针即可获取reluOp的thresh
  float thresh_ = 0.f; // 用于过滤tensor<float>值当中大于thresh的部分
  // reluOp有其特有的成员属性thresh
  // stride padding kernel_size 这些是到时候convOperator需要的
  // operator起到了属性存储、变量的作用
  // operator所有子类不负责具体运算
  // 具体运算由另外一个类Layer类负责
  // y =x  , if x >=0 y = 0 if x < 0

};
}
#endif //KUIPER_COURSE_INCLUDE_OPS_RELU_OP_HPP_
