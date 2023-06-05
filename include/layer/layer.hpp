//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_LAYER_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_LAYER_HPP_
#include <string>
#include "data/tensor.hpp"
namespace kuiper_infer {
class Layer {
 public:
  explicit Layer(const std::string &layer_name);

  // 作为虚函数的forward计算函数，等待具体派生的算子类来重写计算过程
  // 都接受两个vector<shared_ptr<自定义tensor类>>&，作为计算的输入和输出，在forward函数中修改输出的张量vector
  virtual void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs);
  // reluLayer中 inputs 等于 x , outputs 等于 y= x，if x>0
  // 计算得到的结果放在y当中，x是输入，放在inputs中

  virtual ~Layer() = default;
 private:
  // 用一个字符串成员属性标识每个layer
  std::string layer_name_; //relu layer "relu"
};
}
#endif //KUIPER_COURSE_INCLUDE_LAYER_LAYER_HPP_
