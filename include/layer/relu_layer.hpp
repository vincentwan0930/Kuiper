//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#include "layer.hpp"
#include "ops/relu_op.hpp"

namespace kuiper_infer {
class ReluLayer : public Layer {
 public:
  ~ReluLayer() override = default;

  // 通过这里，把relu_op中的thresh参数告知给relu layer, 因为计算的时候要用到
  explicit ReluLayer(const std::shared_ptr<Operator> &op);

  // 执行relu 操作的具体函数Forwards
  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  // 下节的内容，不用管
  static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

 private:
  // relu_layer拥有一个relu_op成员指针，这样才能在计算时获取relu的阈值参赛thresh
  std::unique_ptr<ReluOperator> op_;
};
}
#endif //KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
