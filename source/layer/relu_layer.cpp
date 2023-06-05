//
// Created by fss on 22-12-20.
//
#include <glog/logging.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
ReluLayer::ReluLayer(const std::shared_ptr<Operator> &op) : Layer("Relu") {
  CHECK(op->op_type_ == OpType::kOperatorRelu) << "Operator has a wrong type: " << int(op->op_type_);
  // dynamic_cast是什么意思？ 就是判断一下op指针是不是指向一个relu_op类的指针
  // 这边的op不是ReluOperator类型的指针，就报错
  // 我们这里只接受ReluOperator类型的指针
  // 父类指针必须指向子类ReluOperator类型的指针
  // 为什么不讲构造函数设置为const std::shared_ptr<ReluOperator> &op？
  // 为了接口统一，具体下节会说到
  ReluOperator *relu_op = dynamic_cast<ReluOperator *>(op.get());

  CHECK(relu_op != nullptr) << "Relu operator is empty";
  // 一个op实例和一个layer 一一对应 这里relu op对一个relu layer
  // 注意：这里用make_unique<>重新创建了一个relu_op（即参数op）所指对象的副本
  this->op_ = std::make_unique<ReluOperator>(relu_op->get_thresh());
}

void ReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                         std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  // relu 操作在哪里，这里！
  // 我需要该节点信息的时候 直接这么做
  // 实行了属性存储和运算过程的分离！！！！！！！！！！！！！！！！！！！！！！！！
  //x就是inputs y = outputs
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorRelu);

  const uint32_t batch_size = inputs.size(); //一批x，放在vec当中，理解为batchsize数量的tensor，需要进行relu操作
  for (int i = 0; i < batch_size; ++i) {

    CHECK(!inputs.at(i)->empty());
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i); //取出批次当中的一个张量

    //这里的transform是arma库的fcube类的成员函数，用于对张量fcube中的每一个元素进行transform函数的参数定义的运算
    //注意：这个transform函数的参数是一个函数对象，并且是lambda表达式形式的函数对象
    //lambda表达式的语法为：[捕获列表](函数参数){函数体}
    // [&]：以引用的方式捕获所有外部变量。
    // [=]：以值的方式捕获所有外部变量。
    // [x, &y]：以值的方式捕获变量x，以引用的方式捕获变量y。
    // [&a, &b, &c]：以引用的方式捕获变量a、b、c
    //这里的lambda表达式捕获所有外部变量的引用，从而捕获reluop的指针op_来获取计算所需的阈值thresh。
    //并且接收float为参数，返回relu计算后的float。即：将调用transform的fcube对象的每一个float体素进行relu计算
    input_data->data().transform([&](float value) {
      // 对张良中的没一个元素进行运算
      // 从operator中得到存储的属性
      float thresh = op_->get_thresh();
      //x >= thresh
      if (value >= thresh) {
        return value; // return x
      } else {
        // x<= thresh return 0.f;
        return 0.f;
      }
    });

    // 把结果y放在outputs中
    outputs.push_back(input_data);
  }
}

std::shared_ptr<Layer> ReluLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
  std::shared_ptr<Layer> relu_layer = std::make_shared<ReluLayer>(op);
  return relu_layer;
}

LayerRegistererWrapper kReluLayer(OpType::kOperatorRelu, ReluLayer::CreateInstance);
}