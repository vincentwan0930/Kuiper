//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
namespace kuiper_infer {
// 用一个枚举类登记算子的类型，-1表示未知类型，0这里注册为了relu。后续的算子注册为1,2...
enum class OpType {
  kOperatorUnknown = -1,
  kOperatorRelu = 0,
};

// 算子类的基类Operator
class Operator {
 public:
  OpType op_type_ = OpType::kOperatorUnknown; //不是一个具体节点 制定为unknown

  virtual ~Operator() = default; // 基类的虚析构

  // op类的构造函数接收用于表示算子类型的枚举类变量，并初始化自身的op_type_成员属性
  explicit Operator(OpType op_type);
};


}
#endif //KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
