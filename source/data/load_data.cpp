//
// Created by fss on 22-12-19.
//
#include "data/load_data.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
// 函数作用：读取csv内容，并保存为自定义tensor类对象并返回
// std::shared_ptr<Tensor<float >> 数据类型为指向Tensor<float>对象的shared_ptr智能指针
// 多个shared_ptr指针指向同一变量时可以共享对象所有权。当最后一个共享指针被销毁或者被重新分配时，它所指向的对象将被自动删除。这有助于避免悬空指针和内存泄漏的问题
std::shared_ptr<Tensor<float >> CSVDataLoader::LoadDataWithHeader(const std::string &file_path,
                                                                  std::vector<std::string> &headers,
                                                                  char split_char) {
  CHECK(!file_path.empty()) << "File path is empty!";
  // 创建一个输入文件流对象 in，用于从文件 file_path 中读取数据。
  // std::ifstream 是 C++ 标准库中的一个类，用于处理文件输入操作。
  std::ifstream in(file_path);
  CHECK(in.is_open() && in.good()) << "File open failed! " << file_path;

  // 用于把csv的ifstream流对象中的每行数据搬到sstream对象中
  // sstream对象中的（一行数据）string再以split_char为分隔符，取出csv矩阵中的每个元素
  std::string line_str;
  std::stringstream line_stream;

  // 将csv文件的ifstream文件流对象in传入自定义函数，获取CSV内的矩阵的行数、最大列数
  const auto &[rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
  CHECK(rows >= 1);
  // 初始化对应csv尺寸的自定义tensor对象，用于存储csv内容（矩阵）。并用智能指针指向它
  std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(1, rows - 1, cols);
  // 用&引用，使data与input_tensor第一个通道的矩阵绑定。这样修改一方后，另一方也会一起修改（其实是一体，用一块内存）
  arma::fmat &data = input_tensor->at(0);

  size_t row = 0;
  while (in.good()) {
    std::getline(in, line_str);
    if (line_str.empty()) {
      break;
    }

    std::string token;
    line_stream.clear();
    line_stream.str(line_str);

    size_t col = 0;
    while (line_stream.good()) {
      std::getline(line_stream, token, split_char);
      try {
        //todo 补充
        // 读取到第一行的csv列名，并存放在传入的引用参数vector<string>&headers中
        if (row==0) {
          headers.push_back(token);
        }
        else { // 读取到第二行之后的csv数据，并相应放置在data变量的row，col位置中
          // stof(str1)用于将字符串转换为浮点数。
          // 这里将从CSV文件中读取的单个元素(string)转换为浮点数，并将其存储在arma::mat data的row，col位置中 
          data.at(row-1, col) = std::stof(token);
        }
      }
      catch (std::exception &e) {
        LOG(ERROR) << "Parse CSV File meet error: " << e.what();
        continue;
      }
      col += 1;
      CHECK(col <= cols) << "There are excessive elements on the column";
    }
    row += 1;
    CHECK(row <= rows) << "There are excessive elements on the row";
  }
  return input_tensor;
}

std::shared_ptr<Tensor<float >> CSVDataLoader::LoadData(const std::string &file_path, char split_char) {
  CHECK(!file_path.empty()) << "File path is empty!";
  std::ifstream in(file_path);
  CHECK(in.is_open() && in.good()) << "File open failed! " << file_path;

  std::string line_str;
  std::stringstream line_stream;

  const auto &[rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
  std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(1, rows, cols);
  arma::fmat &data = input_tensor->at(0);

  size_t row = 0;
  while (in.good()) {
    std::getline(in, line_str);
    if (line_str.empty()) {
      break;
    }

    std::string token;
    line_stream.clear();
    line_stream.str(line_str);

    size_t col = 0;
    while (line_stream.good()) {
      std::getline(line_stream, token, split_char);
      try {
        data.at(row, col) = std::stof(token);
      }
      catch (std::exception &e) {
        LOG(ERROR) << "Parse CSV File meet error: " << e.what();
        continue;
      }
      col += 1;
      CHECK(col <= cols) << "There are excessive elements on the column";
    }

    row += 1;
    CHECK(row <= rows) << "There are excessive elements on the row";
  }
  return input_tensor;
}

std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream &file, char split_char) {
  bool load_ok = file.good();
  // clear() 是 std::ifstream 类的一个成员函数，用于重置文件流的状态标志。
  // 此处file.clear() 可清除文件流 file 的任何错误状态，以便后续操作正常进行。
  file.clear();
  // 用于统计csv内的行数、所有行的最大列数（作为矩阵列数）
  size_t fn_rows = 0;
  size_t fn_cols = 0;
  // tellg() 是 std::ifstream 类的一个成员函数，用于获取文件流中当前的读取位置。
  // std::ifstream::pos_type 是 std::ifstream 类中用using或typedef声明的一个类型（实际为std::streampos类型），用于表示文件流中的位置信息
  // 此处获取文件流 file 当前的读取位置，存于std::ifstream::pos_type类型的start_pos中
  // 后面调用 file.seekg(start_pos); 可以将文件流的读取位置恢复到 start_pos，从而实现对文件内容的重新读取。
  const std::ifstream::pos_type start_pos = file.tellg();

  std::string token;
  std::string line_str;
  std::stringstream line_stream;

  // file.good()是一个文件流对象的成员函数，用于判断文件流是否处于可读取状态。
  // 当文件流读取到文件末尾时，file.good()会返回false
  while (file.good() && load_ok) {
    // getline(输入流对象, string变量, 分隔符) 用于从输入流中读取一段/行数据并存储在string变量中 
    // 分隔符默认为换行符\n，所以默认读取一行数据
    std::getline(file, line_str);
    if (line_str.empty()) {
      break;
    }

    // 注意：stringstream的clear()成员函数与ifstream的clear()成员函数的作用不同。
    // ifstream的clear()函数用于清除文件流的任何错误状态，以便后续操作可以正常进行。
    // stringstream的clear()函数用于重置字符串流的读写状态标志，每次读写后都要clear才能再次读写！！！
    // 此处 line_stream.clear()函数用于重置line_stream的状态标志，以便在下一次循环中正确读取下一行的数据 
    // 注意：clear() 方法只重置了stringstream的状态标志，并没有清空数据。如果需要清空数据，可以用：ss对象.str("")
    line_stream.clear();
    // 相当于line_stream << line_str;   line_stream.str();  但下面这句还会重置sstream的状态标志
    line_stream.str(line_str);
    size_t line_cols = 0;

    std::string row_token;
    while (line_stream.good()) {
      std::getline(line_stream, row_token, split_char);
      ++line_cols;
    }
    if (line_cols > fn_cols) {
      fn_cols = line_cols;
    }

    ++fn_rows;
  }
  file.clear();
  // 将ifstream类型的file文件流的读取位置恢复到打开file时保存的start_pos处
  file.seekg(start_pos);
  // 这样可以返回 pair<,> 类型
  return {fn_rows, fn_cols};
}
}
