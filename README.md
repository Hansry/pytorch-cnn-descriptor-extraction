# pytorch-cnn-descriptor-extraction
This code shows how to extract descriptor from CNN model

Requirement:
python == 3.54;
pytorch == 1.0.0;
torchvision == 0.1.80;
libtorch == 1.0.0;

Step:
1. First run the generate.py to generate the ***.pt model
2. Use test.cpp to test the generated ***.pt model (Note that you should compile the C++ pytorch library first.)

如果你电脑的GCC版本超过5.0，那么你需要从官网下载C++版本的libtorch源码重新编译,若GCC版本没有超过5.0,则可以直接下载官网编译好的C++版本libtorch的库。 或者从以下链接下载并编译：https://github.com/Hansry/pytorch
