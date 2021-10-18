#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <deque>
#include <functional>

std::vector<std::thread> threads;

typedef void (*LoopFn)(void*);

void* CreateThread(void* loop_fn, void* param) {
	std::cout << "creating thread" << std::endl;
	threads.push_back(std::thread((LoopFn)loop_fn, param));
	return (void*)threads.back().native_handle();
}

void JoinThread(void* handle) {
	for (auto& t: threads) {
		if ((void*)t.native_handle() == handle) {
			std::cout << "joining thread"<< std::endl;
			t.join();
		}
	}
}

void TestAdd() {
	std::cout << "test thread pool hooks on add." << std::endl;
	Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
	Ort::SessionOptions session_options;
	session_options.SetCreateThreadFn(CreateThread);
	session_options.SetJoinThreadFn(JoinThread);
	Ort::Session session{ env, L"D:\\issue\\CustomThreadPool\\model\\model.onnx", session_options };
	std::cout << "loaded" << std::endl;
	const char* input_names[] = {"X", "Y"};
    Ort::AllocatorWithDefaultOptions allocator_info;
	constexpr int dim = 1024;
    int32_t ints[dim];
	for (int i = 0; i < dim; ++i) ints[i] = 1;
	int64_t shape[] = {dim};
    const char* output_names[] = {"Z"};
	Ort::Value input_tensors[] = {Ort::Value::CreateTensor<int32_t>(allocator_info.GetInfo(), ints, dim, shape, 1), 
	                              Ort::Value::CreateTensor<int32_t>(allocator_info.GetInfo(), ints, dim, shape, 1)};
	Ort::Value output_tensors[] = {Ort::Value::CreateTensor<int32_t>(allocator_info.GetInfo(), ints, dim, shape, 1)};
	session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 2, output_names, output_tensors, 1);
	const int32_t* output_data = output_tensors[0].GetTensorData<int32_t>();
	for (int i = 0; i < 3; ++i) {
		std::cout << output_data[i] << ", ";
	}
	std::cout << "..." << std::endl;
	std::cout << "done" << std::endl;
}

void TestPGAN() {
	std::cout << "test thread pool hooks on PGAN model." << std::endl;
	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };
	Ort::SessionOptions session_options;
	session_options.SetCreateThreadFn(CreateThread);
	session_options.SetJoinThreadFn(JoinThread);
	Ort::Session session{ env, L"D:\\issue\\PerfInvestigation\\pgan\\PGAN_NetG_model.onnx", session_options };
	std::cout << "loaded" << std::endl;
	const char* input_names[] = { "0" };
	Ort::AllocatorWithDefaultOptions allocator_info;
	constexpr int input_dim = 2*2*16*16;
	float* input_floats = new float[input_dim];
	int64_t input_shape[] = { 2,2,16,16 };
	Ort::Value input_tensors[] = { Ort::Value::CreateTensor<float>(allocator_info.GetInfo(), input_floats, input_dim, input_shape, 4) };

	const char* output_names[] = { "566" };
	constexpr int output_dim = 2 * 3 * 256 * 256;
	float* output_floats = new float[output_dim];
	int64_t output_shape[] = {2, 3, 256, 256};
	Ort::Value output_tensors[] = { Ort::Value::CreateTensor<float>(allocator_info.GetInfo(), output_floats, output_dim, output_shape, 4) };

	session.Run(Ort::RunOptions{ nullptr }, input_names, input_tensors, 1, output_names, output_tensors, 1);

	const float* output_data = output_tensors[0].GetTensorData<float>();
	for (int i = 0; i < 3; ++i) {
		std::cout << output_data[i] << ", ";
	}
	std::cout << "..." << std::endl;
	std::cout << "done" << std::endl;
	delete[] input_floats;
	delete[] output_floats;
}

int main() {
	//TestAdd();
	TestPGAN();
}