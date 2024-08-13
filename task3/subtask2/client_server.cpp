#include <fstream>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <cassert>
#include <iostream>

#define TYPE double
#define EPSILON 1e-5

template <typename T>
T fun_sin(T arg) {
    return std::sin(arg);
}

template <typename T>
T fun_sqrt(T arg) {
    return std::sqrt(arg);
}

template <typename T>
T fun_pow(T x, T y) {
    return std::pow(x, y);
}

template <typename T>
class Server {
    std::queue<std::packaged_task<T()>> tasks_queue;
    std::unordered_map<size_t, T> results;
    std::unique_ptr<std::jthread> server_thread;
    std::mutex task_mutex;
    std::mutex result_mutex;
    std::condition_variable cv;
    size_t current_task_id = 0;
    size_t current_result_id = 0;
public:
    void start() {
        server_thread = std::make_unique<std::jthread>(&Server::work, this);
    }

    void stop() {
        server_thread.get_deleter()(server_thread.release());
    }

    size_t add_task(std::packaged_task<T()>& task) {
        std::lock_guard<std::mutex> lock(task_mutex);
        tasks_queue.push(std::move(task));
        current_task_id++;
        return current_task_id;
    }

    T request_result(size_t id) {
        std::unique_lock<std::mutex> lock(result_mutex);
        cv.wait(lock, [this, id]{return results.find(id) != results.end();});
        return results[id];
    }
private:
    void work(const std::stop_token& stop_token) {
        std::packaged_task<T()> task;
        while (!stop_token.stop_requested()) {
            if (!tasks_queue.empty()) {
                {
                    std::lock_guard<std::mutex> lock(task_mutex);
                    task = std::move(tasks_queue.front());
                    tasks_queue.pop();
                }
                auto future = task.get_future();
                task();
                T value = future.get();
                {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    ++current_result_id;
                    results.insert({current_result_id, value});
                }
                cv.notify_all();
            }
        }
    }
};

template <typename T>
void client_simple_work(const std::string& filename_for_args, const std::string& filename_for_result, T(*f)(T), Server<T>& server) {
    std::ifstream read_stream(filename_for_args, std::ios::in);
    std::ofstream write_stream(filename_for_result, std::ios::out);
    while (!read_stream.eof()) {
        T arg;
        read_stream >> arg;
        std::packaged_task<T()> task(std::bind(f, arg));
        size_t id = server.add_task(task);
        T result = server.request_result(id);
        write_stream << std::fixed << result << '\n';
    }
}

template <typename T>
void client_hard_work(const std::string& filename_for_args, const std::string& filename_for_result, T(*f)(T, T), Server<T>& server) {
    std::ifstream read_stream(filename_for_args, std::ios::in);
    std::ofstream write_stream(filename_for_result, std::ios::out);
    while (!read_stream.eof()) {
        T arg1;
        T arg2;
        read_stream >> arg1;
        read_stream >> arg2;
        std::packaged_task<T()> task(std::bind(f, arg1, arg2));
        size_t id = server.add_task(task);
        T result = server.request_result(id);
        write_stream << std::fixed << result << '\n';
    }
}

void make_tasks(const std::string& filename_for_args, size_t N, size_t num_args, std::vector<double> max_values) {
    std::ofstream stream(filename_for_args);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < num_args; ++j) {
            TYPE arg = ((double)rand() / RAND_MAX) * max_values[j];
            stream << arg << ' ';
        }
        stream << '\n';
    }
}

template <typename T>
void simple_test(const std::string& filename_for_args, const std::string& filename_for_result, T(*function)(T)) {
    std::ifstream args_stream(filename_for_args, std::ios::in);
    std::ifstream res_stream(filename_for_result, std::ios::in);
    T arg;
    T result;
    while (not args_stream.eof()) {
        args_stream >> arg;
        res_stream >> result;
        assert(fabs(function(arg) - result) < EPSILON);
    }
    std::cout << "All tests passed!" << std::endl;
}

template <typename T>
void hard_test(const std::string& filename_for_args, const std::string& filename_for_result, T(*function)(T, T)) {
    std::ifstream args_stream(filename_for_args, std::ios::in);
    std::ifstream res_stream(filename_for_result, std::ios::in);
    T arg1, arg2;
    T result;
    while (not args_stream.eof()) {
        args_stream >> arg1;
        args_stream >> arg2;
        res_stream >> result;
        assert(fabs(function(arg1, arg2) - result) < EPSILON);
    }
    std::cout << "All tests passed!" << std::endl;
}

int main() {
    size_t num_tasks = 10000;

    std::string filename_for_args_sin("sin_args.txt");
    std::string filename_for_args_sqrt("sqrt_args.txt");
    std::string filename_for_args_pow("pow_args.txt");

    std::string filename_for_result_sin("sin_res.txt");
    std::string filename_for_result_sqrt("sqrt_res.txt");
    std::string filename_for_result_pow("pow_res.txt");

    make_tasks(filename_for_args_sin, num_tasks, 1, {7});
    make_tasks(filename_for_args_sqrt, num_tasks, 1, {10000});
    make_tasks(filename_for_args_pow, num_tasks, 2, {1000, 3});

    Server<TYPE> server;
    server.start();

    std::thread client1_thread{client_simple_work<TYPE>, std::ref(filename_for_args_sin), std::ref(filename_for_result_sin), fun_sin<TYPE>, std::ref(server)};
    std::thread client2_thread{client_simple_work<TYPE>, std::ref(filename_for_args_sqrt), std::ref(filename_for_result_sqrt), fun_sqrt<TYPE>, std::ref(server)};
    std::thread client3_thread{client_hard_work<TYPE>, std::ref(filename_for_args_pow), std::ref(filename_for_result_pow), fun_pow<TYPE>, std::ref(server)};

    client1_thread.join();
    client2_thread.join();
    client3_thread.join();

    server.stop();

    simple_test<TYPE>(filename_for_args_sin, filename_for_result_sin, fun_sin<TYPE>);
    simple_test<TYPE>(filename_for_args_sqrt, filename_for_result_sqrt, fun_sqrt<TYPE>);
    hard_test<TYPE>(filename_for_args_pow, filename_for_result_pow, fun_pow<TYPE>);
}
