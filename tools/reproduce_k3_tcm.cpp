#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

bool guard_enabled = true;
volatile sig_atomic_t tcm_mappings = 0;
volatile sig_atomic_t fault_reported = 0;

void append_text(char*& out, const char* text) {
    while (*text) *out++ = *text++;
}

void append_number(char*& out, uintptr_t value, unsigned base) {
    char digits[2 * sizeof(value) + 1];
    const char* alphabet = "0123456789abcdef";
    size_t count = 0;
    do {
        digits[count++] = alphabet[value % base];
        value /= base;
    } while (value);
    while (count) *out++ = digits[--count];
}

void fault_handler(int signal_number, siginfo_t* info, void*) {
    if (__sync_lock_test_and_set(&fault_reported, 1)) {
        for (;;) pause();
    }
    char message[160];
    char* out = message;
    append_text(out, "FAULT signal=");
    append_number(out, static_cast<unsigned>(signal_number), 10);
    append_text(out, " code=");
    append_number(out, static_cast<unsigned>(info->si_code), 10);
    append_text(out, " address=0x");
    append_number(out, reinterpret_cast<uintptr_t>(info->si_addr), 16);
    append_text(out, " tcm_mappings=");
    append_number(out, static_cast<unsigned>(tcm_mappings), 10);
    append_text(out, "\nCompare address with the preceding TCM_GUARD end values.\n");
    const size_t length = static_cast<size_t>(out - message);
    ssize_t ignored = write(STDERR_FILENO, message, length);
    (void)ignored;
    _Exit(128 + signal_number);
}

int positive_integer(const char* text, const char* option) {
    try {
        const int value = std::stoi(text);
        if (value < 1) throw std::invalid_argument(option);
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(option) + " requires a positive integer");
    }
}

void set_host_affinity() {
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    for (int cpu = 0; cpu < 8; ++cpu) CPU_SET(cpu, &cpus);
    if (sched_setaffinity(0, sizeof(cpus), &cpus) != 0) {
        std::cerr << "WARN sched_setaffinity: " << std::strerror(errno) << '\n';
    }
}

int inference(const std::string& model_path, int threads, bool guarded) {
    guard_enabled = guarded;
    tcm_mappings = 0;
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    set_host_affinity();
    struct sigaction action {};
    action.sa_sigaction = fault_handler;
    action.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGSEGV, &action, nullptr) != 0) {
        std::cerr << "ERROR sigaction: " << std::strerror(errno) << '\n';
        return 2;
    }
    const rlimit no_core {0, 0};
    if (setrlimit(RLIMIT_CORE, &no_core) != 0) {
        std::cerr << "WARN setrlimit: " << std::strerror(errno) << '\n';
    }
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "k3_tcm_repro");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        std::unordered_map<std::string, std::string> provider_options{
            {"SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(threads)},
        };
        auto status = Ort::SessionOptionsSpaceMITEnvInit(options, provider_options);
        if (!status.IsOK()) throw std::runtime_error(status.GetErrorMessage());
        Ort::Session session(env, model_path.c_str(), options);
        if (session.GetInputCount() != 1) throw std::runtime_error("expected one model input");

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto input_type_info = session.GetInputTypeInfo(0);
        auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
        const auto input_type = input_info.GetElementType();
        std::cout << "INPUT name=" << input_name.get()
                  << " type=" << static_cast<int>(input_type) << std::endl;
        size_t element_size = 0;
        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) element_size = sizeof(float);
        else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) element_size = sizeof(uint16_t);
        else throw std::runtime_error("unsupported model input type " + std::to_string(static_cast<int>(input_type)));
        const auto shape = input_info.GetShape();
        size_t elements = 1;
        for (int64_t dimension : shape) {
            if (dimension <= 0) throw std::runtime_error("this repro requires static input dimensions");
            elements *= static_cast<size_t>(dimension);
        }
        std::vector<uint8_t> pixels(elements * element_size, 0);
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor(memory, pixels.data(), pixels.size(),
                                              shape.data(), shape.size(), input_type);
        std::vector<std::string> output_names;
        std::vector<const char*> output_name_ptrs;
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_names.emplace_back(name.get());
        }
        for (const auto& name : output_names) output_name_ptrs.push_back(name.c_str());
        const char* input_names[] = {input_name.get()};
        Ort::RunOptions run_options;
        std::cout << "RUN mode=" << (guarded ? "guard" : "baseline")
                  << " threads=" << threads << " model=" << model_path << std::endl;
        auto outputs = session.Run(run_options, input_names, &input, 1,
                                   output_name_ptrs.data(), output_name_ptrs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto output_shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "OUTPUT " << output_names[i] << " shape=[";
            for (size_t j = 0; j < output_shape.size(); ++j) {
                if (j) std::cout << ',';
                std::cout << output_shape[j];
            }
            std::cout << "]\n";
        }
        std::cout << "INFERENCE_OK mode=" << (guarded ? "guard" : "baseline")
                  << " tcm_mappings=" << tcm_mappings << std::endl;
        if (guarded && tcm_mappings == 0) {
            std::cerr << "ERROR no guarded /dev/tcm mapping was intercepted; "
                         "check that the binary was linked with -Wl,--export-dynamic.\n";
            return 3;
        }
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ORT_ERROR " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 2;
    }
}

std::string executable_path() {
    std::vector<char> buffer(4096);
    const ssize_t size = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (size < 0) throw std::runtime_error("cannot read /proc/self/exe");
    buffer[static_cast<size_t>(size)] = '\0';
    return buffer.data();
}

int run_child(const std::string& executable, const std::string& model, int threads,
              bool guarded, int run) {
    const std::string thread_count = std::to_string(threads);
    const char* mode = guarded ? "guard" : "baseline";
    std::cout << "START mode=" << mode << " run=" << run << std::endl;
    const pid_t child = fork();
    if (child < 0) throw std::runtime_error(std::string("fork: ") + std::strerror(errno));
    if (child == 0) {
        execl(executable.c_str(), executable.c_str(), "--child", mode,
              "--model", model.c_str(), "--threads", thread_count.c_str(), nullptr);
        _Exit(127);
    }
    int status = 0;
    if (waitpid(child, &status, 0) < 0) {
        throw std::runtime_error(std::string("waitpid: ") + std::strerror(errno));
    }
    if (WIFSIGNALED(status)) {
        const int signal_number = WTERMSIG(status);
        std::cout << "RESULT mode=" << mode << " run=" << run
                  << " signal=" << signal_number << std::endl;
        return 128 + signal_number;
    }
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : 125;
    std::cout << "RESULT mode=" << mode << " run=" << run << " exit=" << code << std::endl;
    return code;
}

void usage(const char* program) {
    std::cout << "Build on K3:\n"
              << "  c++ -std=c++17 -O2 -pthread -Wl,--export-dynamic " << program
              << " -o reproduce_k3_tcm -lonnxruntime -lspacemit_ep -ldl\n\n"
              << "Run from the project directory:\n"
              << "  ./reproduce_k3_tcm --model ../data/best.spacemit.fp16.onnx\n"
              << "  ./reproduce_k3_tcm --model ../data/best.spacemit.fp16.onnx --guard-only --runs 1\n\n"
              << "Options: --model PATH --threads N --runs N --guard-only --help\n"
              << "Default: one unguarded run and one guarded run; EP threads=8.\n"
              << "No OpenCV, video, project binary, external probe or shell script is used.\n";
}

}  // namespace

extern "C" __attribute__((visibility("default")))
void* mmap(void* address, size_t length, int protection, int flags, int fd, off_t offset) {
    using Mmap = void* (*)(void*, size_t, int, int, int, off_t);
    static Mmap next_mmap = nullptr;
    if (!next_mmap) next_mmap = reinterpret_cast<Mmap>(dlsym(RTLD_NEXT, "mmap"));
    if (!next_mmap) {
        errno = ENOSYS;
        return MAP_FAILED;
    }
    if (guard_enabled && fd >= 0 && length == 0x60000 && !(flags & MAP_FIXED)) {
        char descriptor[64];
        char target[256];
        snprintf(descriptor, sizeof(descriptor), "/proc/self/fd/%d", fd);
        const ssize_t size = readlink(descriptor, target, sizeof(target) - 1);
        if (size > 0) {
            target[size] = '\0';
            if (std::strcmp(target, "/dev/tcm") == 0) {
                const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
                void* reservation = next_mmap(nullptr, length + 2 * page, PROT_NONE,
                                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (reservation == MAP_FAILED) return reservation;
                void* mapped = next_mmap(static_cast<char*>(reservation) + page, length,
                                         protection, flags | MAP_FIXED, fd, offset);
                if (mapped == MAP_FAILED) {
                    const int saved_errno = errno;
                    munmap(reservation, length + 2 * page);
                    errno = saved_errno;
                    return MAP_FAILED;
                }
                __sync_add_and_fetch(&tcm_mappings, 1);
                char message[256];
                const int count = snprintf(message, sizeof(message),
                    "TCM_GUARD start=%p end=%p length=%zu offset=%ld\n",
                    mapped, static_cast<char*>(mapped) + length, length,
                    static_cast<long>(offset));
                if (count > 0) {
                    ssize_t ignored = write(STDERR_FILENO, message, static_cast<size_t>(count));
                    (void)ignored;
                }
                return mapped;
            }
        }
    }
    return next_mmap(address, length, protection, flags, fd, offset);
}

int main(int argc, char** argv) {
    try {
        std::string model = "../data/best.spacemit.fp16.onnx";
        int threads = 8;
        int runs = 1;
        bool guard_only = false;
        bool child = false;
        bool guarded = true;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            auto value = [&](const char* option) -> std::string {
                if (++i >= argc) throw std::runtime_error(std::string(option) + " requires a value");
                return argv[i];
            };
            if (argument == "--model") model = value("--model");
            else if (argument == "--threads") threads = positive_integer(value("--threads").c_str(), "--threads");
            else if (argument == "--runs") runs = positive_integer(value("--runs").c_str(), "--runs");
            else if (argument == "--guard-only") guard_only = true;
            else if (argument == "--child") {
                child = true;
                const std::string mode = value("--child");
                if (mode == "baseline") guarded = false;
                else if (mode != "guard") throw std::runtime_error("invalid --child mode");
            } else if (argument == "--help" || argument == "-h") {
                usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown option: " + argument);
            }
        }
        if (child) return inference(model, threads, guarded);
        std::cout << "K3 TCM single-file reproduction; model=" << model
                  << " threads=" << threads << " runs=" << runs << std::endl;
        const std::string executable = executable_path();
        int guard_crashes = 0;
        if (!guard_only) {
            for (int i = 1; i <= runs; ++i) run_child(executable, model, threads, false, i);
        }
        for (int i = 1; i <= runs; ++i) {
            if (run_child(executable, model, threads, true, i) == 139) ++guard_crashes;
        }
        std::cout << "SUMMARY guard_sigsegv=" << guard_crashes << '/' << runs << '\n';
        if (guard_crashes) {
            std::cout << "REPRODUCED: check FAULT.address against TCM_GUARD.end. "
                         "A successful baseline does not prove memory safety.\n";
            return 0;
        }
        std::cerr << "NOT REPRODUCED: no guarded SIGSEGV; check TCM_GUARD and ORT_ERROR output.\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << '\n';
        usage(argv[0]);
        return 2;
    }
}
