# 目标程序名。最终会生成一个名为 export-graph-ops 的可执行文件。
set(TARGET export-graph-ops)

# 源文件只有一个 main.cpp。
add_executable(${TARGET} main.cpp)

# 安装规则：把可执行文件作为运行时程序安装出去。
install(TARGETS ${TARGET} RUNTIME)

# 链接 stable-diffusion 和 ggml 两个库。
# 前者提供模型/runner/loader 等高层逻辑，后者提供底层张量与计算图实现。
target_link_libraries(${TARGET} PRIVATE stable-diffusion ggml)

# 头文件搜索路径：
# - src / include：stable-diffusion.cpp 自身头文件
# - ggml/src / ggml/include：ggml 相关头文件
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/src
    ${PROJECT_SOURCE_DIR}/ggml/include
)

# 要求 C11 与 C++17。
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)