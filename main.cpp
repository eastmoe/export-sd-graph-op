// export-graph-ops.cpp - Export unique GGML ops from diffusion model graphs
// Adapted from llama.cpp export-graph-ops for stable-diffusion.cpp
//
// 中文说明：
// 这个工具不是拿来“生成图片”的，而是拿来“导出算子集合”的。
// 它会：
// 1. 根据模型权重推断出当前模型家族；
// 2. 为对应模型构造一组最小占位输入；
// 3. 调用 build_graph() 构建 GGML 计算图；
// 4. 遍历图中的节点，提取每个节点的算子类型、输出形状、参数和输入布局；
// 5. 对重复节点去重后写入文本文件。
// 这样可以用来做算子覆盖率分析、后端适配检查、测试样本生成，以及不同模型家族的图结构对比。

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "auto_encoder_kl.hpp"
#include "clip.hpp"
#include "diffusion_model.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include "model.h"
#include "t5.hpp"
#include "tae.hpp"
#include "tensor.hpp"
#include "vae.hpp"
#include "wan.hpp"

// input_tensor：描述“某个节点输入张量”的静态信息。
// 注意：这里只记录类型、形状和步长，不记录实际数据。
// 对本工具来说，区分算子时这些信息已经足够。
struct input_tensor {
    ggml_type type;
    std::array<int64_t, 4> ne;
    std::array<size_t, 4> nb;

    input_tensor(ggml_type type, const int64_t* ne, const size_t* nb) : type(type) {
        memcpy(this->ne.data(), ne, 4 * sizeof(int64_t));
        memcpy(this->nb.data(), nb, 4 * sizeof(size_t));
    }

    // 为 std::set / std::map 提供稳定排序规则。
    bool operator<(const input_tensor& b) const {
        return std::tie(type, ne, nb) < std::tie(b.type, b.ne, b.nb);
    }

    // 把输入张量信息序列化成纯文本，方便后续写入测试文件。
    void serialize(std::ostream& out) const {
        out << type << ' ';
        for (size_t i = 0; i < 4; ++i) {
            out << ne[i] << ' ';
        }
        for (size_t i = 0; i < 4; ++i) {
            out << nb[i] << ' ';
        }
    }
};

// test_object：描述一个“可导出的唯一算子样本”。
// 唯一性由 op/type/ne/op_params/sources 决定，name 只用于调试显示。
// 后面会把它放到 std::set 中自动去重。
struct test_object {
    ggml_op op;
    ggml_type type;
    std::array<int64_t, 4> ne;
    std::vector<int32_t> op_params;
    std::vector<input_tensor> sources;
    std::string name;

    void serialize(std::ostream& out) const {
        out << op << ' ' << type << ' ';
        for (size_t i = 0; i < 4; ++i) {
            out << ne[i] << ' ';
        }

        out << op_params.size() << ' ';
        for (size_t i = 0; i < op_params.size(); ++i) {
            out << op_params[i] << ' ';
        }

        out << sources.size() << ' ';
        for (size_t s = 0; s < sources.size(); ++s) {
            sources[s].serialize(out);
        }

        if (!name.empty()) {
            out << name;
        } else {
            out << '-';
        }

        out << '\n';
    }

    // 比较时故意不包含 name。
    // 这意味着：只要两个节点的“算子语义签名”一致，即使名字不同，也会被视为同一种样本。
    bool operator<(const test_object& b) const {
        return std::tie(op, type, ne, op_params, sources) <
               std::tie(b.op, b.type, b.ne, b.op_params, b.sources);
    }
};

// 遍历一张 GGML 计算图，提取其中有计算意义的节点，并写入 tests 集合。
// label 只是日志标签，用来告诉你这些节点来自哪个组件。
static void extract_graph_ops(ggml_cgraph* cgraph, const char* label, std::set<test_object>& tests) {
    const int n_nodes  = ggml_graph_n_nodes(cgraph);
    int n_skipped      = 0;
    const int n_before = (int)tests.size();

    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor* node = ggml_graph_node(cgraph, i);
        if (node == nullptr) {
            continue;
        }

        // 这些节点更多是“视图/布局解释”，不是真正值得统计的数值算子。
        if (node->op == GGML_OP_NONE ||
            node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_TRANSPOSE) {
            n_skipped++;
            continue;
        }

        // 把当前节点压缩成一个可比较、可序列化的算子签名。
        test_object test;
        test.op   = node->op;
        test.type = node->type;
        memcpy(test.ne.data(), node->ne, 4 * sizeof(int64_t));

        test.op_params.resize(GGML_MAX_OP_PARAMS / sizeof(int32_t));
        memcpy(test.op_params.data(), node->op_params, GGML_MAX_OP_PARAMS);

        for (size_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] == nullptr) {
                break;
            }
            test.sources.emplace_back(node->src[s]->type, node->src[s]->ne, node->src[s]->nb);
        }

        if (node->name[0] != '\0') {
            test.name = node->name;
        }

        // 插入 set 时会自动按签名去重。
        tests.insert(std::move(test));
    }

    const int n_new = (int)tests.size() - n_before;
    fprintf(stdout, "export-graph-ops: %s: %d unique ops, %d total nodes, %d skipped (view ops)\n",
            label, n_new, n_nodes, n_skipped);
}

// 把内部版本枚举转成日志友好的字符串。
static const char* version_desc(SDVersion version) {
    if (sd_version_is_sd3(version)) {
        return "SD3 (MMDiT)";
    }
    if (sd_version_is_unet(version)) {
        if (sd_version_is_sdxl(version)) {
            return "SDXL-family (UNet)";
        }
        if (sd_version_is_sd2(version)) {
            return "SD2-family (UNet)";
        }
        return "SD1-family (UNet)";
    }
    if (sd_version_is_flux(version) || sd_version_is_flux2(version)) {
        return "Flux-family";
    }
    if (sd_version_is_qwen_image(version)) {
        return "Qwen Image";
    }
    if (sd_version_is_anima(version)) {
        return "Anima";
    }
    if (sd_version_is_z_image(version)) {
        return "Z-Image";
    }
    if (sd_version_is_wan(version)) {
        return "Wan";
    }
    return "Other";
}

// 在张量表中查找“以某个后缀结尾”的权重。
// 之所以常用后缀匹配，是因为模型前缀可能变化，但核心权重尾名通常比较稳定。
static ggml_tensor* find_tensor_suffix(const std::map<std::string, ggml_tensor*>& tensors, const std::string& suffix) {
    for (const auto& kv : tensors) {
        const std::string& name = kv.first;
        if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return kv.second;
        }
    }
    return nullptr;
}

// 在张量表中查找“名字包含某个子串”的权重。
static ggml_tensor* find_tensor_contains(const std::map<std::string, ggml_tensor*>& tensors, const std::string& needle) {
    for (const auto& kv : tensors) {
        if (kv.first.find(needle) != std::string::npos) {
            return kv.second;
        }
    }
    return nullptr;
}

// 判断是否存在指定后缀的权重。
static bool has_tensor_suffix(const std::map<std::string, ggml_tensor*>& tensors, const std::string& suffix) {
    return find_tensor_suffix(tensors, suffix) != nullptr;
}

// 打印一条关键权重的形状信息。
// 只要候选后缀里有一个命中，就立刻输出并返回。
static void print_tensor_shape_line(const std::map<std::string, ggml_tensor*>& tensors,
                                    const char* label,
                                    const std::vector<std::string>& suffixes) {
    for (const auto& suffix : suffixes) {
        for (const auto& kv : tensors) {
            const std::string& name = kv.first;
            if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                const ggml_tensor* t = kv.second;
                fprintf(stdout, "  %s (%s): [%lld, %lld, %lld, %lld]\n",
                        label,
                        name.c_str(),
                        (long long)t->ne[0],
                        (long long)t->ne[1],
                        (long long)t->ne[2],
                        (long long)t->ne[3]);
                return;
            }
        }
    }
}

// 针对不同模型家族，打印几组最关键的权重维度。
// 这有助于快速确认：输入通道数、文本维度、时间嵌入维度等是否被正确推断。
static void print_key_dimensions(const std::map<std::string, ggml_tensor*>& tensors, SDVersion version) {
    fprintf(stdout, "\nexport-graph-ops: Key model dimensions:\n");

    if (sd_version_is_unet(version)) {
        print_tensor_shape_line(tensors, "First conv weight", {"input_blocks.0.0.weight"});
        print_tensor_shape_line(tensors, "Time embed weight", {"time_embed.0.weight", "time_embed.2.weight"});
        print_tensor_shape_line(tensors, "Label emb weight", {"label_emb.0.0.weight", "label_emb.0.2.weight"});
        return;
    }

    if (sd_version_is_sd3(version)) {
        print_tensor_shape_line(tensors, "Patch embed weight", {"x_embedder.proj.weight"});
        print_tensor_shape_line(tensors, "Context embed weight", {"context_embedder.weight"});
        print_tensor_shape_line(tensors, "Y embed weight", {"y_embedder.mlp.0.weight"});
        return;
    }

    if (sd_version_is_flux(version) || sd_version_is_flux2(version)) {
        print_tensor_shape_line(tensors, "Image input weight", {"img_in_patch.weight", "img_in.weight"});
        print_tensor_shape_line(tensors, "Text input weight", {"txt_in.weight"});
        print_tensor_shape_line(tensors, "Vector input weight", {"vector_in.in_layer.weight"});
        print_tensor_shape_line(tensors, "Guidance input weight", {"guidance_in.in_layer.weight", "distilled_guidance_layer.in_proj.weight"});
        return;
    }

    if (sd_version_is_qwen_image(version)) {
        print_tensor_shape_line(tensors, "Image input weight", {"img_in.weight"});
        print_tensor_shape_line(tensors, "Text input weight", {"txt_in.weight"});
        return;
    }

    if (sd_version_is_anima(version)) {
        print_tensor_shape_line(tensors, "X embed weight", {"net.x_embedder.proj.1.weight"});
        print_tensor_shape_line(tensors, "Cross-attn K weight", {"net.blocks.0.cross_attn.k_proj.weight"});
        return;
    }

    if (sd_version_is_z_image(version)) {
        print_tensor_shape_line(tensors, "Image embed weight", {"x_embedder.weight"});
        print_tensor_shape_line(tensors, "Caption embed weight", {"cap_embedder.1.weight"});
        return;
    }

    if (sd_version_is_wan(version)) {
        print_tensor_shape_line(tensors, "Patch embed weight", {"patch_embedding.weight"});
        print_tensor_shape_line(tensors, "Text embed weight", {"text_embedding.0.weight", "text_embedding.2.weight"});
        print_tensor_shape_line(tensors, "Time embed weight", {"time_embedding.0.weight", "time_embedding.2.weight"});
        print_tensor_shape_line(tensors, "Image embed weight", {"img_emb.proj.0.weight", "img_emb.proj.1.weight", "img_emb.emb_pos"});
        print_tensor_shape_line(tensors, "VACE patch embed weight", {"vace_patch_embedding.weight"});
        return;
    }
}

// 把去重后的算子样本写入输出文件。
static void write_tests(const char* output_file, const std::set<test_object>& tests) {
    std::ofstream f(output_file);
    if (!f.is_open()) {
        throw std::runtime_error("unable to open output file");
    }

    for (const auto& test : tests) {
        test.serialize(f);
    }
}

// 通用加载助手：
// 1. 让模型对象分配参数缓冲区；
// 2. 让模型对象注册需要加载的参数张量；
// 3. 由 ModelLoader 把权重读入这些张量。
template <typename ModelT>
static bool load_model_tensors(ModelLoader& model_loader,
                               ModelT& model,
                               std::map<std::string, ggml_tensor*>& tensors) {
    model.alloc_params_buffer();
    model.get_param_tensors(tensors);
    return model_loader.load_tensors(tensors);
}

// 与 load_model_tensors 类似，但只处理某个前缀命名空间下的组件。
// 比如 text_encoders.xxx、first_stage_model、tae 等。
template <typename ModelT>
static bool load_prefixed_model_tensors(ModelLoader& model_loader,
                                        ModelT& model,
                                        std::map<std::string, ggml_tensor*>& tensors,
                                        const std::string& prefix) {
    model.alloc_params_buffer();
    model.get_param_tensors(tensors, prefix);
    return model_loader.load_tensors(tensors);
}

// UNet 构图所需的占位输入集合。
struct UnetInputs {
    sd::Tensor<float> x;          // 潜空间输入。
    sd::Tensor<float> timesteps;  // 扩散时间步。
    sd::Tensor<float> context;    // 文本上下文。
    sd::Tensor<float> c_concat;   // 额外拼接条件（很多模型为空）。
    sd::Tensor<float> y;          // 类别/ADM 条件。
};

// 为 UNet 家族构造一组“只用于搭图”的占位输入。
// 数值本身不重要，关键是形状必须与模型结构兼容。
static UnetInputs make_unet_inputs(SDVersion version, const std::map<std::string, ggml_tensor*>& tensors) {
    int64_t in_channels = 4;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "input_blocks.0.0.weight")) {
        in_channels = t->ne[2];
    }

    int64_t context_dim = 768;
    if (sd_version_is_sd2(version) || version == VERSION_SVD) {
        context_dim = 1024;
    } else if (sd_version_is_sdxl(version)) {
        context_dim = 2048;
    }

    int64_t adm_in_channels = 0;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "label_emb.0.0.weight")) {
        adm_in_channels = t->ne[0];
    }

    UnetInputs inputs;
    // 这里用一个较小但典型的 latent 空间尺寸即可，重点是让图完整展开。
    inputs.x = sd::Tensor<float>({64, 64, in_channels, 1});
    inputs.x.fill_(0.0f);

    // 时间步数值只要合法即可；这里用一个固定占位值。
    std::vector<float> timestep_vec(1, 999.0f);
    inputs.timesteps = sd::Tensor<float>::from_vector(timestep_vec);

    inputs.context = sd::Tensor<float>({context_dim, 77, 1});
    inputs.context.fill_(0.0f);

    if (adm_in_channels > 0) {
        inputs.y = sd::Tensor<float>({adm_in_channels, 1});
        inputs.y.fill_(0.0f);
    }

    return inputs;
}

// MMDiT（SD3）构图所需的占位输入集合。
struct MMDiTInputs {
    sd::Tensor<float> x;          // 图像潜变量输入。
    sd::Tensor<float> timesteps;  // 时间步。
    sd::Tensor<float> context;    // 文本上下文。
    sd::Tensor<float> y;          // 额外条件向量。
};

// 为 SD3 的 MMDiT 主干构造占位输入。
static MMDiTInputs make_mmdit_inputs(const std::map<std::string, ggml_tensor*>& tensors) {
    int64_t in_channels = 16;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "x_embedder.proj.weight")) {
        in_channels = t->ne[2];
    }

    int64_t context_dim = 4096;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "context_embedder.weight")) {
        context_dim = t->ne[0];
    }

    int64_t y_dim = 2048;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "y_embedder.mlp.0.weight")) {
        y_dim = t->ne[0];
    }

    MMDiTInputs inputs;
    inputs.x = sd::Tensor<float>({128, 128, in_channels, 1});
    inputs.x.fill_(0.0f);

    std::vector<float> timestep_vec(1, 999.0f);
    inputs.timesteps = sd::Tensor<float>::from_vector(timestep_vec);

    inputs.context = sd::Tensor<float>({context_dim, 154, 1});
    inputs.context.fill_(0.0f);

    inputs.y = sd::Tensor<float>({y_dim, 1});
    inputs.y.fill_(0.0f);

    return inputs;
}

// Flux / Flux2 构图所需的占位输入集合。
struct FluxInputs {
    sd::Tensor<float> x;          // 图像潜变量输入。
    sd::Tensor<float> timesteps;  // 时间步。
    sd::Tensor<float> context;    // 文本上下文。
    sd::Tensor<float> c_concat;   // 额外拼接条件。
    sd::Tensor<float> y;          // 向量条件。
    sd::Tensor<float> guidance;   // guidance / distilled guidance 条件。
};

// 为 Flux / Flux2 家族构造占位输入。
static FluxInputs make_flux_inputs(SDVersion version, const std::map<std::string, ggml_tensor*>& tensors) {
    int64_t in_channels = 64;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "img_in_patch.weight")) {
        in_channels = t->ne[2];
    } else if (const ggml_tensor* t = find_tensor_suffix(tensors, "img_in.weight")) {
        in_channels = t->ne[0];
    }

    int64_t context_dim = 4096;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "txt_in.weight")) {
        context_dim = t->ne[0];
    }

    int64_t y_dim = 0;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "vector_in.in_layer.weight")) {
        y_dim = t->ne[0];
    }

    FluxInputs inputs;
    inputs.x = sd::Tensor<float>({16, 16, in_channels, 1});
    inputs.x.fill_(0.0f);

    // Flux 分支常用不同的时间步标度，这里给一个简单占位值。
    std::vector<float> timestep_vec(1, 1.0f);
    inputs.timesteps = sd::Tensor<float>::from_vector(timestep_vec);

    inputs.context = sd::Tensor<float>({context_dim, 256, 1});
    inputs.context.fill_(0.0f);

    if (y_dim > 0) {
        inputs.y = sd::Tensor<float>({y_dim, 1});
        inputs.y.fill_(0.0f);
    }

    std::vector<float> guidance_vec(1, 0.0f);
    // Chroma Radiance 需要显式 guidance 占位值。
    if (version == VERSION_CHROMA_RADIANCE) {
        guidance_vec[0] = 1.0f;
    }
    inputs.guidance = sd::Tensor<float>::from_vector(guidance_vec);

    return inputs;
}

// Qwen Image 构图所需的占位输入集合。
struct QwenImageInputs {
    sd::Tensor<float> x;          // 图像潜变量输入。
    sd::Tensor<float> timesteps;  // 时间步。
    sd::Tensor<float> context;    // 文本上下文。
};

// 为 Qwen Image 家族构造占位输入。
static QwenImageInputs make_qwen_image_inputs(const std::map<std::string, ggml_tensor*>& tensors) {
    int64_t in_channels = 64;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "img_in.weight")) {
        in_channels = t->ne[0];
    }

    int64_t context_dim = 3584;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "txt_in.weight")) {
        context_dim = t->ne[0];
    }

    QwenImageInputs inputs;
    inputs.x = sd::Tensor<float>({16, 16, in_channels, 1});
    inputs.x.fill_(0.0f);

    std::vector<float> timestep_vec(1, 1000.0f);
    inputs.timesteps = sd::Tensor<float>::from_vector(timestep_vec);

    inputs.context = sd::Tensor<float>({context_dim, 256, 1});
    inputs.context.fill_(0.0f);

    return inputs;
}

// Anima 构图所需的占位输入集合。
struct AnimaInputs {
    sd::Tensor<float> x;          // 图像潜变量输入。
    sd::Tensor<float> timesteps;  // 时间步。
    sd::Tensor<float> context;    // 文本上下文。
};

// 为 Anima 家族构造占位输入。
static AnimaInputs make_anima_inputs(const std::map<std::string, ggml_tensor*>& tensors) {
    int64_t in_channels = 16;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "net.x_embedder.proj.1.weight")) {
        const int64_t patch_size = 2;
        if (t->ne[0] % (patch_size * patch_size) == 0) {
            in_channels = t->ne[0] / (patch_size * patch_size) - 1;
        }
    }

    int64_t context_dim = 1024;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "net.blocks.0.cross_attn.k_proj.weight")) {
        context_dim = t->ne[0];
    }

    AnimaInputs inputs;
    inputs.x = sd::Tensor<float>({16, 16, in_channels, 1});
    inputs.x.fill_(0.0f);

    std::vector<float> timestep_vec(1, 1000.0f);
    inputs.timesteps = sd::Tensor<float>::from_vector(timestep_vec);

    inputs.context = sd::Tensor<float>({context_dim, 256, 1});
    inputs.context.fill_(0.0f);

    return inputs;
}

// Z-Image 构图所需的占位输入集合。
struct ZImageInputs {
    sd::Tensor<float> x;          // 图像潜变量输入。
    sd::Tensor<float> timesteps;  // 时间步。
    sd::Tensor<float> context;    // 文本上下文。
};

// Wan 构图所需的占位输入集合。
struct WanInputs {
    sd::Tensor<float> x;                // 视频/图像潜变量输入。
    sd::Tensor<float> timesteps;        // 多帧时间步。
    sd::Tensor<float> context;          // 文本上下文。
    sd::Tensor<float> clip_fea;         // 图像编码特征。
    sd::Tensor<float> c_concat;         // 额外拼接条件。
    sd::Tensor<float> time_dim_concat;  // 预留的时间维拼接条件。
    sd::Tensor<float> vace_context;     // VACE 相关上下文。
};

// 为 Z-Image 家族构造占位输入。
static ZImageInputs make_z_image_inputs(const std::map<std::string, ggml_tensor*>& tensors) {
    int64_t in_channels = 16;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "x_embedder.weight")) {
        const int64_t patch_size = 2;
        if (t->ne[0] % (patch_size * patch_size) == 0) {
            in_channels = t->ne[0] / (patch_size * patch_size);
        }
    }

    int64_t context_dim = 2560;
    if (const ggml_tensor* t = find_tensor_suffix(tensors, "cap_embedder.1.weight")) {
        context_dim = t->ne[0];
    }

    ZImageInputs inputs;
    inputs.x = sd::Tensor<float>({16, 16, in_channels, 1});
    inputs.x.fill_(0.0f);

    std::vector<float> timestep_vec(1, 1000.0f);
    inputs.timesteps = sd::Tensor<float>::from_vector(timestep_vec);

    inputs.context = sd::Tensor<float>({context_dim, 256, 1});
    inputs.context.fill_(0.0f);

    return inputs;
}

// 为 Wan 家族构造占位输入。
// 相比普通图像扩散模型，Wan 可能还需要图像特征、额外拼接条件和 VACE 上下文。
static WanInputs make_wan_inputs(WAN::WanRunner& model,
                                 const std::map<std::string, ggml_tensor*>& tensors) {
    WanInputs inputs;

    const int64_t width           = 16;
    const int64_t height          = 16;
    const int64_t time            = 3;
    const int64_t latent_channels = model.wan_params.out_dim;

    inputs.x = sd::Tensor<float>({width, height, time, latent_channels, 1});
    inputs.x.fill_(0.0f);

    std::vector<float> timestep_vec(static_cast<size_t>(time), 1000.0f);
    timestep_vec[0]  = 0.0f;
    inputs.timesteps = sd::Tensor<float>::from_vector(timestep_vec);

    inputs.context = sd::Tensor<float>({model.wan_params.text_dim, model.wan_params.text_len, 1});
    inputs.context.fill_(0.0f);

    // 只有当权重里真的带有图像嵌入相关参数时，才构造对应输入。
    const bool has_img_emb = has_tensor_suffix(tensors, "img_emb.proj.0.weight") ||
                             has_tensor_suffix(tensors, "img_emb.proj.1.weight") ||
                             has_tensor_suffix(tensors, "img_emb.emb_pos");
    if (has_img_emb) {
        const int64_t clip_tokens = model.wan_params.flf_pos_embed_token_number > 0 ? model.wan_params.flf_pos_embed_token_number : 257;
        inputs.clip_fea           = sd::Tensor<float>({1280, clip_tokens, 1});
        inputs.clip_fea.fill_(0.0f);

        // Wan2.1 I2V / FLF2V use clip_fea + c_concat. Wan2.2 I2V only needs clip_fea.
        if (model.get_desc() != "Wan2.2-I2V-14B") {
            const int64_t concat_channels = 4 + model.wan_params.out_dim;
            inputs.c_concat               = sd::Tensor<float>({width, height, time, concat_channels, 1});
            inputs.c_concat.fill_(0.0f);
        }
    }

    if (model.wan_params.vace_layers > 0) {
        inputs.vace_context = sd::Tensor<float>({width, height, time, model.wan_params.vace_in_dim, 1});
        inputs.vace_context.fill_(0.0f);
    }

    return inputs;
}

// 可以按阶段导出不同计算图。
enum class ExportStage {
    DIFFUSION,
    TEXT,
    VAE,
    ALL,
};

// 解析命令行里的 stage 字符串。
static ExportStage parse_stage(const char* stage) {
    // 默认导出 diffusion，因为这是最核心、最普遍的阶段。
    if (stage == nullptr || strcmp(stage, "diffusion") == 0) {
        return ExportStage::DIFFUSION;
    }
    if (strcmp(stage, "text") == 0 || strcmp(stage, "text-encoder") == 0 || strcmp(stage, "text_encoder") == 0) {
        return ExportStage::TEXT;
    }
    if (strcmp(stage, "vae") == 0) {
        return ExportStage::VAE;
    }
    if (strcmp(stage, "all") == 0) {
        return ExportStage::ALL;
    }
    throw std::runtime_error(std::string("unknown stage: ") + stage);
}

// 把导出阶段枚举转换成便于打印的字符串。
static const char* stage_desc(ExportStage stage) {
    switch (stage) {
        case ExportStage::DIFFUSION:
            return "diffusion";
        case ExportStage::TEXT:
            return "text";
        case ExportStage::VAE:
            return "vae";
        case ExportStage::ALL:
            return "all";
    }
    return "unknown";
}

// 当前 stage 是否包含 diffusion 主干。
static bool includes_diffusion(ExportStage stage) {
    return stage == ExportStage::DIFFUSION || stage == ExportStage::ALL;
}

// 当前 stage 是否包含 text encoder。
static bool includes_text(ExportStage stage) {
    return stage == ExportStage::TEXT || stage == ExportStage::ALL;
}

// 当前 stage 是否包含 VAE。
static bool includes_vae(ExportStage stage) {
    return stage == ExportStage::VAE || stage == ExportStage::ALL;
}

// 在已加载的权重表中检查是否存在某个前缀命名空间。
static bool tensor_storage_has_prefix(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
    for (const auto& kv : tensor_storage_map) {
        if (starts_with(kv.first, prefix)) {
            return true;
        }
    }
    return false;
}

// 在已加载的权重表中检查是否存在某个关键子串。
static bool tensor_storage_contains(const String2TensorStorage& tensor_storage_map, const std::string& needle) {
    for (const auto& kv : tensor_storage_map) {
        if (kv.first.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// 给不同模型家族选择一个默认的 clip_skip。
static int default_clip_skip(SDVersion version) {
    return sd_version_is_sd1(version) ? 1 : 2;
}

// 构造一组假的 CLIP token id。
// 只要长度与特殊 token 合理，就足以触发完整构图。
static sd::Tensor<int32_t> make_clip_input_ids(int64_t n_token) {
    if (n_token <= 0) {
        n_token = 77;
    }
    std::vector<int32_t> ids((size_t)n_token, 1);
    if (n_token > 0) {
        ids[0]                   = 49406;
        ids[(size_t)n_token - 1] = 49407;
    }
    return sd::Tensor<int32_t>({n_token}, ids);
}

// 构造一组假的 T5 token id。
static sd::Tensor<int32_t> make_t5_input_ids(int64_t n_token) {
    if (n_token <= 0) {
        n_token = 77;
    }
    std::vector<int32_t> ids((size_t)n_token);
    for (int64_t i = 0; i < n_token; ++i) {
        ids[(size_t)i] = (int32_t)((i % 32000) + 1);
    }
    return sd::Tensor<int32_t>({n_token}, ids);
}

// 构造 T5 attention mask。
// 这里约定：0 表示可见，-inf 表示被 mask。
static sd::Tensor<float> make_t5_attention_mask(int64_t n_token, int64_t active_tokens = -1) {
    if (n_token <= 0) {
        n_token = 77;
    }
    if (active_tokens < 0 || active_tokens > n_token) {
        int64_t masked_tokens = std::min<int64_t>(8, std::max<int64_t>(1, n_token / 8));
        active_tokens         = std::max<int64_t>(1, n_token - masked_tokens);
    }

    std::vector<float> mask((size_t)n_token, -HUGE_VALF);
    for (int64_t i = 0; i < active_tokens; ++i) {
        mask[(size_t)i] = 0.0f;
    }
    return sd::Tensor<float>({n_token}, mask);
}

// 导出一个 CLIP 文本编码器的算子。
// 可以分别覆盖 hidden 输出路径和 pooled 输出路径。
static void export_clip_runner_ops(CLIPTextModelRunner& model,
                                   const std::string& label,
                                   int64_t n_token,
                                   int clip_skip,
                                   bool export_hidden,
                                   bool export_pooled,
                                   std::set<test_object>& tests) {
    sd::Tensor<int32_t> input_ids = make_clip_input_ids(n_token);
    size_t max_token_idx          = input_ids.shape()[0] > 0 ? (size_t)input_ids.shape()[0] - 1 : 0;

    // hidden 路径：保留逐 token 的隐藏状态输出。
    if (export_hidden) {
        model.reset_compute_ctx();
        ggml_cgraph* gf          = model.build_graph(input_ids, 0, nullptr, max_token_idx, false, clip_skip);
        std::string hidden_label = label + ".hidden";
        extract_graph_ops(gf, hidden_label.c_str(), tests);
    }

    // pooled 路径：导出池化后的全局文本表示。
    if (export_pooled) {
        model.reset_compute_ctx();
        ggml_cgraph* gf          = model.build_graph(input_ids, 0, nullptr, max_token_idx, true, clip_skip);
        std::string pooled_label = label + ".pooled";
        extract_graph_ops(gf, pooled_label.c_str(), tests);
    }
}

// 导出一个 T5 文本编码器的算子。
static void export_t5_runner_ops(T5Runner& model,
                                 const std::string& label,
                                 int64_t n_token,
                                 bool use_attention_mask,
                                 std::set<test_object>& tests,
                                 int64_t active_tokens = -1) {
    sd::Tensor<int32_t> input_ids = make_t5_input_ids(n_token);
    sd::Tensor<float> attention_mask;
    // 对需要 attention mask 的 T5 变体，补一份占位掩码。
    if (use_attention_mask) {
        attention_mask = make_t5_attention_mask(n_token, active_tokens);
    }

    model.reset_compute_ctx();
    ggml_cgraph* gf = model.build_graph(input_ids, attention_mask);
    extract_graph_ops(gf, label.c_str(), tests);
}

// 统一的组件加载日志。
static void log_component_loaded(const std::string& label, size_t tensor_count) {
    fprintf(stdout, "export-graph-ops: component '%s' loaded successfully\n", label.c_str());
    fprintf(stdout, "export-graph-ops: component '%s' tensors: %zu\n", label.c_str(), tensor_count);
}

// 导出扩散主干阶段的算子。
// 根据版本分发到 UNet / MMDiT / Flux / Qwen / Anima / Z-Image / Wan 等不同实现。
static void export_diffusion_stage(const char* model_path,
                                   ggml_backend_t backend,
                                   std::set<test_object>& tests) {
    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(model_path, "model.diffusion_model.")) {
        throw std::runtime_error(std::string("failed to load model from '") + model_path + "'");
    }

    auto& tensor_storage_map = model_loader.get_tensor_storage_map();
    SDVersion version        = model_loader.get_sd_version();

    fprintf(stdout, "export-graph-ops: diffusion stage model version = %d (%s)\n", (int)version, version_desc(version));

    std::map<std::string, ggml_tensor*> tensors;
    std::string model_type;

    // SD1 / SD2 / SDXL：扩散主干是 UNet。
    if (sd_version_is_unet(version)) {
        UNetModel model(backend, false, tensor_storage_map, version);
        if (!load_model_tensors(model_loader, model, tensors)) {
            throw std::runtime_error("failed to load diffusion tensors");
        }

        model_type = model.get_desc();
        fprintf(stdout, "export-graph-ops: model loaded successfully\n");
        fprintf(stdout, "export-graph-ops: total tensors: %zu\n", tensors.size());
        print_key_dimensions(tensors, version);
        fprintf(stdout, "\nexport-graph-ops: model type: %s\n", model_type.c_str());

        UnetInputs inputs = make_unet_inputs(version, tensors);
        // reset_compute_ctx() 是为了确保本次 build_graph 使用干净的计算上下文。
        model.unet.reset_compute_ctx();
        // build_graph() 只搭图，不执行推理。
        ggml_cgraph* gf = model.unet.build_graph(inputs.x, inputs.timesteps, inputs.context, inputs.c_concat, inputs.y);
        extract_graph_ops(gf, model_type.c_str(), tests);
        return;
    }

    // SD3：扩散主干换成了 MMDiT。
    if (sd_version_is_sd3(version)) {
        MMDiTModel model(backend, false, tensor_storage_map);
        if (!load_model_tensors(model_loader, model, tensors)) {
            throw std::runtime_error("failed to load diffusion tensors");
        }

        model_type = model.get_desc();
        fprintf(stdout, "export-graph-ops: model loaded successfully\n");
        fprintf(stdout, "export-graph-ops: total tensors: %zu\n", tensors.size());
        print_key_dimensions(tensors, version);
        fprintf(stdout, "\nexport-graph-ops: model type: %s\n", model_type.c_str());

        MMDiTInputs inputs = make_mmdit_inputs(tensors);
        model.mmdit.reset_compute_ctx();
        ggml_cgraph* gf = model.mmdit.build_graph(inputs.x, inputs.timesteps, inputs.context, inputs.y);
        extract_graph_ops(gf, model_type.c_str(), tests);
        return;
    }

    // Flux / Flux2：走独立的 FluxModel 逻辑。
    if (sd_version_is_flux(version) || sd_version_is_flux2(version)) {
        FluxModel model(backend, false, tensor_storage_map, version, false);
        if (!load_model_tensors(model_loader, model, tensors)) {
            throw std::runtime_error("failed to load diffusion tensors");
        }

        model_type = model.get_desc();
        fprintf(stdout, "export-graph-ops: model loaded successfully\n");
        fprintf(stdout, "export-graph-ops: total tensors: %zu\n", tensors.size());
        print_key_dimensions(tensors, version);
        fprintf(stdout, "\nexport-graph-ops: model type: %s\n", model_type.c_str());

        FluxInputs inputs = make_flux_inputs(version, tensors);
        const std::vector<sd::Tensor<float>> empty_ref_latents;
        const std::vector<int> empty_skip_layers;
        model.flux.reset_compute_ctx();
        ggml_cgraph* gf = model.flux.build_graph(inputs.x,
                                                 inputs.timesteps,
                                                 inputs.context,
                                                 inputs.c_concat,
                                                 inputs.y,
                                                 inputs.guidance,
                                                 empty_ref_latents,
                                                 false,
                                                 empty_skip_layers);
        extract_graph_ops(gf, model_type.c_str(), tests);
        return;
    }

    // Qwen Image：使用专门的 QwenImageModel。
    if (sd_version_is_qwen_image(version)) {
        QwenImageModel model(backend, false, tensor_storage_map, "model.diffusion_model", version, false);
        if (!load_model_tensors(model_loader, model, tensors)) {
            throw std::runtime_error("failed to load diffusion tensors");
        }

        model_type = model.get_desc();
        fprintf(stdout, "export-graph-ops: model loaded successfully\n");
        fprintf(stdout, "export-graph-ops: total tensors: %zu\n", tensors.size());
        print_key_dimensions(tensors, version);
        fprintf(stdout, "\nexport-graph-ops: model type: %s\n", model_type.c_str());

        QwenImageInputs inputs = make_qwen_image_inputs(tensors);
        const std::vector<sd::Tensor<float>> empty_ref_latents;
        model.qwen_image.reset_compute_ctx();
        ggml_cgraph* gf = model.qwen_image.build_graph(inputs.x, inputs.timesteps, inputs.context, empty_ref_latents, false);
        extract_graph_ops(gf, model_type.c_str(), tests);
        return;
    }

    // Anima：使用专门的 AnimaModel。
    if (sd_version_is_anima(version)) {
        AnimaModel model(backend, false, tensor_storage_map, "model.diffusion_model");
        if (!load_model_tensors(model_loader, model, tensors)) {
            throw std::runtime_error("failed to load diffusion tensors");
        }

        model_type = model.get_desc();
        fprintf(stdout, "export-graph-ops: model loaded successfully\n");
        fprintf(stdout, "export-graph-ops: total tensors: %zu\n", tensors.size());
        print_key_dimensions(tensors, version);
        fprintf(stdout, "\nexport-graph-ops: model type: %s\n", model_type.c_str());

        AnimaInputs inputs = make_anima_inputs(tensors);
        const sd::Tensor<int32_t> empty_t5_ids;
        const sd::Tensor<float> empty_t5_weights;
        model.anima.reset_compute_ctx();
        ggml_cgraph* gf = model.anima.build_graph(inputs.x, inputs.timesteps, inputs.context, empty_t5_ids, empty_t5_weights);
        extract_graph_ops(gf, model_type.c_str(), tests);
        return;
    }

    // Z-Image：使用专门的 ZImageModel。
    if (sd_version_is_z_image(version)) {
        ZImageModel model(backend, false, tensor_storage_map, "model.diffusion_model", version);
        if (!load_model_tensors(model_loader, model, tensors)) {
            throw std::runtime_error("failed to load diffusion tensors");
        }

        model_type = model.get_desc();
        fprintf(stdout, "export-graph-ops: model loaded successfully\n");
        fprintf(stdout, "export-graph-ops: total tensors: %zu\n", tensors.size());
        print_key_dimensions(tensors, version);
        fprintf(stdout, "\nexport-graph-ops: model type: %s\n", model_type.c_str());

        ZImageInputs inputs = make_z_image_inputs(tensors);
        const std::vector<sd::Tensor<float>> empty_ref_latents;
        model.z_image.reset_compute_ctx();
        ggml_cgraph* gf = model.z_image.build_graph(inputs.x, inputs.timesteps, inputs.context, empty_ref_latents, false);
        extract_graph_ops(gf, model_type.c_str(), tests);
        return;
    }

    // Wan：视频/多模态家族，输入接口最复杂。
    if (sd_version_is_wan(version)) {
        WAN::WanRunner model(backend, false, tensor_storage_map, "model.diffusion_model", version);
        if (!load_prefixed_model_tensors(model_loader, model, tensors, "model.diffusion_model")) {
            throw std::runtime_error("failed to load diffusion tensors");
        }

        model_type = model.get_desc();
        fprintf(stdout, "export-graph-ops: model loaded successfully\n");
        fprintf(stdout, "export-graph-ops: total tensors: %zu\n", tensors.size());
        print_key_dimensions(tensors, version);
        fprintf(stdout, "\nexport-graph-ops: model type: %s\n", model_type.c_str());

        WanInputs inputs = make_wan_inputs(model, tensors);
        model.reset_compute_ctx();
        ggml_cgraph* gf = model.build_graph(inputs.x,
                                            inputs.timesteps,
                                            inputs.context,
                                            inputs.clip_fea,
                                            inputs.c_concat,
                                            inputs.time_dim_concat,
                                            inputs.vace_context,
                                            1.0f);
        extract_graph_ops(gf, model_type.c_str(), tests);
        return;
    }

    throw std::runtime_error(std::string("unsupported diffusion model version: ") + version_desc(version));
}

// 导出文本编码器阶段的算子。
// 当前支持的重点是 CLIP / OpenCLIP / T5；
// 对纯 LLM 文本路径，本工具会有意识地跳过。
static bool export_text_stage(const char* model_path,
                              ggml_backend_t backend,
                              std::set<test_object>& tests,
                              bool fail_if_unsupported) {
    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(model_path)) {
        throw std::runtime_error(std::string("failed to load model from '") + model_path + "'");
    }

    auto& tensor_storage_map = model_loader.get_tensor_storage_map();
    SDVersion version        = model_loader.get_sd_version();
    int clip_skip            = default_clip_skip(version);

    fprintf(stdout, "export-graph-ops: text stage model version = %d (%s)\n", (int)version, version_desc(version));

    bool exported_any = false;

    // 对“纯 LLM 或以 LLM 为主的文本条件路径”统一走这里。
    // 当前工具聚焦 stable-diffusion.cpp 中已有的 GGML 文本编码器图，因此选择跳过。
    auto unsupported_llm_only = [&]() {
        if (fail_if_unsupported) {
            throw std::runtime_error("this model family uses a pure LLM text encoder (or an LLM-dominant conditioner), which is intentionally skipped here");
        }
        fprintf(stdout, "export-graph-ops: text stage skipped: model uses a pure LLM text encoder path that is intentionally not exported here\n");
        return false;
    };

    if (sd_version_is_unet(version)) {
        if (!tensor_storage_has_prefix(tensor_storage_map, "cond_stage_model.transformer.text_model")) {
            if (fail_if_unsupported) {
                throw std::runtime_error("no CLIP text encoder tensors found in model file");
            }
            fprintf(stdout, "export-graph-ops: text stage skipped: no CLIP text encoder tensors found\n");
            return false;
        }

        // SDXL 可能同时包含 clip_l 和 clip_g 两套编码器，因此分别导出。
        if (sd_version_is_sdxl(version)) {
            {
                CLIPTextModelRunner clip_l(backend, false, tensor_storage_map,
                                           "cond_stage_model.transformer.text_model",
                                           OPENAI_CLIP_VIT_L_14,
                                           false);
                std::map<std::string, ggml_tensor*> tensors;
                if (!load_prefixed_model_tensors(model_loader, clip_l, tensors, "cond_stage_model.transformer.text_model")) {
                    throw std::runtime_error("failed to load SDXL clip_l tensors");
                }
                log_component_loaded("clip_l", tensors.size());
                export_clip_runner_ops(clip_l, "clip_l", clip_l.model.n_token, clip_skip, true, false, tests);
                exported_any = true;
            }
            if (tensor_storage_has_prefix(tensor_storage_map, "cond_stage_model.1.transformer.text_model")) {
                CLIPTextModelRunner clip_g(backend, false, tensor_storage_map,
                                           "cond_stage_model.1.transformer.text_model",
                                           OPEN_CLIP_VIT_BIGG_14,
                                           false);
                std::map<std::string, ggml_tensor*> tensors;
                if (!load_prefixed_model_tensors(model_loader, clip_g, tensors, "cond_stage_model.1.transformer.text_model")) {
                    throw std::runtime_error("failed to load SDXL clip_g tensors");
                }
                log_component_loaded("clip_g", tensors.size());
                export_clip_runner_ops(clip_g, "clip_g", clip_g.model.n_token, clip_skip, true, true, tests);
                exported_any = true;
            }
        } else {
            CLIPVersion clip_version = sd_version_is_sd2(version) ? OPEN_CLIP_VIT_H_14 : OPENAI_CLIP_VIT_L_14;
            CLIPTextModelRunner clip(backend, false, tensor_storage_map,
                                     "cond_stage_model.transformer.text_model",
                                     clip_version,
                                     true);
            std::map<std::string, ggml_tensor*> tensors;
            if (!load_prefixed_model_tensors(model_loader, clip, tensors, "cond_stage_model.transformer.text_model")) {
                throw std::runtime_error("failed to load CLIP text tensors");
            }
            log_component_loaded("clip", tensors.size());
            export_clip_runner_ops(clip, "clip", clip.model.n_token, clip_skip, true, false, tests);
            exported_any = true;
        }
        return exported_any;
    }

    if (sd_version_is_sd3(version)) {
        if (tensor_storage_has_prefix(tensor_storage_map, "text_encoders.clip_l.transformer.text_model")) {
            CLIPTextModelRunner clip_l(backend, false, tensor_storage_map,
                                       "text_encoders.clip_l.transformer.text_model",
                                       OPENAI_CLIP_VIT_L_14,
                                       false);
            std::map<std::string, ggml_tensor*> tensors;
            if (!load_prefixed_model_tensors(model_loader, clip_l, tensors, "text_encoders.clip_l.transformer.text_model")) {
                throw std::runtime_error("failed to load SD3 clip_l tensors");
            }
            log_component_loaded("clip_l", tensors.size());
            export_clip_runner_ops(clip_l, "clip_l", clip_l.model.n_token, clip_skip, true, true, tests);
            exported_any = true;
        }
        if (tensor_storage_has_prefix(tensor_storage_map, "text_encoders.clip_g.transformer.text_model")) {
            CLIPTextModelRunner clip_g(backend, false, tensor_storage_map,
                                       "text_encoders.clip_g.transformer.text_model",
                                       OPEN_CLIP_VIT_BIGG_14,
                                       false);
            std::map<std::string, ggml_tensor*> tensors;
            if (!load_prefixed_model_tensors(model_loader, clip_g, tensors, "text_encoders.clip_g.transformer.text_model")) {
                throw std::runtime_error("failed to load SD3 clip_g tensors");
            }
            log_component_loaded("clip_g", tensors.size());
            export_clip_runner_ops(clip_g, "clip_g", clip_g.model.n_token, clip_skip, true, true, tests);
            exported_any = true;
        }
        if (tensor_storage_has_prefix(tensor_storage_map, "text_encoders.t5xxl.transformer")) {
            T5Runner t5(backend, false, tensor_storage_map, "text_encoders.t5xxl.transformer");
            std::map<std::string, ggml_tensor*> tensors;
            if (!load_prefixed_model_tensors(model_loader, t5, tensors, "text_encoders.t5xxl.transformer")) {
                throw std::runtime_error("failed to load SD3 t5 tensors");
            }
            log_component_loaded("t5xxl", tensors.size());
            export_t5_runner_ops(t5, "t5xxl", 77, false, tests);
            exported_any = true;
        }
        if (!exported_any && fail_if_unsupported) {
            throw std::runtime_error("no supported SD3 text encoders found in model file");
        }
        return exported_any;
    }

    if (sd_version_is_flux(version)) {
        if (version == VERSION_OVIS_IMAGE) {
            return unsupported_llm_only();
        }

        // Chroma 是 Flux 体系里的特殊变体，文本路径更偏向 T5。
        bool is_chroma = tensor_storage_contains(tensor_storage_map, "distilled_guidance_layer.in_proj.weight");
        if (is_chroma) {
            if (!tensor_storage_has_prefix(tensor_storage_map, "text_encoders.t5xxl.transformer")) {
                if (fail_if_unsupported) {
                    throw std::runtime_error("no Chroma T5 text encoder tensors found in model file");
                }
                fprintf(stdout, "export-graph-ops: text stage skipped: no Chroma T5 tensors found\n");
                return false;
            }
            T5Runner t5(backend, false, tensor_storage_map, "text_encoders.t5xxl.transformer", false);
            std::map<std::string, ggml_tensor*> tensors;
            if (!load_prefixed_model_tensors(model_loader, t5, tensors, "text_encoders.t5xxl.transformer")) {
                throw std::runtime_error("failed to load Chroma t5 tensors");
            }
            log_component_loaded("t5xxl", tensors.size());
            export_t5_runner_ops(t5, "t5xxl", 512, false, tests);
            export_t5_runner_ops(t5, "t5xxl.masked", 512, true, tests, 512 - 8);
            return true;
        }

        if (tensor_storage_has_prefix(tensor_storage_map, "text_encoders.clip_l.transformer.text_model")) {
            CLIPTextModelRunner clip_l(backend, false, tensor_storage_map,
                                       "text_encoders.clip_l.transformer.text_model",
                                       OPENAI_CLIP_VIT_L_14,
                                       true);
            std::map<std::string, ggml_tensor*> tensors;
            if (!load_prefixed_model_tensors(model_loader, clip_l, tensors, "text_encoders.clip_l.transformer.text_model")) {
                throw std::runtime_error("failed to load Flux clip_l tensors");
            }
            log_component_loaded("clip_l", tensors.size());
            export_clip_runner_ops(clip_l, "clip_l", 77, clip_skip, false, true, tests);
            exported_any = true;
        }
        if (tensor_storage_has_prefix(tensor_storage_map, "text_encoders.t5xxl.transformer")) {
            T5Runner t5(backend, false, tensor_storage_map, "text_encoders.t5xxl.transformer");
            std::map<std::string, ggml_tensor*> tensors;
            if (!load_prefixed_model_tensors(model_loader, t5, tensors, "text_encoders.t5xxl.transformer")) {
                throw std::runtime_error("failed to load Flux t5 tensors");
            }
            log_component_loaded("t5xxl", tensors.size());
            export_t5_runner_ops(t5, "t5xxl", 256, false, tests);
            exported_any = true;
        }
        if (!exported_any && fail_if_unsupported) {
            throw std::runtime_error("no supported Flux text encoders found in model file");
        }
        return exported_any;
    }

    if (sd_version_is_flux2(version) || sd_version_is_qwen_image(version) ||
        sd_version_is_anima(version) || sd_version_is_z_image(version)) {
        return unsupported_llm_only();
    }

    if (sd_version_is_wan(version)) {
        if (!tensor_storage_has_prefix(tensor_storage_map, "text_encoders.t5xxl.transformer")) {
            if (fail_if_unsupported) {
                throw std::runtime_error("no Wan UMT5 text encoder tensors found in model file");
            }
            fprintf(stdout, "export-graph-ops: text stage skipped: no Wan UMT5 tensors found\n");
            return false;
        }
        T5Runner t5(backend, false, tensor_storage_map, "text_encoders.t5xxl.transformer", true);
        std::map<std::string, ggml_tensor*> tensors;
        if (!load_prefixed_model_tensors(model_loader, t5, tensors, "text_encoders.t5xxl.transformer")) {
            throw std::runtime_error("failed to load Wan UMT5 tensors");
        }
        log_component_loaded("umt5", tensors.size());
        export_t5_runner_ops(t5, "umt5.masked", 512, true, tests, 512 - 8);
        return true;
    }

    if (fail_if_unsupported) {
        throw std::runtime_error(std::string("unsupported text encoder family for version: ") + version_desc(version));
    }

    fprintf(stdout, "export-graph-ops: text stage skipped: unsupported text encoder family for version %s\n", version_desc(version));
    return false;
}

// 图像 VAE 编码/解码所需的占位输入集合。
struct ImageVAEInputs {
    sd::Tensor<float> image;   // 编码路径输入：图像。
    sd::Tensor<float> latent;  // 解码路径输入：潜变量。
};

// 视频 VAE 编码/解码所需的占位输入集合。
struct VideoVAEInputs {
    sd::Tensor<float> video;   // 编码路径输入：视频。
    sd::Tensor<float> latent;  // 解码路径输入：潜变量。
};

// 从候选卷积权重里推断输入通道数；找不到就退回默认值。
static int64_t find_conv_input_channels(const std::map<std::string, ggml_tensor*>& tensors,
                                        const std::vector<std::string>& suffixes,
                                        int64_t fallback) {
    for (const auto& suffix : suffixes) {
        if (const ggml_tensor* t = find_tensor_suffix(tensors, suffix)) {
            return t->ne[2];
        }
    }
    return fallback;
}

// 构造图像 VAE 的占位输入。
// image 用于走 encode 路径，latent 用于走 decode 路径。
static ImageVAEInputs make_image_vae_inputs(int64_t image_channels,
                                            int64_t latent_channels,
                                            int scale_factor) {
    // 选择一个小尺寸占位图像即可，只要能完整展开 VAE 图。
    const int64_t image_size  = 64;
    const int64_t latent_size = std::max<int64_t>(1, image_size / std::max(1, scale_factor));

    ImageVAEInputs inputs;
    inputs.image = sd::Tensor<float>({image_size, image_size, image_channels, 1});
    inputs.image.fill_(0.5f);

    inputs.latent = sd::Tensor<float>({latent_size, latent_size, latent_channels, 1});
    inputs.latent.fill_(0.0f);

    return inputs;
}

// 构造视频 VAE 的占位输入。
static VideoVAEInputs make_video_vae_inputs(int64_t image_channels,
                                            int64_t latent_channels,
                                            int scale_factor,
                                            int64_t frames = 1) {
    const int64_t image_size  = 64;
    const int64_t latent_size = std::max<int64_t>(1, image_size / std::max(1, scale_factor));

    VideoVAEInputs inputs;
    inputs.video = sd::Tensor<float>({image_size, image_size, frames, image_channels, 1});
    inputs.video.fill_(0.5f);

    inputs.latent = sd::Tensor<float>({latent_size, latent_size, frames, latent_channels, 1});
    inputs.latent.fill_(0.0f);

    return inputs;
}

// 导出标准 AutoEncoderKL 的 encode / decode 两条路径。
static void export_autoencoderkl_ops(AutoEncoderKL& model,
                                     SDVersion version,
                                     const std::map<std::string, ggml_tensor*>& tensors,
                                     const std::string& label,
                                     std::set<test_object>& tests) {
    int64_t image_channels  = find_conv_input_channels(tensors, {"encoder.conv_in.weight"}, 3);
    int64_t latent_channels = find_conv_input_channels(tensors, {"decoder.conv_in.weight"}, 4);
    if (version == VERSION_FLUX2 || version == VERSION_FLUX2_KLEIN) {
        latent_channels *= 4;
    }

    ImageVAEInputs inputs = make_image_vae_inputs(image_channels, latent_channels, model.get_scale_factor());

    // encode 路径：图像 -> 潜变量。
    model.reset_compute_ctx();
    ggml_cgraph* gf_encode   = model.build_graph(inputs.image, false);
    std::string encode_label = label + ".encode";
    extract_graph_ops(gf_encode, encode_label.c_str(), tests);

    // decode 路径：潜变量 -> 图像。
    model.reset_compute_ctx();
    ggml_cgraph* gf_decode   = model.build_graph(inputs.latent, true);
    std::string decode_label = label + ".decode";
    extract_graph_ops(gf_decode, decode_label.c_str(), tests);
}

// 导出轻量级图像自编码器（TAE）的 encode / decode 两条路径。
static void export_tiny_image_ae_ops(TinyImageAutoEncoder& model,
                                     const std::map<std::string, ggml_tensor*>& tensors,
                                     const std::string& label,
                                     std::set<test_object>& tests) {
    int64_t image_channels  = find_conv_input_channels(tensors, {"encoder.layers.0.weight"}, 3);
    int64_t latent_channels = find_conv_input_channels(tensors, {"decoder.layers.0.weight"}, 4);

    ImageVAEInputs inputs = make_image_vae_inputs(image_channels, latent_channels, model.get_scale_factor());

    model.reset_compute_ctx();
    ggml_cgraph* gf_encode   = model.build_graph(inputs.image, false);
    std::string encode_label = label + ".encode";
    extract_graph_ops(gf_encode, encode_label.c_str(), tests);

    model.reset_compute_ctx();
    ggml_cgraph* gf_decode   = model.build_graph(inputs.latent, true);
    std::string decode_label = label + ".decode";
    extract_graph_ops(gf_decode, decode_label.c_str(), tests);
}

// 导出轻量级视频自编码器的 encode / decode 两条路径。
static void export_tiny_video_ae_ops(TinyVideoAutoEncoder& model,
                                     const std::map<std::string, ggml_tensor*>& tensors,
                                     const std::string& label,
                                     std::set<test_object>& tests) {
    int64_t image_channels  = 3;
    int64_t latent_channels = find_conv_input_channels(tensors, {"decoder.0.weight", "decoder.layers.0.weight"}, 16);

    VideoVAEInputs inputs = make_video_vae_inputs(image_channels, latent_channels, model.get_scale_factor(), 1);

    // encode 路径：视频 -> 潜变量。
    model.reset_compute_ctx();
    ggml_cgraph* gf_encode   = model.build_graph(inputs.video, false);
    std::string encode_label = label + ".encode";
    extract_graph_ops(gf_encode, encode_label.c_str(), tests);

    model.reset_compute_ctx();
    ggml_cgraph* gf_decode   = model.build_graph(inputs.latent, true);
    std::string decode_label = label + ".decode";
    extract_graph_ops(gf_decode, decode_label.c_str(), tests);
}

// 导出 Wan 专用 VAE 的 encode / decode 两条路径。
static void export_wan_vae_ops(WAN::WanVAERunner& model,
                               const std::string& label,
                               std::set<test_object>& tests) {
    const int64_t image_channels  = 3;
    const int64_t latent_channels = model.get_encoder_output_channels(image_channels);
    VideoVAEInputs inputs         = make_video_vae_inputs(image_channels, latent_channels, model.get_scale_factor(), 1);

    model.reset_compute_ctx();
    ggml_cgraph* gf_encode   = model.build_graph(inputs.video, false);
    std::string encode_label = label + ".encode";
    extract_graph_ops(gf_encode, encode_label.c_str(), tests);

    model.reset_compute_ctx();
    ggml_cgraph* gf_decode   = model.build_graph(inputs.latent, true);
    std::string decode_label = label + ".decode";
    extract_graph_ops(gf_decode, decode_label.c_str(), tests);
}

// 导出 VAE 阶段。
// 既会尝试 first_stage_model，也会尝试 tae 预览自编码器。
static bool export_vae_stage(const char* model_path,
                             ggml_backend_t backend,
                             std::set<test_object>& tests,
                             bool fail_if_unsupported) {
    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(model_path)) {
        throw std::runtime_error(std::string("failed to load model from '") + model_path + "'");
    }

    auto& tensor_storage_map = model_loader.get_tensor_storage_map();
    SDVersion version        = model_loader.get_sd_version();
    fprintf(stdout, "export-graph-ops: vae stage model version = %d (%s)\n", (int)version, version_desc(version));

    bool exported_any = false;

    // 对当前模型文件里“找不到可导出的 VAE”或“该版本根本没有真实 VAE 图”做统一处理。
    auto unsupported = [&](const std::string& msg) {
        if (fail_if_unsupported) {
            throw std::runtime_error(msg);
        }
        fprintf(stdout, "export-graph-ops: vae stage skipped: %s\n", msg.c_str());
        return false;
    };

    if (version == VERSION_CHROMA_RADIANCE) {
        return unsupported("Chroma Radiance uses FakeVAE, which has no GGML graph to export");
    }

    if (tensor_storage_has_prefix(tensor_storage_map, "first_stage_model")) {
        std::map<std::string, ggml_tensor*> tensors;

        if (sd_version_is_wan(version) || sd_version_is_qwen_image(version) || sd_version_is_anima(version)) {
            WAN::WanVAERunner model(backend, false, tensor_storage_map, "first_stage_model", false, version);
            if (!load_prefixed_model_tensors(model_loader, model, tensors, "first_stage_model")) {
                throw std::runtime_error("failed to load first_stage_model VAE tensors");
            }
            log_component_loaded("vae", tensors.size());
            export_wan_vae_ops(model, model.get_desc(), tests);
            exported_any = true;
        } else {
            AutoEncoderKL model(backend, false, tensor_storage_map, "first_stage_model", false, false, version);
            if (!load_prefixed_model_tensors(model_loader, model, tensors, "first_stage_model")) {
                throw std::runtime_error("failed to load first_stage_model VAE tensors");
            }
            log_component_loaded("vae", tensors.size());
            export_autoencoderkl_ops(model, version, tensors, model.get_desc(), tests);
            exported_any = true;
        }
    }

    if (tensor_storage_has_prefix(tensor_storage_map, "tae")) {
        std::map<std::string, ggml_tensor*> tensors;
        if (sd_version_is_wan(version) || sd_version_is_qwen_image(version) || sd_version_is_anima(version)) {
            TinyVideoAutoEncoder model(backend, false, tensor_storage_map, "decoder", false, version);
            if (!load_prefixed_model_tensors(model_loader, model, tensors, "tae")) {
                throw std::runtime_error("failed to load tae preview tensors");
            }
            log_component_loaded("tae", tensors.size());
            export_tiny_video_ae_ops(model, tensors, model.get_desc(), tests);
            exported_any = true;
        } else {
            TinyImageAutoEncoder model(backend, false, tensor_storage_map, "decoder.layers", false, version);
            if (!load_prefixed_model_tensors(model_loader, model, tensors, "tae")) {
                throw std::runtime_error("failed to load tae preview tensors");
            }
            log_component_loaded("tae", tensors.size());
            export_tiny_image_ae_ops(model, tensors, model.get_desc(), tests);
            exported_any = true;
        }
    }

    if (!exported_any) {
        return unsupported("no supported VAE tensors found in model file");
    }

    return true;
}

// 程序入口：解析参数、初始化后端、按阶段导出、最后写出结果文件。
int main(int argc, char** argv) {
    const char* model_path  = nullptr;
    const char* output_file = "tests.txt";
    ExportStage stage       = ExportStage::DIFFUSION;

    // 非常轻量的命令行解析：
    // -m <model_path>   指定模型文件
    // -o <output_file>  指定输出文件
    // -s / --stage      指定导出阶段
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--stage") == 0) && i + 1 < argc) {
            stage = parse_stage(argv[++i]);
        }
    }

    if (model_path == nullptr) {
        fprintf(stderr, "Usage: %s -m <model_path> [-o output_file] [--stage diffusion|text|vae|all]\n", argv[0]);
        return 1;
    }

    fprintf(stdout, "export-graph-ops: loading model from '%s'\n", model_path);
    fprintf(stdout, "export-graph-ops: stage = %s\n", stage_desc(stage));

    // 这里只需要一个能成功构图的后端；CPU backend 最稳妥。
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        fprintf(stderr, "failed to initialize CPU backend\n");
        return 1;
    }

    int rc = 0;

    try {
        // 这里保存全程序最终产出的“唯一算子样本”。
        std::set<test_object> tests;

        if (includes_diffusion(stage)) {
            export_diffusion_stage(model_path, backend, tests);
        }

        if (includes_text(stage)) {
            bool fail_if_unsupported = (stage == ExportStage::TEXT);
            export_text_stage(model_path, backend, tests, fail_if_unsupported);
        }

        if (includes_vae(stage)) {
            bool fail_if_unsupported = (stage == ExportStage::VAE);
            export_vae_stage(model_path, backend, tests, fail_if_unsupported);
        }

        // 统计全阶段合并后的唯一算子数，并写出到文本文件。
        fprintf(stdout, "export-graph-ops: %d unique ops total\n", (int)tests.size());
        write_tests(output_file, tests);
        fprintf(stdout, "export-graph-ops: see %s for output\n", output_file);
    } catch (const std::exception& e) {
        fprintf(stderr, "export-graph-ops: exception: %s\n", e.what());
        rc = 1;
    } catch (...) {
        fprintf(stderr, "export-graph-ops: unknown exception\n");
        rc = 1;
    }

    // 释放后端资源，避免泄漏。
    ggml_backend_free(backend);
    return rc;
}