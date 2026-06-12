// TinyLLM grammar-constrained decoding — model-specific FC format state machines.
//
// Provides:
//   - Shared constraint types (ConstrainedParam, GrammarConstraint, ConstraintType)
//   - FcGrammar: constrained function-call format
//       <start_function_call>call:TOOL{p:<escape>V<escape>}<end_function_call>
//   - Gemma4Grammar: Gemma 4 format
//       <|tool_call>call:TOOL{p:<|"|>V<|"|>}<tool_call|>
//   - generate_constrained<Grammar>(): templated decode loop
//
#pragma once

#include "tinyllm_core.hpp"

// MARK: - Shared constraint types

struct ConstrainedParam {
    std::string name;
    std::vector<std::string> enum_values;  // empty = free text
    bool is_enum() const { return !enum_values.empty(); }
};

enum class ConstraintType { SINGLE_TOKEN, TOKEN_SET, FREE_TEXT };

struct GrammarConstraint {
    ConstraintType type;
    int single_token_id = -1;               // SINGLE_TOKEN
    std::vector<int> valid_ids;             // TOKEN_SET
    std::string param_name;                 // FREE_TEXT
    int step_in_value = 0;                  // FREE_TEXT
};

struct ConstrainedResult {
    std::string response;                      // Full FC string
    std::map<std::string, std::string> params; // Extracted param values
    std::vector<TokenSetLogEntry> logs;
    int decode_steps = 0;
    int target_decode_calls = 0;
    int verify_batches = 0;
    int verify_fixed_tokens = 0;
    int verify_rows = 0;
    int64_t target_decode_ms = 0;
    int64_t verify_ms = 0;
    int64_t first_decode_ms = 0;
};

static inline bool constraint_fixed_token(const GrammarConstraint& constraint, int* token_id) {
    if (constraint.type == ConstraintType::SINGLE_TOKEN) {
        if (token_id) *token_id = constraint.single_token_id;
        return constraint.single_token_id >= 0;
    }
    if (constraint.type == ConstraintType::TOKEN_SET &&
        constraint.valid_ids.size() == 1) {
        if (token_id) *token_id = constraint.valid_ids[0];
        return constraint.valid_ids[0] >= 0;
    }
    return false;
}

template<typename Grammar>
static inline std::vector<int> peek_fixed_tokens(Grammar grammar, int max_tokens) {
    std::vector<int> tokens;
    for (int i = 0; i < max_tokens && !grammar.is_done(); i++) {
        int token_id = -1;
        if (!constraint_fixed_token(grammar.get_constraint(), &token_id))
            break;
        tokens.push_back(token_id);
        grammar.advance(token_id);
    }
    return tokens;
}

template<typename Grammar>
static inline bool advance_fixed_token(Grammar& grammar, int token_id) {
    int expected = -1;
    if (!constraint_fixed_token(grammar.get_constraint(), &expected))
        return false;
    if (expected != token_id)
        return false;
    grammar.advance(token_id);
    return true;
}

// MARK: - Tool definition types (shared across model families)

struct ToolParam {
    std::string name;
    std::string description;
    std::string type;  // "STRING", "INTEGER", etc.
    std::vector<std::string> enum_values;
    bool required = false;
};

struct ToolDef {
    std::string name;
    std::string description;
    std::vector<ToolParam> params;
};

// MARK: - FcGrammar — constrained function call format
// Format: <start_function_call>call:TOOL{p1:<escape>V1<escape>,...}<end_function_call>

class FcGrammar {
public:
    FcGrammar(const Tokenizer& tok, const std::string& tool_name,
              const std::vector<ConstrainedParam>& params,
              const std::set<int>& stop_tokens)
        : tok_(tok), params_(params), stop_tokens_(stop_tokens) {

        int start_fc_id = tok.piece_to_id("<start_function_call>");
        int end_fc_id = tok.piece_to_id("<end_function_call>");
        escape_id_ = tok.piece_to_id("<escape>");
        end_fc_id_ = end_fc_id;
        start_fc_id_ = start_fc_id;

        structural_.push_back(start_fc_id);
        auto call_prefix = tok.encode("call:" + tool_name + "{");
        structural_.insert(structural_.end(), call_prefix.begin(), call_prefix.end());

        for (size_t i = 0; i < params.size(); i++) {
            if (i > 0) {
                auto comma = tok.encode(",");
                structural_.insert(structural_.end(), comma.begin(), comma.end());
            }
            auto name_colon = tok.encode(params[i].name + ":");
            structural_.insert(structural_.end(), name_colon.begin(), name_colon.end());
            structural_.push_back(escape_id_);
            param_start_indices_.push_back((int)structural_.size());
            structural_.push_back(-1);  // placeholder for value
            structural_.push_back(escape_id_);
        }

        auto close = tok.encode("}");
        structural_.insert(structural_.end(), close.begin(), close.end());
        structural_.push_back(end_fc_id);

        for (auto& p : params) values_[p.name] = {};

        // Build FC string template parts for result reconstruction
        tool_name_ = tool_name;
    }

    bool is_done() const { return done_; }

    GrammarConstraint get_constraint() const {
        if (done_) return {ConstraintType::SINGLE_TOKEN, 1, {}, {}, 0};

        int current = structural_[struct_idx_];

        if (current == -1) {
            int param_idx = find_param_idx();
            const auto& param = params_[param_idx];

            if (param.is_enum()) {
                if (enum_candidates_.empty() && !enum_inited_)
                    const_cast<FcGrammar*>(this)->init_enum_candidates(param_idx);

                if (enum_candidates_.size() <= 1) {
                    if (enum_candidates_.size() == 1 &&
                        enum_step_ < (int)enum_candidates_[0].second.size())
                        return {ConstraintType::SINGLE_TOKEN,
                                enum_candidates_[0].second[enum_step_], {}, {}, 0};
                    return {ConstraintType::SINGLE_TOKEN,
                            structural_[struct_idx_ + 1], {}, {}, 0};
                }

                std::vector<int> valid;
                for (auto& [_, seq] : enum_candidates_) {
                    if ((int)seq.size() > enum_step_)
                        valid.push_back(seq[enum_step_]);
                }
                std::sort(valid.begin(), valid.end());
                valid.erase(std::unique(valid.begin(), valid.end()), valid.end());

                GrammarConstraint c;
                c.type = ConstraintType::TOKEN_SET;
                c.valid_ids = std::move(valid);
                return c;
            }

            GrammarConstraint c;
            c.type = ConstraintType::FREE_TEXT;
            c.param_name = param.name;
            c.step_in_value = value_step_count_;
            return c;
        }

        return {ConstraintType::SINGLE_TOKEN, current, {}, {}, 0};
    }

    void advance(int token_id) {
        int current = structural_[struct_idx_];

        if (current == -1) {
            int param_idx = find_param_idx();
            const auto& param = params_[param_idx];

            if (param.is_enum()) {
                if (enum_candidates_.size() <= 1 && !enum_candidates_.empty() &&
                    enum_step_ < (int)enum_candidates_[0].second.size()) {
                    values_[param.name].push_back(token_id);
                    enum_step_++;
                    return;
                }
                if (enum_candidates_.size() <= 1) { finish_value(); return; }

                values_[param.name].push_back(token_id);
                std::vector<std::pair<std::string, std::vector<int>>> remaining;
                for (auto& [val, seq] : enum_candidates_) {
                    if ((int)seq.size() > enum_step_ && seq[enum_step_] == token_id)
                        remaining.push_back({val, seq});
                }
                enum_candidates_ = std::move(remaining);
                enum_step_++;
                return;
            }

            // Free text
            if (token_id == escape_id_ || stop_tokens_.count(token_id) ||
                token_id == end_fc_id_ || token_id == start_fc_id_ ||
                value_step_count_ >= max_value_tokens_) {
                finish_value();
                return;
            }

            values_[param.name].push_back(token_id);
            value_step_count_++;
            return;
        }

        struct_idx_++;
        if (struct_idx_ >= (int)structural_.size()) done_ = true;
    }

    std::map<std::string, std::string> get_result() const {
        std::map<std::string, std::string> result;
        for (auto& param : params_) {
            auto it = values_.find(param.name);
            if (it == values_.end()) continue;
            std::string decoded = tok_.decode(it->second);
            if (param.is_enum()) {
                for (auto& ev : param.enum_values) {
                    if (ev == decoded) { result[param.name] = ev; goto next; }
                }
            }
            result[param.name] = decoded;
            next:;
        }
        return result;
    }

    std::string build_response_string(const std::map<std::string, std::string>& params) const {
        std::string fc = "<start_function_call>call:" + tool_name_ + "{";
        bool first = true;
        for (auto& [key, value] : params) {
            if (!first) fc += ",";
            fc += key + ":<escape>" + value + "<escape>";
            first = false;
        }
        fc += "}<end_function_call>";
        return fc;
    }

private:
    const Tokenizer& tok_;
    std::vector<ConstrainedParam> params_;
    std::set<int> stop_tokens_;
    std::string tool_name_;
    int escape_id_, end_fc_id_, start_fc_id_;

    std::vector<int> structural_;
    std::vector<int> param_start_indices_;
    int struct_idx_ = 0;

    bool done_ = false;
    int value_step_count_ = 0;
    static constexpr int max_value_tokens_ = 64;

    mutable std::vector<std::pair<std::string, std::vector<int>>> enum_candidates_;
    mutable int enum_step_ = 0;
    mutable bool enum_inited_ = false;
    mutable std::map<std::string, std::vector<int>> values_;

    int find_param_idx() const {
        for (size_t i = 0; i < param_start_indices_.size(); i++)
            if (param_start_indices_[i] == struct_idx_) return (int)i;
        return 0;
    }

    void init_enum_candidates(int param_idx) {
        enum_candidates_.clear();
        for (auto& v : params_[param_idx].enum_values)
            enum_candidates_.push_back({v, tok_.encode(v)});
        enum_step_ = 0;
        enum_inited_ = true;
    }

    void finish_value() {
        struct_idx_ += 2;
        value_step_count_ = 0;
        enum_candidates_.clear();
        enum_inited_ = false;
        enum_step_ = 0;
    }
};

// MARK: - Gemma4Grammar — Gemma 4 tool_call format
// Format: <|tool_call>call:TOOL{p1:<|"|>V1<|"|>,...}<tool_call|>

class Gemma4Grammar {
public:
    Gemma4Grammar(const Tokenizer& tok, const std::string& tool_name,
                  const std::vector<ConstrainedParam>& params,
                  const std::set<int>& stop_tokens)
        : tok_(tok), params_(params), stop_tokens_(stop_tokens) {

        int start_tc_id = tok.piece_to_id("<|tool_call>");
        int end_tc_id = tok.piece_to_id("<tool_call|>");
        int tool_response_id = tok.piece_to_id("<|tool_response>");
        escape_id_ = tok.piece_to_id("<|\"|>");
        end_tc_id_ = end_tc_id;
        start_tc_id_ = start_tc_id;

        structural_.push_back(start_tc_id);
        auto call_prefix = tok.encode("call:" + tool_name + "{");
        structural_.insert(structural_.end(), call_prefix.begin(), call_prefix.end());

        for (size_t i = 0; i < params.size(); i++) {
            if (i > 0) {
                auto comma = tok.encode(",");
                structural_.insert(structural_.end(), comma.begin(), comma.end());
            }
            auto name_colon = tok.encode(params[i].name + ":");
            structural_.insert(structural_.end(), name_colon.begin(), name_colon.end());
            structural_.push_back(escape_id_);
            param_start_indices_.push_back((int)structural_.size());
            structural_.push_back(-1);
            structural_.push_back(escape_id_);
        }

        auto close = tok.encode("}");
        structural_.insert(structural_.end(), close.begin(), close.end());
        structural_.push_back(end_tc_id);
        structural_.push_back(tool_response_id);

        for (auto& p : params) values_[p.name] = {};
        tool_name_ = tool_name;
    }

    bool is_done() const { return done_; }

    // Identical state machine logic to FcGrammar — only tokens differ
    GrammarConstraint get_constraint() const {
        if (done_) return {ConstraintType::SINGLE_TOKEN, 1, {}, {}, 0};

        int current = structural_[struct_idx_];

        if (current == -1) {
            int param_idx = find_param_idx();
            const auto& param = params_[param_idx];

            if (param.is_enum()) {
                if (enum_candidates_.empty() && !enum_inited_)
                    const_cast<Gemma4Grammar*>(this)->init_enum_candidates(param_idx);

                if (enum_candidates_.size() <= 1) {
                    if (enum_candidates_.size() == 1 &&
                        enum_step_ < (int)enum_candidates_[0].second.size())
                        return {ConstraintType::SINGLE_TOKEN,
                                enum_candidates_[0].second[enum_step_], {}, {}, 0};
                    return {ConstraintType::SINGLE_TOKEN,
                            structural_[struct_idx_ + 1], {}, {}, 0};
                }

                std::vector<int> valid;
                for (auto& [_, seq] : enum_candidates_) {
                    if ((int)seq.size() > enum_step_)
                        valid.push_back(seq[enum_step_]);
                }
                std::sort(valid.begin(), valid.end());
                valid.erase(std::unique(valid.begin(), valid.end()), valid.end());

                GrammarConstraint c;
                c.type = ConstraintType::TOKEN_SET;
                c.valid_ids = std::move(valid);
                return c;
            }

            GrammarConstraint c;
            c.type = ConstraintType::FREE_TEXT;
            c.param_name = param.name;
            c.step_in_value = value_step_count_;
            return c;
        }

        return {ConstraintType::SINGLE_TOKEN, current, {}, {}, 0};
    }

    void advance(int token_id) {
        int current = structural_[struct_idx_];

        if (current == -1) {
            int param_idx = find_param_idx();
            const auto& param = params_[param_idx];

            if (param.is_enum()) {
                if (enum_candidates_.size() <= 1 && !enum_candidates_.empty() &&
                    enum_step_ < (int)enum_candidates_[0].second.size()) {
                    values_[param.name].push_back(token_id);
                    enum_step_++;
                    return;
                }
                if (enum_candidates_.size() <= 1) { finish_value(); return; }

                values_[param.name].push_back(token_id);
                std::vector<std::pair<std::string, std::vector<int>>> remaining;
                for (auto& [val, seq] : enum_candidates_) {
                    if ((int)seq.size() > enum_step_ && seq[enum_step_] == token_id)
                        remaining.push_back({val, seq});
                }
                enum_candidates_ = std::move(remaining);
                enum_step_++;
                return;
            }

            if (token_id == escape_id_ || stop_tokens_.count(token_id) ||
                token_id == end_tc_id_ || token_id == start_tc_id_ ||
                value_step_count_ >= max_value_tokens_) {
                finish_value();
                return;
            }

            values_[param.name].push_back(token_id);
            value_step_count_++;
            return;
        }

        struct_idx_++;
        if (struct_idx_ >= (int)structural_.size()) done_ = true;
    }

    std::map<std::string, std::string> get_result() const {
        std::map<std::string, std::string> result;
        for (auto& param : params_) {
            auto it = values_.find(param.name);
            if (it == values_.end()) continue;
            std::string decoded = tok_.decode(it->second);
            if (param.is_enum()) {
                for (auto& ev : param.enum_values) {
                    if (ev == decoded) { result[param.name] = ev; goto next; }
                }
            }
            result[param.name] = decoded;
            next:;
        }
        return result;
    }

    std::string build_response_string(const std::map<std::string, std::string>& params) const {
        std::string fc = "<|tool_call>call:" + tool_name_ + "{";
        bool first = true;
        for (auto& [key, value] : params) {
            if (!first) fc += ",";
            fc += key + ":<|\"|>" + value + "<|\"|>";
            first = false;
        }
        fc += "}<tool_call|>";
        fc += "<|tool_response>";
        return fc;
    }

private:
    const Tokenizer& tok_;
    std::vector<ConstrainedParam> params_;
    std::set<int> stop_tokens_;
    std::string tool_name_;
    int escape_id_, end_tc_id_, start_tc_id_;

    std::vector<int> structural_;
    std::vector<int> param_start_indices_;
    int struct_idx_ = 0;

    bool done_ = false;
    int value_step_count_ = 0;
    static constexpr int max_value_tokens_ = 64;

    mutable std::vector<std::pair<std::string, std::vector<int>>> enum_candidates_;
    mutable int enum_step_ = 0;
    mutable bool enum_inited_ = false;
    mutable std::map<std::string, std::vector<int>> values_;

    int find_param_idx() const {
        for (size_t i = 0; i < param_start_indices_.size(); i++)
            if (param_start_indices_[i] == struct_idx_) return (int)i;
        return 0;
    }

    void init_enum_candidates(int param_idx) {
        enum_candidates_.clear();
        for (auto& v : params_[param_idx].enum_values)
            enum_candidates_.push_back({v, tok_.encode(v)});
        enum_step_ = 0;
        enum_inited_ = true;
    }

    void finish_value() {
        struct_idx_ += 2;
        value_step_count_ = 0;
        enum_candidates_.clear();
        enum_inited_ = false;
        enum_step_ = 0;
    }
};

// MARK: - Templated constrained generation — works with any grammar type
// Grammar must provide: is_done(), get_constraint(), advance(int),
//                       get_result(), build_response_string()
//
// decode_step_fn: bool(int32_t token_id, float* logits_out)
//   — runs one decode step, returns false on failure

template<typename Grammar, typename DecodeStepFn>
static inline ConstrainedResult generate_constrained(
    Grammar& grammar,
    DecodeStepFn decode_step_fn,
    std::mt19937& rng,
    int pending_token,
    float temperature, int top_k, float top_p,
    int diag_top_k = 8,
    std::chrono::steady_clock::time_point decode_start = std::chrono::steady_clock::now())
{
    ConstrainedResult result;
    int current_token = pending_token;
    bool intent_selected = false;

    std::vector<float> logits(VOCAB_SIZE);

    while (!grammar.is_done()) {
        auto constraint = grammar.get_constraint();

        if (!decode_step_fn(current_token, logits.data()))
            break;

        if (result.decode_steps == 0) {
            result.first_decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - decode_start).count();
        }

        std::vector<std::pair<int, float>> diag_topk_snapshot;
        if (diag_top_k > 0)
            extract_topk(logits.data(), VOCAB_SIZE, diag_top_k, diag_topk_snapshot);

        int sampled;
        int valid_count = 1;

        switch (constraint.type) {
        case ConstraintType::SINGLE_TOKEN:
            sampled = constraint.single_token_id;
            valid_count = 1;
            break;

        case ConstraintType::TOKEN_SET:
            valid_count = (int)constraint.valid_ids.size();
            sampled = constrained_sample(logits.data(), VOCAB_SIZE,
                                         constraint.valid_ids,
                                         temperature, top_k, top_p, rng);
            break;

        case ConstraintType::FREE_TEXT:
            intent_selected = true;
            sampled = sample_topk_topp(logits.data(), VOCAB_SIZE,
                                       temperature, top_k, top_p, rng);
            break;
        }

        if (!diag_topk_snapshot.empty()) {
            TokenSetLogEntry entry;
            entry.step = result.decode_steps;
            entry.valid_count = valid_count;
            entry.sampled_id = sampled;
            entry.top_logits = std::move(diag_topk_snapshot);
            result.logs.push_back(std::move(entry));
        }

        result.decode_steps++;
        grammar.advance(sampled);
        current_token = sampled;
    }

    result.params = grammar.get_result();
    result.response = grammar.build_response_string(result.params);

    return result;
}
