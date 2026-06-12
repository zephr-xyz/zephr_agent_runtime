// tinylog.hpp — header-only structured logging, metrics, and snapshot testing.
//
// Zero dependencies beyond C++20 stdlib. Four output formats (json, logfmt, human, gcp).
// Metrics and logs share one event type and one pipeline.
// Format-compatible with the Python tinylog implementation.
//
// Usage:
//   #include "tinylog.hpp"
//
//   tinylog::Logger log;
//   log.add_sink(std::cerr, tinylog::Format::Human);
//   log.info("request processed", {{"latency_ms", 42}, {"user", "alice"}});
//   log.metric("request_latency", 42, "ms");
//
//   // with bound fields:
//   {
//       auto ctx = log.bind({{"request_id", "abc-123"}});
//       log.info("stage1 complete");  // includes request_id
//   }
//
//   // snapshot recording:
//   tinylog::Recorder rec;
//   log.add_sink(rec);
//   // ... exercise code ...
//   rec.save("snapshots/test.jsonl");

#pragma once

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <source_location>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <os/log.h>
#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif
#endif

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace tinylog {

// ---------------------------------------------------------------------------
// Level
// ---------------------------------------------------------------------------

enum class Level : int {
    Trace = 5,
    Debug = 10,
    Info = 20,
    Warn = 30,
    Error = 40,
};

inline const char* level_str(Level l) {
    switch (l) {
        case Level::Trace: return "trace";
        case Level::Debug: return "debug";
        case Level::Info:  return "info";
        case Level::Warn:  return "warn";
        case Level::Error: return "error";
    }
    return "unknown";
}

inline const char* level_str_upper(Level l) {
    switch (l) {
        case Level::Trace: return "TRACE";
        case Level::Debug: return "DEBUG";
        case Level::Info:  return "INFO";
        case Level::Warn:  return "WARN";
        case Level::Error: return "ERROR";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Value — variant for structured fields
// ---------------------------------------------------------------------------

using Value = std::variant<std::string, int64_t, double, bool, std::nullptr_t>;
using Fields = std::vector<std::pair<std::string, Value>>;

inline std::string value_to_string(const Value& v) {
    return std::visit([](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) return arg;
        else if constexpr (std::is_same_v<T, int64_t>) return std::to_string(arg);
        else if constexpr (std::is_same_v<T, double>) {
            std::ostringstream oss;
            oss << arg;
            return oss.str();
        }
        else if constexpr (std::is_same_v<T, bool>) return arg ? "true" : "false";
        else if constexpr (std::is_same_v<T, std::nullptr_t>) return "null";
        else return "?";
    }, v);
}

inline std::string value_to_json(const Value& v) {
    return std::visit([](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) {
            std::string out = "\"";
            for (char c : arg) {
                if (c == '"') out += "\\\"";
                else if (c == '\\') out += "\\\\";
                else if (c == '\n') out += "\\n";
                else out += c;
            }
            out += '"';
            return out;
        }
        else if constexpr (std::is_same_v<T, int64_t>) return std::to_string(arg);
        else if constexpr (std::is_same_v<T, double>) {
            std::ostringstream oss;
            oss << arg;
            return oss.str();
        }
        else if constexpr (std::is_same_v<T, bool>) return arg ? "true" : "false";
        else if constexpr (std::is_same_v<T, std::nullptr_t>) return "null";
        else return "null";
    }, v);
}

// ---------------------------------------------------------------------------
// Caller
// ---------------------------------------------------------------------------

struct Caller {
    const char* file = "";
    int line = 0;
    const char* function = "";
    std::vector<std::string> chain;  // optional call chain (outermost→innermost)

    Caller() = default;
    Caller(std::source_location loc)
        : file(loc.file_name()), line(static_cast<int>(loc.line())), function(loc.function_name()) {}
    Caller(const char* f, int l, const char* fn, std::vector<std::string> ch = {})
        : file(f), line(l), function(fn), chain(std::move(ch)) {}

    std::string short_str() const {
        std::string f(file);
        auto slash = f.rfind('/');
        if (slash != std::string::npos) f = f.substr(slash + 1);
        auto dot = f.rfind('.');
        if (dot != std::string::npos) f = f.substr(0, dot);
        return f + ":" + std::to_string(line);
    }

    std::string chain_str() const {
        std::string out;
        for (size_t i = 0; i < chain.size(); i++) {
            if (i > 0) out += "->";
            out += chain[i];
        }
        return out;
    }

    std::string str() const {
        if (!chain.empty()) return chain_str();
        std::string fn(function);
        if (!fn.empty() && fn != "main") return fn;
        return short_str();
    }
};

// ---------------------------------------------------------------------------
// Timestamp
// ---------------------------------------------------------------------------

inline std::string format_ts(double ts) {
    auto secs = static_cast<time_t>(ts);
    int millis = static_cast<int>((ts - secs) * 1000);
    struct tm utc;
#ifdef _WIN32
    gmtime_s(&utc, &secs);
#else
    gmtime_r(&secs, &utc);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
                  utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday,
                  utc.tm_hour, utc.tm_min, utc.tm_sec, millis);
    return buf;
}

inline double now() {
    auto tp = std::chrono::system_clock::now();
    auto dur = tp.time_since_epoch();
    return std::chrono::duration<double>(dur).count();
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

struct Event {
    double ts;
    Level level;
    std::string msg;
    Fields fields;
    Caller caller;
    std::string kind = "log";  // "log" or "metric"
    std::string metric_name;
    double metric_value = 0.0;
    std::string metric_unit;
    std::string exception;  // empty = no exception
    int seq = -1;           // deterministic mode sequence (-1 = not set)
};

// Capture the current exception's what() string (call inside a catch block)
inline std::string capture_exception() {
    try {
        auto eptr = std::current_exception();
        if (!eptr) return {};
        std::rethrow_exception(eptr);
    } catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "unknown exception";
    }
    return {};
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

enum class Format { Json, Logfmt, Human, Gcp };

inline bool human_color_enabled() {
#if defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_TV || TARGET_OS_WATCH)
    return false;
#else
    return true;
#endif
}

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    out += '"';
    return out;
}

inline std::string format_json(const Event& e) {
    std::string out = "{";
    if (e.seq >= 0)
        out += "\"seq\":" + std::to_string(e.seq);
    else
        out += "\"ts\":" + json_escape(format_ts(e.ts));
    out += ",\"level\":" + json_escape(level_str(e.level));
    out += ",\"msg\":" + json_escape(e.msg);
    out += ",\"caller\":" + json_escape(e.caller.str());
    out += ",\"kind\":" + json_escape(e.kind);
    if (e.kind == "metric") {
        out += ",\"metric_name\":" + json_escape(e.metric_name);
        out += ",\"metric_value\":" + std::to_string(e.metric_value);
        if (!e.metric_unit.empty())
            out += ",\"metric_unit\":" + json_escape(e.metric_unit);
    }
    for (auto& [k, v] : e.fields) {
        out += "," + json_escape(k) + ":" + value_to_json(v);
    }
    if (!e.exception.empty())
        out += ",\"exception\":" + json_escape(e.exception);
    out += "}";
    return out;
}

inline std::string logfmt_escape(const std::string& s) {
    if (s.find(' ') != std::string::npos || s.find('"') != std::string::npos ||
        s.find('=') != std::string::npos || s.empty()) {
        std::string out = "\"";
        for (char c : s) {
            if (c == '"') out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else if (c == '\n') out += "\\n";
            else out += c;
        }
        return out + "\"";
    }
    return s;
}

inline void append_logfmt_kv(std::string& out, const std::string& key, const std::string& value) {
    if (!out.empty()) out += " ";
    out += key + "=" + logfmt_escape(value);
}

inline std::string format_logfmt(const Event& e) {
    std::string out;
    if (e.seq >= 0)
        out += "seq=" + std::to_string(e.seq);
    else
        out += "ts=" + format_ts(e.ts);
    out += " level=" + std::string(level_str(e.level));
    out += " caller=" + e.caller.str();
    out += " msg=" + logfmt_escape(e.msg);
    out += " kind=" + e.kind;
    if (e.kind == "metric") {
        out += " metric_name=" + logfmt_escape(e.metric_name);
        out += " metric_value=" + std::to_string(e.metric_value);
        if (!e.metric_unit.empty())
            out += " metric_unit=" + logfmt_escape(e.metric_unit);
    }
    for (auto& [k, v] : e.fields) {
        out += " " + k + "=" + logfmt_escape(value_to_string(v));
    }
    if (!e.exception.empty())
        out += " exception=" + logfmt_escape(e.exception);
    return out;
}

inline std::string format_oslog_text(const Event& e) {
    std::string source;
    append_logfmt_kv(source, "source", std::string(e.caller.file) + ":" + std::to_string(e.caller.line));
    if (!e.caller.chain.empty()) {
        append_logfmt_kv(source, "chain", e.caller.chain_str());
    }

    std::string details;
    if (e.seq >= 0) {
        append_logfmt_kv(details, "seq", std::to_string(e.seq));
    }
    if (e.kind == "metric") {
        append_logfmt_kv(details, "metric_name", e.metric_name);
        append_logfmt_kv(details, "metric_value", std::to_string(e.metric_value));
        if (!e.metric_unit.empty()) append_logfmt_kv(details, "metric_unit", e.metric_unit);
    }
    for (auto& [k, v] : e.fields) {
        append_logfmt_kv(details, k, value_to_string(v));
    }
    if (!e.exception.empty()) {
        append_logfmt_kv(details, "exception", e.exception);
    }

    std::string out = e.msg;
    if (!source.empty()) out += "\n" + source;
    if (!details.empty()) out += "\n" + details;
    return out;
}

inline std::string format_human(const Event& e) {
    // ANSI colors
    const bool use_color = human_color_enabled();
    const char* RESET = use_color ? "\033[0m" : "";
    const char* DIM = use_color ? "\033[2m" : "";
    auto color = [](Level l) -> const char* {
        if (!human_color_enabled()) return "";
        switch (l) {
            case Level::Trace: return "\033[37m";
            case Level::Debug: return "\033[36m";
            case Level::Info:  return "\033[32m";
            case Level::Warn:  return "\033[33m";
            case Level::Error: return "\033[31m";
        }
        return "";
    };

    const char* c = color(e.level);

    std::string time_part;
    if (e.seq >= 0) {
        time_part = "#" + std::to_string(e.seq);
    } else {
        std::string ts = format_ts(e.ts);
        auto t_pos = ts.find('T');
        time_part = (t_pos != std::string::npos) ? ts.substr(t_pos + 1) : ts;
        if (!time_part.empty() && time_part.back() == 'Z') time_part.pop_back();
    }

    char level_buf[8];
    std::snprintf(level_buf, sizeof(level_buf), "%-5s", level_str_upper(e.level));

    std::string loc;
    if (e.seq >= 0) {
        // deterministic: function/chain only
        loc = e.caller.str();
    } else {
        loc = std::string(e.caller.file) + ":" + std::to_string(e.caller.line);
        if (!e.caller.chain.empty()) {
            loc += ":" + e.caller.chain_str();
        } else {
            std::string fn(e.caller.function);
            if (!fn.empty()) loc += ":" + fn;
        }
    }

    std::string out;
    out += std::string(DIM) + time_part + RESET + " "
         + c + level_buf + RESET + " "
         + DIM + loc + RESET + "\n";
    out += std::string("  ") + c + e.msg + RESET;

    if (e.kind == "metric") {
        out += std::string("\n  ") + DIM + "metric:" + RESET + " "
             + e.metric_name + "=" + std::to_string(e.metric_value);
        if (!e.metric_unit.empty()) out += " " + e.metric_unit;
    }

    if (!e.fields.empty()) {
        out += "\n  " + std::string(DIM);
        bool first = true;
        for (auto& [k, v] : e.fields) {
            if (!first) out += " ";
            out += k + "=" + value_to_string(v);
            first = false;
        }
        out += RESET;
    }

    if (!e.exception.empty()) {
        // Indent each line of the exception
        std::istringstream iss(e.exception);
        std::string eline;
        while (std::getline(iss, eline)) {
            out += std::string("\n  ") + DIM + eline + RESET;
        }
    }
    return out;
}

inline std::string format_gcp(const Event& e) {
    std::string severity = level_str_upper(e.level);
    if (severity == "TRACE") severity = "DEBUG";
    else if (severity == "WARN") severity = "WARNING";

    auto secs = static_cast<int64_t>(e.ts);
    auto nanos = static_cast<int64_t>((e.ts - secs) * 1e9);

    std::string out = "{";
    out += "\"severity\":" + json_escape(severity);
    out += ",\"timestamp\":{\"seconds\":" + std::to_string(secs)
         + ",\"nanos\":" + std::to_string(nanos) + "}";
    out += ",\"logging.googleapis.com/sourceLocation\":{"
           "\"file\":" + json_escape(e.caller.file)
         + ",\"line\":\"" + std::to_string(e.caller.line) + "\""
         + ",\"function\":" + json_escape(e.caller.function) + "}";
    out += ",\"message\":" + json_escape(e.msg);
    if (!e.fields.empty()) {
        out += ",\"extra\":{";
        bool first = true;
        for (auto& [k, v] : e.fields) {
            if (!first) out += ",";
            out += json_escape(k) + ":" + value_to_json(v);
            first = false;
        }
        out += "}";
    }
    if (e.kind == "metric") {
        out += ",\"metric\":{\"name\":" + json_escape(e.metric_name)
             + ",\"value\":" + std::to_string(e.metric_value);
        if (!e.metric_unit.empty())
            out += ",\"unit\":" + json_escape(e.metric_unit);
        out += "}";
    }
    if (!e.exception.empty())
        out += ",\"exception\":" + json_escape(e.exception);
    out += "}";
    return out;
}

inline std::string format_event(const Event& e, Format fmt) {
    switch (fmt) {
        case Format::Json:   return format_json(e);
        case Format::Logfmt: return format_logfmt(e);
        case Format::Human:  return format_human(e);
        case Format::Gcp:    return format_gcp(e);
    }
    return format_json(e);
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

class Recorder {
public:
    explicit Recorder(const std::string& path = "", bool deterministic = false)
        : path_(path), deterministic_(deterministic) {}

    void operator()(const Event& e) {
        std::lock_guard<std::mutex> lock(mu_);
        if (deterministic_) {
            events_.push_back(normalize(e));
        } else {
            events_.push_back(e);
        }
    }

    const std::vector<Event>& events() const { return events_; }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        events_.clear();
        committed_ = 0;
    }

    /// Append uncommitted events to the JSONL file.
    bool commit(const std::string& path = "") {
        auto p = path.empty() ? path_ : path;
        if (p.empty()) return false;
        std::ofstream f(p, std::ios::app);
        if (!f) return false;
        std::lock_guard<std::mutex> lock(mu_);
        for (size_t i = committed_; i < events_.size(); i++) {
            f << format_json(events_[i]) << "\n";
        }
        committed_ = events_.size();
        return true;
    }

    /// Write all captured events as JSONL (overwrites).
    bool save(const std::string& path) {
        std::ofstream f(path);
        if (!f) return false;
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& e : events_) {
            f << format_json(e) << "\n";
        }
        committed_ = events_.size();
        return true;
    }

private:
    Event normalize(const Event& e) {
        Event out = e;
        out.seq = seq_++;
        out.ts = 0.0;
        out.caller = Caller{"", 0, e.caller.function, e.caller.chain};
        if (!out.exception.empty()) {
            // Keep only the last line (the exception itself, not the traceback)
            auto pos = out.exception.rfind('\n');
            if (pos != std::string::npos)
                out.exception = out.exception.substr(pos + 1);
        }
        return out;
    }

    std::mutex mu_;
    std::vector<Event> events_;
    std::string path_;
    size_t committed_ = 0;
    bool deterministic_ = false;
    int seq_ = 0;
};

// ---------------------------------------------------------------------------
// Sink
// ---------------------------------------------------------------------------

using RawSink = std::function<void(const Event&)>;
using TextSink = std::function<void(const std::string&)>;

struct Sink {
    enum class Kind { Text, Raw };
    Kind kind;
    TextSink text_fn;
    RawSink raw_fn;
    Format format = Format::Human;
    Level level = Level::Trace;

    void emit(const Event& e) const {
        if (static_cast<int>(e.level) < static_cast<int>(level)) return;
        if (kind == Kind::Raw) {
            raw_fn(e);
        } else {
            text_fn(format_event(e, format) + "\n");
        }
    }
};

// ---------------------------------------------------------------------------
// BindGuard — RAII bound fields
// ---------------------------------------------------------------------------

class Logger;  // forward
class OptLogger;  // forward

class BindGuard {
public:
    BindGuard(Logger& logger, Fields fields);
    ~BindGuard();
    BindGuard(const BindGuard&) = delete;
    BindGuard& operator=(const BindGuard&) = delete;
    BindGuard(BindGuard&& other) noexcept
        : logger_(other.logger_), fields_(std::move(other.fields_)), active_(other.active_) {
        other.active_ = false;
    }

private:
    Logger& logger_;
    Fields fields_;
    bool active_ = true;
};

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

class Logger {
public:
    Logger() = default;

    void set_level(Level level) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& s : sinks_) s.level = level;
    }

    void add_sink(std::ostream& os, Format fmt = Format::Human, Level level = Level::Trace) {
        std::lock_guard<std::mutex> lock(mu_);
        auto* ptr = &os;
        sinks_.push_back(Sink{
            Sink::Kind::Text,
            [ptr](const std::string& s) { *ptr << s; ptr->flush(); },
            {},
            fmt,
            level,
        });
    }

    void add_sink(Recorder& rec, Level level = Level::Trace) {
        std::lock_guard<std::mutex> lock(mu_);
        auto* ptr = &rec;
        sinks_.push_back(Sink{
            Sink::Kind::Raw,
            {},
            [ptr](const Event& e) { (*ptr)(e); },
            Format::Json,
            level,
        });
    }

    void add_sink(RawSink fn, Level level = Level::Trace) {
        std::lock_guard<std::mutex> lock(mu_);
        sinks_.push_back(Sink{Sink::Kind::Raw, {}, std::move(fn), Format::Json, level});
    }

    void add_sink(TextSink fn, Format fmt = Format::Human, Level level = Level::Trace) {
        std::lock_guard<std::mutex> lock(mu_);
        sinks_.push_back(Sink{Sink::Kind::Text, std::move(fn), {}, fmt, level});
    }

    // Logging methods — caller location captured automatically via std::source_location
    void log(Level level, const std::string& msg, Fields fields = {},
             std::source_location loc = std::source_location::current()) {
        emit(Event{
            .ts = now(), .level = level, .msg = msg, .fields = merge_fields(std::move(fields)),
            .caller = Caller(loc), .kind = "log",
        });
    }

    void metric(const std::string& name, double value, const std::string& unit,
                Fields fields = {},
                std::source_location loc = std::source_location::current()) {
        emit(Event{
            .ts = now(), .level = Level::Info, .msg = name, .fields = merge_fields(std::move(fields)),
            .caller = Caller(loc), .kind = "metric", .metric_name = name, .metric_value = value, .metric_unit = unit,
        });
    }

    void trace(const std::string& msg, Fields f = {}, std::source_location loc = std::source_location::current()) { log(Level::Trace, msg, std::move(f), loc); }
    void debug(const std::string& msg, Fields f = {}, std::source_location loc = std::source_location::current()) { log(Level::Debug, msg, std::move(f), loc); }
    void info(const std::string& msg, Fields f = {}, std::source_location loc = std::source_location::current())  { log(Level::Info, msg, std::move(f), loc); }
    void warn(const std::string& msg, Fields f = {}, std::source_location loc = std::source_location::current())  { log(Level::Warn, msg, std::move(f), loc); }
    void error(const std::string& msg, Fields f = {}, std::source_location loc = std::source_location::current()) { log(Level::Error, msg, std::move(f), loc); }

    BindGuard bind(Fields fields) { return BindGuard(*this, std::move(fields)); }

    OptLogger opt(bool exception = false);

private:
    friend class BindGuard;
    friend class OptLogger;

    void push_fields(const Fields& fields) {
        std::lock_guard<std::mutex> lock(mu_);
        bound_stack_.push_back(fields);
    }

    void pop_fields() {
        std::lock_guard<std::mutex> lock(mu_);
        if (!bound_stack_.empty()) bound_stack_.pop_back();
    }

    Fields merge_fields(Fields event_fields) {
        Fields merged;
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (auto& layer : bound_stack_) {
                for (auto& kv : layer) merged.push_back(kv);
            }
        }
        for (auto& kv : event_fields) merged.push_back(std::move(kv));
        return merged;
    }

    void emit(Event e) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& s : sinks_) {
            try {
                s.emit(e);
            } catch (...) {
                // never let logging crash the app
            }
        }
    }

    std::mutex mu_;
    std::vector<Sink> sinks_;
    std::vector<Fields> bound_stack_;
};

// BindGuard implementation (needs Logger to be complete)
inline BindGuard::BindGuard(Logger& logger, Fields fields)
    : logger_(logger), fields_(std::move(fields)) {
    logger_.push_fields(fields_);
}

inline BindGuard::~BindGuard() {
    if (active_) logger_.pop_fields();
}

// ---------------------------------------------------------------------------
// OptLogger — wraps Logger with capture options (exception, etc.)
// ---------------------------------------------------------------------------

class OptLogger {
public:
    OptLogger(Logger& logger, bool exception)
        : logger_(logger), exception_(exception) {}

    void log(Level level, const std::string& msg, Fields fields = {},
             std::source_location loc = std::source_location::current()) {
        Event e{
            .ts = now(), .level = level, .msg = msg, .fields = logger_.merge_fields(std::move(fields)),
            .caller = Caller(loc), .kind = "log",
        };
        if (exception_) e.exception = capture_exception();
        logger_.emit(e);
    }

    void metric(const std::string& name, double value, const std::string& unit,
                Fields fields = {},
                std::source_location loc = std::source_location::current()) {
        Event e{
            .ts = now(), .level = Level::Info, .msg = name, .fields = logger_.merge_fields(std::move(fields)),
            .caller = Caller(loc), .kind = "metric", .metric_name = name, .metric_value = value, .metric_unit = unit,
        };
        if (exception_) e.exception = capture_exception();
        logger_.emit(e);
    }

    void trace(const std::string& msg, Fields f = {}, std::source_location loc = std::source_location::current()) { log(Level::Trace, msg, std::move(f), loc); }
    void debug(const std::string& msg, Fields f = {}, std::source_location loc = std::source_location::current()) { log(Level::Debug, msg, std::move(f), loc); }
    void info(const std::string& msg, Fields f = {}, std::source_location loc = std::source_location::current())  { log(Level::Info, msg, std::move(f), loc); }
    void warn(const std::string& msg, Fields f = {}, std::source_location loc = std::source_location::current())  { log(Level::Warn, msg, std::move(f), loc); }
    void error(const std::string& msg, Fields f = {}, std::source_location loc = std::source_location::current()) { log(Level::Error, msg, std::move(f), loc); }

private:
    Logger& logger_;
    bool exception_;
};

inline OptLogger Logger::opt(bool exception) {
    return OptLogger(*this, exception);
}

// ---------------------------------------------------------------------------
// Platform sinks
// ---------------------------------------------------------------------------

inline bool apple_unified_logging_enabled() {
#if defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_TV || TARGET_OS_WATCH || TARGET_OS_VISION || TARGET_OS_MACCATALYST)
    return true;
#else
    return false;
#endif
}

#if defined(__APPLE__)
inline os_log_type_t apple_log_type(Level level) {
    switch (level) {
        case Level::Trace: return OS_LOG_TYPE_DEBUG;
        case Level::Debug: return OS_LOG_TYPE_DEBUG;
        case Level::Info:  return OS_LOG_TYPE_INFO;
        case Level::Warn:  return OS_LOG_TYPE_DEFAULT;
        case Level::Error: return OS_LOG_TYPE_ERROR;
    }
    return OS_LOG_TYPE_DEFAULT;
}

inline void add_apple_unified_logging_sink(Logger& logger,
                                           const char* subsystem,
                                           const char* category,
                                           Level level = Level::Trace) {
    os_log_t oslog = os_log_create(subsystem, category);
    logger.add_sink([oslog](const Event& e) {
        std::string line = format_oslog_text(e);
        os_log_with_type(oslog, apple_log_type(e.level), "%{public}s", line.c_str());
    }, level);
}
#endif

#if defined(__ANDROID__)
inline int android_log_priority(Level level) {
    switch (level) {
        case Level::Trace: return ANDROID_LOG_VERBOSE;
        case Level::Debug: return ANDROID_LOG_DEBUG;
        case Level::Info:  return ANDROID_LOG_INFO;
        case Level::Warn:  return ANDROID_LOG_WARN;
        case Level::Error: return ANDROID_LOG_ERROR;
    }
    return ANDROID_LOG_INFO;
}

inline void add_android_log_sink(Logger& logger,
                                 const char* subsystem,
                                 const char* category,
                                 Level level = Level::Trace) {
    std::string tag = (category && category[0])
        ? category
        : ((subsystem && subsystem[0]) ? subsystem : "tinylog");
    logger.add_sink([tag = std::move(tag)](const Event& e) {
        std::string line = format_logfmt(e);
        __android_log_print(android_log_priority(e.level), tag.c_str(), "%s", line.c_str());
    }, level);
}
#endif

inline void add_platform_default_sink(Logger& logger,
                                      const char* subsystem,
                                      const char* category,
                                      Level level = Level::Trace) {
#if defined(__ANDROID__)
    add_android_log_sink(logger, subsystem, category, level);
#elif defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_TV || TARGET_OS_WATCH || TARGET_OS_VISION || TARGET_OS_MACCATALYST)
    add_apple_unified_logging_sink(logger, subsystem, category, level);
#else
    (void)subsystem;
    (void)category;
    logger.add_sink(std::cerr, Format::Human, level);
#endif
}

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Environment-based level
// ---------------------------------------------------------------------------

inline Level level_from_env(Level fallback = Level::Info) {
    const char* val = std::getenv("TINYLOG_LEVEL");
    if (!val) return fallback;
    std::string s(val);
    for (auto& c : s) c = (char)std::toupper((unsigned char)c);
    if (s == "TRACE") return Level::Trace;
    if (s == "DEBUG") return Level::Debug;
    if (s == "INFO")  return Level::Info;
    if (s == "WARN")  return Level::Warn;
    if (s == "ERROR") return Level::Error;
    return fallback;
}

// ---------------------------------------------------------------------------
// Default global logger
// ---------------------------------------------------------------------------

namespace detail {
inline Logger*& default_logger_ptr() {
    static Logger* instance = nullptr;
    return instance;
}
}  // namespace detail

inline void set_default_logger(Logger* l) { detail::default_logger_ptr() = l; }
inline Logger& logger() {
    if (Logger* l = detail::default_logger_ptr())
        return *l;
    static Logger fallback;
    return fallback;
}

}  // namespace tinylog
